"""Replay the scenario over the Gemini Developer API *Interactions* endpoint.

Unlike generateContent (stateless — the client resends the history every turn),
the Interactions API keeps the conversation history on the server: the first turn
opens an interaction, later turns pass `previous_interaction_id` and send only the
new question, and the server prepends the remembered history.

What the server does NOT keep is the instruction context. `system_instruction`,
`tools`, and `generation_config` are interaction-scoped, so a stateful conversation
re-sends the system prompt on every turn. Its byte savings over the stateless arm
come from the history alone (verified: docs/interactions-api-fields.md).

This runs the plain-model path only, with `stream:false` so wire bytes and total
latency mean the same thing as on the generateContent arms — one request, one JSON
response, top-level `usage`. No agent, no sandbox/environment, no background, no
warmup, no tools.

  POST https://generativelanguage.googleapis.com/v1beta/interactions
  headers: x-goog-api-key: <key>, Content-Type: application/json
  body: {"model": "...", "stream": false, "store": true,
         "system_instruction": "...",           # re-sent every turn
         "previous_interaction_id": "...",       # absent on turn 1
         "input": [{"type":"user_input","content":[{"type":"text","text":"..."}]}]}
"""

from __future__ import annotations

import json
import os
import secrets
import time

import gemini_client as gc
from gemini_client import is_mock, wire_counter, _text_tokens
from experiment import load_request

INTERACTION_TIMEOUT = float(os.environ.get("INTERACTION_TIMEOUT", "180"))


def interactions_url() -> str:
    return f"{gc.api_base()}/interactions"


def _input(text: str) -> list:
    return [{"type": "user_input", "content": [{"type": "text", "text": text}]}]


def interaction_body(model: str, text: str, system_instruction: str,
                     prev_id: str | None) -> dict:
    """One interaction request body. system_instruction is sent on every turn
    because the server does not keep it; previous_interaction_id carries the
    history and is absent on the first turn."""
    body: dict = {
        "model": model,
        "stream": False,
        "store": True,
        "input": _input(text),
    }
    if system_instruction:
        body["system_instruction"] = system_instruction
    if prev_id:
        body["previous_interaction_id"] = prev_id
    return body


def _usage_common(usage: dict) -> dict:
    """Map the interaction response's `usage` to the fields shared across arms."""
    u = usage or {}
    return {
        "input_tokens": int(u.get("total_input_tokens", 0)),
        "cached_tokens": int(u.get("total_cached_tokens", 0)),
        "output_tokens": int(u.get("total_output_tokens", 0)),
        "thought_tokens": int(u.get("total_thought_tokens", 0)),
        "total_tokens": int(u.get("total_tokens", 0)),
    }


def _extract_text(obj) -> str:
    """Collect every {"type":"text","text":...} leaf in a payload."""
    out: list[str] = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("type") == "text" and isinstance(o.get("text"), str):
                out.append(o["text"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return "".join(out)


def _record(turn, question, text, iid, usage, wire_sent, wire_recv, elapsed_ms,
            req_raw, resp_raw, error) -> dict:
    return {
        "turn": turn, "question": question,
        "response_text": text, "interaction_id": iid,
        "wire_sent": wire_sent, "wire_recv": wire_recv, "elapsed_ms": elapsed_ms,
        **usage,
        "request_raw": req_raw, "response_raw": resp_raw,
        "error": error,
    }


def _mock_interaction(turn: int, text: str, system: str, prev_id: str | None) -> dict:
    iid = f"mock_interaction_{turn:03d}"
    ans = f"(mock interaction answer, turn {turn}, prev={prev_id or 'none'}) " + ("lorem ipsum " * 12)
    body = interaction_body("mock-model", text, system, prev_id)
    req = json.dumps(body)
    inp = _text_tokens([{"parts": [{"text": (system if turn == 1 else '') + text}]}])
    usage = {"input_tokens": inp, "cached_tokens": 0, "output_tokens": 40,
             "thought_tokens": 0, "total_tokens": inp + 40}
    resp = json.dumps({"id": iid, "status": "completed",
                       "steps": [{"type": "model_output",
                                  "content": [{"type": "text", "text": ans}]}]})
    return _record(turn, text, ans, iid, usage,
                   len(req) + 200, len(resp) + 200, 0, req, resp, "")


def _call_interaction(model: str, text: str, system_instruction: str,
                      prev_id: str | None, turn: int = 1) -> dict:
    """One interaction request. Never raises; errors land in the record's error.
    Returns the common per-turn record."""
    body = interaction_body(model, text, system_instruction, prev_id)
    req_raw = json.dumps(body)
    headers = {"Content-Type": "application/json", **gc.auth_headers()}
    t0 = time.monotonic()
    try:
        with wire_counter() as w:
            resp = gc._session().post(interactions_url(), data=req_raw,
                                      headers=headers, timeout=INTERACTION_TIMEOUT)
            _ = resp.content
    except Exception as exc:
        return _record(turn, text, "", None, _usage_common({}), 0, 0,
                       int((time.monotonic() - t0) * 1000), req_raw, "",
                       f"request_failed: {exc}")

    elapsed = int((time.monotonic() - t0) * 1000)
    resp_raw = resp.text
    if resp.status_code not in (200, 201):
        return _record(turn, text, "", None, _usage_common({}),
                       w.sent, w.recv, elapsed, req_raw, resp_raw,
                       f"http_{resp.status_code}: {resp_raw[:200]}")
    try:
        data = resp.json()
    except Exception as exc:
        return _record(turn, text, "", None, _usage_common({}),
                       w.sent, w.recv, elapsed, req_raw, resp_raw,
                       f"parse_failed: {exc}")

    iid = data.get("id")
    text_out = _extract_text(data.get("steps", data))
    usage = _usage_common(data.get("usage") or {})
    return _record(turn, text, text_out, iid, usage,
                   w.sent, w.recv, elapsed, req_raw, resp_raw, "")


def run_interaction(model: str, request_name: str = "perf",
                    turns: int | None = None, on_progress=None) -> dict:
    """Replay the scenario over the Interactions API. Turn 1 opens the interaction
    (system prompt + first question); each later turn sends the system prompt again
    plus the new question with previous_interaction_id, so the server keeps the
    history. Returns {"params": {...}, "interaction_records": [...]}."""
    system, steps, source = load_request(request_name)
    if turns:
        steps = steps[:max(1, min(turns, len(steps)))]
    n = len(steps)

    records = []
    prev_id: str | None = None
    for k, q in enumerate(steps, start=1):
        if on_progress:
            on_progress({"stage": "interaction", "turn": k, "turns": n})
        if is_mock():
            r = _mock_interaction(k, q, system, prev_id)
        else:
            r = _call_interaction(model, q, system, prev_id, turn=k)
        r["turn"] = k
        if r.get("interaction_id"):
            prev_id = r["interaction_id"]     # chain the next turn onto this one
        records.append(r)

    return {
        "params": {"mode": "interaction", "turns": n, "model": model,
                   "stream": False, "endpoint": interactions_url(),
                   "request_source": source},
        "interaction_records": records,
    }
