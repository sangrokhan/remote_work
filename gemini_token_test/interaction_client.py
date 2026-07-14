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

  With client_history (the interaction_stateless arm) the trade flips: store is
  false, previous_interaction_id is gone, and `input` carries the whole
  conversation as steps.

  body: {"model": "...", "stream": False, "store": False,
         "system_instruction": "...",           # still re-sent every turn
         "input": [{"type":"user_input","content":[{"type":"text","text":"q1"}]},
                   {"type":"model_output","content":[{"type":"text","text":"a1"}]},
                   {"type":"user_input","content":[{"type":"text","text":"q2"}]}]}
"""

from __future__ import annotations

import json
import os
import time

import gemini_client as gc
import streaming
from gemini_client import is_mock, wire_counter, _text_tokens
from experiment import load_request
from payloads import (
    answer_text, extract_text, model_steps_from_response,
    single_step_input, user_step,
)

INTERACTION_TIMEOUT = float(os.environ.get("INTERACTION_TIMEOUT", "180"))

# What `store:true` adds to a mock turn, after the answer is already out. The real
# thing measured ~1.8 s (docs/interactions-api-fields.md); the mock keeps the shape
# so a mock run cannot pretend the stored arms are free.
MOCK_STORE_TAIL_MS = 1800


def interactions_url() -> str:
    return f"{gc.api_base()}/interactions"


def interaction_body(model: str, text: str, system_instruction: str,
                     prev_id: str | None, store: bool = True,
                     history: list | None = None) -> dict:
    """One interaction request body.

    Two ways to give the server a conversation. Chained (the default): `input` is
    the new question alone and `previous_interaction_id` points at the interaction
    that holds everything before it. Client-side (`history`): `input` is the whole
    conversation as a Step[] -- prior questions, prior model answers, new question
    last -- and no previous_interaction_id is sent, because there is nothing stored
    to point at.

    system_instruction rides every turn either way: the server does not keep it.
    """
    body: dict = {
        "model": model,
        # Every arm streams. TTFT cannot be measured any other way, and without it
        # this arm is charged for the ~1.8 s write that lands *after* its last token
        # -- a wait no streaming client ever does (see streaming.py).
        "stream": True,
        "store": store,
        "input": list(history) if history is not None else single_step_input(text),
    }
    if system_instruction:
        body["system_instruction"] = system_instruction
    if prev_id and history is None:
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


def _record(turn, question, text, iid, usage, wire_sent, wire_recv, elapsed_ms,
            req_raw, resp_raw, error, steps=None, ttft_ms=None, ttlt_ms=None,
            turn_end_ms=None, req_sent_ms=None, ttfb_ms=None) -> dict:
    # An arm with no timings of its own (an error, a dead connection) still ended;
    # zeros there would read as "instant". Fall back to the one number that exists.
    end = elapsed_ms if turn_end_ms is None else turn_end_ms
    return {
        "turn": turn, "question": question,
        "response_text": text, "interaction_id": iid,
        # The model's turn as the server sent it -- thought step, signature and all.
        # A client-side history echoes this; rebuilding the turn from response_text
        # would drop the thought step, which no real client does.
        "response_steps": list(steps or []),
        "wire_sent": wire_sent, "wire_recv": wire_recv, "elapsed_ms": elapsed_ms,
        # Five marks on one turn: the upload finishes, the server starts answering,
        # the answer starts, the answer ends, the server lets go. An arm that resends
        # a long history pays in req_sent; a stored interaction pays after ttlt.
        "req_sent_ms": end if req_sent_ms is None else req_sent_ms,
        "ttfb_ms": end if ttfb_ms is None else ttfb_ms,
        "ttft_ms": end if ttft_ms is None else ttft_ms,
        "ttlt_ms": end if ttlt_ms is None else ttlt_ms,
        "turn_end_ms": end,
        **usage,
        "request_raw": req_raw, "response_raw": resp_raw,
        "error": error,
    }


# A real thought signature is an opaque base64 blob of roughly this length. The mock
# carries one of its own so that a client-side history echoing the response is
# exercised offline, byte cost included -- the signature is most of what the echo
# adds to the upload.
def _mock_signature(turn: int) -> str:
    return f"MOCKSIG{turn:03d}" + ("A" * 60)


def _mock_interaction(turn: int, text: str, system: str, prev_id: str | None,
                      store: bool = True, history: list | None = None) -> dict:
    iid = f"mock_interaction_{turn:03d}"
    ans = f"(mock interaction answer, turn {turn}, prev={prev_id or 'none'}) " + ("lorem ipsum " * 12)
    body = interaction_body("mock-model", text, system, prev_id, store=store,
                            history=history)
    req = json.dumps(body)
    if history is not None:
        # client_history arm: the payload is the system prompt (every turn,
        # never stored) plus every text leaf actually carried in the Step[].
        # This must grow with the conversation -- that growth is the whole
        # point of the arm -- unlike the chained-arm estimate below, which is
        # untouched.
        history_text = "".join(extract_text(step) for step in history)
        inp = _text_tokens([{"parts": [{"text": system + history_text}]}])
    else:
        inp = _text_tokens([{"parts": [{"text": (system if turn == 1 else '') + text}]}])
    usage = {"input_tokens": inp, "cached_tokens": 0, "output_tokens": 40,
             "thought_tokens": 0, "total_tokens": inp + 40}
    steps = [{"signature": _mock_signature(turn), "type": "thought"},
             {"content": [{"text": ans, "type": "text"}], "type": "model_output"}]
    resp = json.dumps({"id": iid, "status": "completed", "steps": steps})
    # Mock timings shaped like the real ones: the answer lands at ttlt, and a stored
    # interaction then holds the stream open for the write. store:false does not.
    end = gc.MOCK_TTLT_MS + (MOCK_STORE_TAIL_MS if store else 0)
    return _record(turn, text, ans, iid, usage,
                   len(req) + 200, len(resp) + 200, end, req, resp, "", steps=steps,
                   req_sent_ms=gc._mock_req_sent_ms(len(req)), ttfb_ms=gc.MOCK_TTFB_MS,
                   ttft_ms=gc.MOCK_TTFT_MS, ttlt_ms=gc.MOCK_TTLT_MS, turn_end_ms=end)


def _call_interaction(model: str, text: str, system_instruction: str,
                      prev_id: str | None, turn: int = 1,
                      store: bool = True, history: list | None = None) -> dict:
    """One streamed interaction request. Never raises; errors land in the record's
    error. Returns the common per-turn record, timings included."""
    body = interaction_body(model, text, system_instruction, prev_id, store=store,
                            history=history)
    req_raw = json.dumps(body)
    headers = {"Content-Type": "application/json", **gc.auth_headers()}
    t0 = time.monotonic()
    try:
        with wire_counter() as w:
            with gc._session().post(interactions_url(), data=req_raw,
                                    headers=headers, timeout=INTERACTION_TIMEOUT,
                                    stream=True) as resp:
                if resp.status_code not in (200, 201):
                    err_body, stream = resp.text, None
                else:
                    stream = streaming.read_stream(
                        resp, streaming.interaction_text, t0)
    except Exception as exc:
        return _record(turn, text, "", None, _usage_common({}), 0, 0,
                       int((time.monotonic() - t0) * 1000), req_raw, "",
                       f"request_failed: {exc}")

    elapsed = int((time.monotonic() - t0) * 1000)
    if stream is None:
        return _record(turn, text, "", None, _usage_common({}),
                       w.sent, w.recv, elapsed, req_raw, err_body,
                       f"http_{resp.status_code}: {err_body[:200]}")

    # The events, reassembled into the body a stream:false call would have returned.
    # The completed event carries the usage but *not* the steps (measured), so this
    # is where the model's turn -- thought step, signature and all -- comes back.
    data = streaming.interaction_response(stream.events)
    steps = model_steps_from_response(data)
    # The answer is the model_output steps, never the thought step: with
    # thinking_summaries on, a thought step carries text of its own.
    text_out = answer_text(steps) or stream.text
    usage = _usage_common(data.get("usage") or {})
    return _record(turn, text, text_out, data.get("id"), usage,
                   w.sent, w.recv, elapsed, req_raw, json.dumps(data), "",
                   steps=steps,
                   req_sent_ms=streaming.since(t0, w.last_send_at),
                   ttfb_ms=streaming.since(t0, w.first_recv_at, stream.ttft_ms),
                   ttft_ms=stream.ttft_ms, ttlt_ms=stream.ttlt_ms,
                   turn_end_ms=stream.turn_end_ms)


def run_interaction(model: str, request_name: str = "perf",
                    turns: int | None = None, on_progress=None,
                    inline_system: bool = False, stage: str = "interaction",
                    client_history: bool = False) -> dict:
    """Replay the scenario over the Interactions API.

    Default: turn 1 opens the interaction (system prompt + first question), and each
    later turn sends the system prompt *again* plus the new question with
    previous_interaction_id. The server keeps the conversation, but not the
    system_instruction -- that field is interaction-scoped -- so a 20 KB system
    prompt is re-uploaded on every single turn and the arm saves almost nothing on
    the wire.

    inline_system=True: no system_instruction at all. The system prompt rides in the
    first user message, which makes it part of the server-side history, so every turn
    after the first sends only its question. Same content reaches the model either
    way; what changes is who stores it, and whether the model still treats it with
    the weight a system instruction carries.

    client_history=True: the server stores nothing (store:false) and there is no
    previous_interaction_id. The client keeps the conversation and resends it whole
    every turn as a Step[] in `input` -- its own prior questions and the model's own
    prior answers. This is the stateless arm's bargain, made on the interactions
    endpoint: what separates it from the chained arm is exactly what
    previous_interaction_id buys.

    Returns {"params": {...}, "interaction_records": [...]}.
    """
    system, steps, source = load_request(request_name)
    if turns:
        steps = steps[:max(1, min(turns, len(steps)))]
    n = len(steps)

    records = []
    prev_id: str | None = None
    history: list = []              # only used when client_history
    for k, q in enumerate(steps, start=1):
        if on_progress:
            on_progress({"stage": stage, "turn": k, "turns": n})
        if inline_system:
            # The prompt goes in the user turn, once, and the server holds it after.
            text = f"{system}\n\n{q}" if (k == 1 and system) else q
            sys_instruction = ""
        else:
            text, sys_instruction = q, system

        if client_history:
            history.append(user_step(text))
            sent_history, store = list(history), False
        else:
            sent_history, store = None, True

        if is_mock():
            r = _mock_interaction(k, text, sys_instruction, prev_id,
                                  store=store, history=sent_history)
        else:
            r = _call_interaction(model, text, sys_instruction, prev_id, turn=k,
                                  store=store, history=sent_history)
        r["turn"] = k
        r["question"] = q          # the step, never the prompt bolted onto it
        if client_history:
            # The arm's own answer, as the server sent it -- thought step included.
            # Not another arm's transcript (a history of a conversation that never
            # happened measures nothing), and not a step rebuilt from the text (that
            # drops the signature and under-reports what a real client uploads).
            history.extend(model_steps_from_response(
                {"steps": r.get("response_steps")},
                fallback_text=r.get("response_text") or ""))
        elif r.get("interaction_id"):
            prev_id = r["interaction_id"]     # chain the next turn onto this one
        records.append(r)

    return {
        "params": {"mode": stage, "turns": n, "model": model,
                   "stream": False, "endpoint": interactions_url(),
                   "inline_system": inline_system,
                   "client_history": client_history, "store": not client_history,
                   "request_source": source},
        "interaction_records": records,
    }
