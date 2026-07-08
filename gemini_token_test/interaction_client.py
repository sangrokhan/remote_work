"""Experimental: run the scenario through the Gemini Enterprise Agent Platform
*Interactions API* (server-side stateful conversation).

Unlike generateContent (stateless — the client resends history every turn), the
Interactions API keeps conversation state on the server: the first turn opens an
interaction (and provisions a remote environment on demand) and returns an id +
environment_id; later turns pass `previous_interaction_id` and the returned
environment and send ONLY the new question — the server remembers the rest.

Auth is the same ADC / service-account bearer token as the rest of the app (no
Gemini API key). Additive and separate from the generateContent flow.

Schema per the docs (the API is new/experimental — subject to change):
  POST https://aiplatform.googleapis.com/v1beta1/projects/<project>/locations/global/interactions
  headers: Authorization: Bearer <adc>, Api-Revision: <rev>
  body (first turn):
    {"stream":true,"background":true,"store":true,
     "agent":"antigravity-preview-05-2026",
     "environment":{"type":"remote"},
     "input":[{"type":"user_input","content":[{"type":"text","text":"..."}]}]}
  body (later turns): same, but "environment":"<env_id>" and
     "previous_interaction_id":"<id>".
  response: an SSE stream of InteractionSSEEvent objects; the final
     `interaction.complete` event carries interaction.id + environment_id.
The exact text-delta event shape isn't fully documented, so we extract text
best-effort across all events AND keep the raw stream so it's inspectable.
"""

from __future__ import annotations

import json
import os

from gemini_client import (
    PROJECT, _bearer_token, _session, is_mock, _text_tokens,
)
from experiment import load_request, _user  # scenario steps + message shape

INTERACTION_HOST = os.environ.get("INTERACTION_HOST", "aiplatform.googleapis.com")
# Interactions live under locations/global, not the regional Vertex location.
INTERACTION_LOCATION = os.environ.get("INTERACTION_LOCATION", "global")
# The base first-party agent works on the fly; override with a custom agent id.
INTERACTION_AGENT = os.environ.get("INTERACTION_AGENT", "antigravity-preview-05-2026")
INTERACTION_API_REVISION = os.environ.get("INTERACTION_API_REVISION", "2026-05-20")


def interactions_url() -> str:
    return (f"https://{INTERACTION_HOST}/v1beta1/projects/{PROJECT}"
            f"/locations/{INTERACTION_LOCATION}/interactions")


def _extract_text(obj) -> str:
    """Best-effort: collect every {"type":"text","text":...} leaf in a payload."""
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


def _input(text: str) -> list:
    return [{"type": "user_input", "content": [{"type": "text", "text": text}]}]


def _parse_sse(raw: str) -> list:
    """Collect JSON objects from the `data:` lines of an SSE stream."""
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            chunk = line[5:].strip()
            if chunk and chunk != "[DONE]":
                try:
                    events.append(json.loads(chunk))
                except Exception:
                    pass
    return events


def _mock_interaction(turn: int, text: str, prev_id: str | None) -> dict:
    """Offline stand-in so the button works under GEMINI_MOCK with no network."""
    iid = f"mock_interaction_{turn:03d}"
    ans = f"(mock interaction answer, turn {turn}, prev={prev_id or 'none'}) " + ("lorem ipsum " * 12)
    stream = (
        f"data: {json.dumps({'event_type': 'interaction.output', 'output': _input(ans)})}\n\n"
        f"data: {json.dumps({'event_type': 'interaction.complete', 'interaction': {'id': iid, 'environment_id': 'mock_env', 'status': 'completed', 'usage': {'total_tokens': _text_tokens([_user(text)]) + 40}}})}\n\n"
    )
    return {"id": iid, "environment_id": "mock_env", "text": ans,
            "request_json": json.dumps({"input": _input(text),
                                        "previous_interaction_id": prev_id}),
            "response_json": stream, "error": ""}


def _call_interaction(text: str, prev_id: str | None, environment) -> dict:
    """One interaction request. `environment` is {"type":"remote"} on the first
    turn or the returned env id (str) on continuation. Returns
    {id, environment_id, text, request_json, response_json, error}."""
    body = {
        "stream": True,
        "background": True,
        "store": True,               # persist so previous_interaction_id works next turn
        "agent": INTERACTION_AGENT,
        "environment": environment,
        "input": _input(text),
    }
    if prev_id:
        body["previous_interaction_id"] = prev_id
    payload = json.dumps(body)

    try:
        token = _bearer_token()
    except Exception as exc:
        return {"error": f"auth_failed: {exc}", "request_json": payload,
                "response_json": ""}

    try:
        resp = _session().post(
            interactions_url(), data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {token}",
                     "Api-Revision": INTERACTION_API_REVISION},
            timeout=180, stream=True,
        )
    except Exception as exc:
        return {"error": f"request_failed: {exc}", "request_json": payload,
                "response_json": ""}

    raw = resp.text  # consume the whole stream (small for our turns)
    out = {"request_json": payload, "response_json": raw}
    if resp.status_code not in (200, 201):
        out["error"] = f"http_{resp.status_code}: {raw[:300]}"
        return out

    events = _parse_sse(raw)
    iid = env_id = None
    texts = []
    for ev in events:
        inter = ev.get("interaction") if isinstance(ev, dict) else None
        if isinstance(inter, dict):
            iid = inter.get("id") or iid
            env_id = inter.get("environment_id") or env_id
        texts.append(_extract_text(ev))
    out["id"] = iid
    out["environment_id"] = env_id
    out["text"] = "".join(texts)
    out["error"] = "" if events else "no_sse_events (see response_json)"
    return out


def run_interaction(model: str, request_name: str = "default",
                    turns: int | None = None, on_progress=None) -> dict:
    """Replay the scenario over the Interactions API. Turn 1 sends system + first
    question and opens the interaction (provisioning a remote environment); each
    later turn sends only the new question with previous_interaction_id + the
    returned environment, so the server keeps the history. `model` is ignored —
    the interaction runs on INTERACTION_AGENT. on_progress fires per turn with
    {stage:'interaction', turn, turns}."""
    system, steps, source = load_request(request_name)
    if turns:
        steps = steps[:max(1, min(turns, len(steps)))]
    n = len(steps)

    records = []
    prev_id: str | None = None
    env_id: str | None = None
    for k, q in enumerate(steps, start=1):
        if on_progress:
            on_progress({"stage": "interaction", "turn": k, "turns": n})
        # System instruction placement isn't documented; prepend it to the first
        # turn's text so the agent is primed once, then rely on server-side state.
        text = f"{system}\n\n{q}" if (k == 1 and system) else q
        # First turn provisions a remote environment; later turns reuse it by id.
        environment = env_id if env_id else {"type": "remote"}

        if is_mock():
            r = _mock_interaction(k, text, prev_id)
        else:
            r = _call_interaction(text, prev_id, environment)

        if r.get("id"):
            prev_id = r["id"]            # chain the next turn onto this interaction
        if r.get("environment_id"):
            env_id = r["environment_id"]

        records.append({
            "turn": k, "question": q,
            "response_text": r.get("text", ""),
            "interaction_id": r.get("id"),
            "environment_id": r.get("environment_id"),
            "request_json": r.get("request_json", ""),
            "response_json": r.get("response_json", ""),
            "error": r.get("error", ""),
        })

    return {
        "params": {"mode": "interaction", "turns": n, "model": model,
                   "agent": INTERACTION_AGENT, "endpoint": interactions_url(),
                   "request_source": source},
        "interaction_records": records,
    }
