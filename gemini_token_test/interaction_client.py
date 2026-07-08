"""Experimental: run the scenario through the Vertex/Gemini Enterprise Agent
Platform *Interactions API* (server-side stateful conversation).

Unlike generateContent (stateless — the client resends history every turn), the
Interactions API keeps conversation state on the server: the first turn creates
an interaction and returns an id; later turns pass `previous_interaction_id` (plus
the returned `environment`) and send ONLY the new question — the server remembers
the rest.

Auth is the same ADC / service-account bearer token as the rest of the app (no
Gemini API key). This is additive and separate from the generateContent flow.

Schema per the docs (subject to change — the API is new):
  POST https://<host>/v1beta1/projects/<project>/locations/<location>/interactions
  body: {stream,background,store, agent:<model>, previous_interaction_id?, environment?,
         input:[{type:"user_input", content:[{type:"text", text:"..."}]}]}
  resp: {"interaction": {"id","environment_id",...}, ...} with the model text in
        the streamed/returned content. The exact output path isn't fully documented,
        so we extract text best-effort AND keep the raw response so it's inspectable.
"""

from __future__ import annotations

import json
import os

from gemini_client import (
    PROJECT, LOCATION, _bearer_token, _session, is_mock, _text_tokens,
)
from experiment import load_request, _user  # scenario steps + message shape

# Interactions live on the global aiplatform host per the docs; override if needed.
INTERACTION_HOST = os.environ.get("INTERACTION_HOST", "aiplatform.googleapis.com")


def interactions_url() -> str:
    return (f"https://{INTERACTION_HOST}/v1beta1/projects/{PROJECT}"
            f"/locations/{LOCATION}/interactions")


def _extract_text(obj) -> str:
    """Best-effort: collect every {"type":"text","text":...} leaf in the response."""
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


def _mock_interaction(turn: int, text: str, prev_id: str | None) -> dict:
    """Offline stand-in so the button works under GEMINI_MOCK with no network."""
    iid = f"mock_interaction_{turn:03d}"
    ans = f"(mock interaction answer, turn {turn}, prev={prev_id or 'none'}) " + ("lorem ipsum " * 12)
    resp = {"interaction": {"id": iid, "environment_id": "mock_env",
                            "status": "completed",
                            "usage": {"total_tokens": _text_tokens([_user(text)]) + 40}},
            "output": _input(ans), "event_type": "interaction.complete"}
    return {"id": iid, "environment_id": "mock_env", "text": ans, "raw": resp}


def _call_interaction(model: str, text: str, prev_id: str | None,
                      env_id: str | None) -> dict:
    """One interaction request. Returns {id, environment_id, text, raw, error}."""
    body = {
        "stream": False,
        "background": False,
        "store": True,          # persist so previous_interaction_id works next turn
        "agent": model,
        "input": _input(text),
    }
    if prev_id:
        body["previous_interaction_id"] = prev_id
    if env_id:
        body["environment"] = env_id
    payload = json.dumps(body)

    try:
        token = _bearer_token()
    except Exception as exc:
        return {"error": f"auth_failed: {exc}", "request_json": payload}

    try:
        resp = _session().post(
            interactions_url(), data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {token}"},
            timeout=120,
        )
    except Exception as exc:
        return {"error": f"request_failed: {exc}", "request_json": payload}

    out = {"request_json": payload, "response_json": resp.text}
    if resp.status_code not in (200, 201):
        out["error"] = f"http_{resp.status_code}: {resp.text[:200]}"
        return out
    try:
        data = resp.json()
        inter = data.get("interaction", {}) if isinstance(data, dict) else {}
        out["id"] = inter.get("id")
        out["environment_id"] = inter.get("environment_id")
        out["text"] = _extract_text(data)
        out["error"] = ""
    except Exception as exc:
        out["error"] = f"parse_failed: {exc}"
    return out


def run_interaction(model: str, request_name: str = "default",
                    turns: int | None = None, on_progress=None) -> dict:
    """Replay the scenario over the Interactions API. Turn 1 sends system + first
    question and opens the interaction; each later turn sends only the new question
    with previous_interaction_id, so the server keeps the history.
    on_progress(event) fires per turn with {stage:'interaction', turn, turns}."""
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

        if is_mock():
            r = _mock_interaction(k, text, prev_id)
            r["request_json"] = json.dumps({"input": _input(text),
                                            "previous_interaction_id": prev_id})
            r["response_json"] = json.dumps(r.pop("raw"))
            r["error"] = ""
        else:
            r = _call_interaction(model, text, prev_id, env_id)

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
                   "endpoint": interactions_url(), "request_source": source},
        "interaction_records": records,
    }
