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
import re
import time

from gemini_client import (
    PROJECT, _bearer_token, _session, is_mock, _text_tokens,
)
from experiment import load_request, _user  # scenario steps + message shape

INTERACTION_HOST = os.environ.get("INTERACTION_HOST", "aiplatform.googleapis.com")
# Interactions live under locations/global, not the regional Vertex location.
INTERACTION_LOCATION = os.environ.get("INTERACTION_LOCATION", "global")
# Two targets are possible. Leave INTERACTION_AGENT empty (default) to run a plain
# Gemini model via the `model` field — no agent, no sandbox, no provisioning, so it
# is far faster. Set INTERACTION_AGENT (e.g. "antigravity-preview-05-2026") to run
# the coding agent instead, which provisions a remote sandbox per environment.
INTERACTION_AGENT = os.environ.get("INTERACTION_AGENT", "")
INTERACTION_API_REVISION = os.environ.get("INTERACTION_API_REVISION", "2026-05-20")


# The scenario's system prompt describes tools and tells the model to call them.
# generateContent declares no tools, so the model only *describes* what it would do.
# The Interactions API can actually invoke built-in tools (google_search,
# code_execution, computer_use, …), so a plain Q&A comparison must switch them off:
# an empty `tools` list plus tool_choice="none" (valid: auto|any|none|validated).
# Both are interaction-scoped and must be re-sent on every request.
# Set INTERACTION_TOOL_CHOICE="" to omit the field, or INTERACTION_TOOLS to a JSON
# list like [{"type":"google_search"}] to allow specific tools.
INTERACTION_TOOL_CHOICE = os.environ.get("INTERACTION_TOOL_CHOICE", "none")
_TOOLS_ENV_SET = "INTERACTION_TOOLS" in os.environ or "INTERACTION_TOOL_CHOICE" in os.environ


def _tools() -> list:
    try:
        v = json.loads(os.environ.get("INTERACTION_TOOLS", "[]"))
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _agent_mode() -> bool:
    return bool(INTERACTION_AGENT)


def _background_default() -> bool:
    """Agent interactions REQUIRE background=true (the service rejects false with
    "agent interactions must set background to true"). Model interactions don't, so
    they stream in the foreground. INTERACTION_BACKGROUND overrides either way."""
    env = os.environ.get("INTERACTION_BACKGROUND")
    if env is not None:
        return env == "1"
    return _agent_mode()
INTERACTION_TIMEOUT = float(os.environ.get("INTERACTION_TIMEOUT", "180"))
# The first interaction provisions a sandbox container on demand and can come back
# with a "resource setup has just started" style error instead of an answer. So we
# warm the environment up first with a throwaway prompt, retry until it yields an
# environment id, then run the real turns against that id (reuse = no provisioning
# latency). Set INTERACTION_ENV_ID to skip warmup entirely (sandbox TTL is 7 days).
INTERACTION_ENV_ID = os.environ.get("INTERACTION_ENV_ID", "")
INTERACTION_WARMUP_TIMEOUT = float(os.environ.get("INTERACTION_WARMUP_TIMEOUT", "300"))
INTERACTION_WARMUP_INTERVAL = float(os.environ.get("INTERACTION_WARMUP_INTERVAL", "5"))
INTERACTION_WARMUP_TEXT = os.environ.get("INTERACTION_WARMUP_TEXT", "ready")

# A setup-in-progress response is retryable; a real bad request is not.
_SETUP_PENDING = re.compile(r"setup|provision|not ready|unavailable", re.I)


def _setup_pending(r: dict) -> bool:
    status = r.get("status")
    if status in (200, 201) or status is None:
        return False
    return bool(_SETUP_PENDING.search(r.get("response_json") or ""))


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
    req = {"input": _input(text), "previous_interaction_id": prev_id}
    return {"id": iid, "environment_id": "mock_env", "text": ans,
            "request": req, "request_json": json.dumps(req),
            "response_json": stream, "response_events": _parse_sse(stream),
            "error": "", "elapsed_ms": 0, "first_event_ms": 0,
            "event_types": [{"event_type": "interaction.output", "at_ms": 0},
                            {"event_type": "interaction.complete", "at_ms": 0}]}


def _call_interaction(text: str, prev_id: str | None, environment,
                      model: str = "", on_event=None) -> dict:
    """One interaction request. `environment` is {"type":"remote"} on the first
    turn or the returned env id (str) on continuation.

    Events are parsed as they arrive (not after the fact) and reported through
    on_event(event_type, elapsed_ms) so a long stall is attributable to the stage
    it happens in — e.g. environment provisioning vs the agent thinking. Returns
    {id, environment_id, text, request_json, response_json, error, elapsed_ms,
     event_types, first_event_ms}."""
    body = {
        "stream": True,
        "background": _background_default(),
        "store": True,               # persist so previous_interaction_id works next turn
        "input": _input(text),
    }
    if _agent_mode():
        # Agent target: needs a sandbox environment provisioned for it. Tooling is
        # the agent's whole purpose, so only constrain it when asked explicitly.
        body["agent"] = INTERACTION_AGENT
        body["environment"] = environment
        if _TOOLS_ENV_SET:
            body["tools"] = _tools()
            if INTERACTION_TOOL_CHOICE:
                body["tool_choice"] = INTERACTION_TOOL_CHOICE
    else:
        # Plain model target: `agent` is only required when `model` is absent.
        # Declare no tools and forbid tool calls so we get a pure text answer.
        body["model"] = model
        body["tools"] = _tools()
        if INTERACTION_TOOL_CHOICE:
            body["tool_choice"] = INTERACTION_TOOL_CHOICE
    if prev_id:
        body["previous_interaction_id"] = prev_id
    payload = json.dumps(body)

    t0 = time.monotonic()

    def _ms() -> int:
        return int((time.monotonic() - t0) * 1000)

    try:
        token = _bearer_token()
    except Exception as exc:
        return {"error": f"auth_failed: {exc}", "request_json": payload,
                "response_json": "", "elapsed_ms": _ms()}

    try:
        resp = _session().post(
            interactions_url(), data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {token}",
                     "Api-Revision": INTERACTION_API_REVISION},
            timeout=INTERACTION_TIMEOUT, stream=True,
        )
    except Exception as exc:
        return {"error": f"request_failed: {exc}", "request_json": payload,
                "response_json": "", "elapsed_ms": _ms()}

    out = {"request": body, "request_json": payload, "status": resp.status_code}
    if resp.status_code not in (200, 201):
        raw = resp.text
        out["response_json"] = raw
        out["error"] = f"http_{resp.status_code}: {raw[:300]}"
        out["elapsed_ms"] = _ms()
        return out

    # Read the SSE stream incrementally so each event is timestamped on arrival.
    lines: list[str] = []
    events: list = []
    event_types: list[dict] = []
    iid = env_id = None
    texts: list[str] = []
    first_event_ms = None
    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue
        lines.append(line)
        s = line.strip()
        if not s.startswith("data:"):
            continue
        chunk = s[5:].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            ev = json.loads(chunk)
        except Exception:
            continue
        events.append(ev)
        at = _ms()
        if first_event_ms is None:
            first_event_ms = at
        et = ev.get("event_type") if isinstance(ev, dict) else None
        event_types.append({"event_type": et, "at_ms": at})
        if on_event:
            on_event(et, at)
        inter = ev.get("interaction") if isinstance(ev, dict) else None
        if isinstance(inter, dict):
            iid = inter.get("id") or iid
            env_id = inter.get("environment_id") or env_id
        texts.append(_extract_text(ev))

    out["response_json"] = "\n".join(lines)   # raw SSE text, exactly as received
    out["response_events"] = events           # same stream, parsed into objects
    out["id"] = iid
    out["environment_id"] = env_id
    out["text"] = "".join(texts)
    out["event_types"] = event_types
    out["first_event_ms"] = first_event_ms
    out["elapsed_ms"] = _ms()
    out["error"] = "" if event_types else "no_sse_events (see response_raw)"
    return out


def warmup_environment(on_progress=None) -> dict:
    """Provision the agent sandbox before the scenario runs.

    Sends a throwaway prompt with environment={"type":"remote"} and retries while
    the service reports setup-in-progress, until it returns an environment id or
    INTERACTION_WARMUP_TIMEOUT elapses. The warmup interaction is deliberately NOT
    chained into the scenario (no previous_interaction_id), so turn 1 still starts
    a clean conversation — only the environment is reused.
    Returns {env_id, attempts, elapsed_ms, error}."""
    if not _agent_mode():
        # Model interactions have no sandbox to provision.
        return {"env_id": "", "attempts": 0, "elapsed_ms": 0, "error": "",
                "skipped": True, "reason": "model mode (no agent sandbox)"}
    if is_mock():
        return {"env_id": "mock_env", "attempts": 1, "elapsed_ms": 0, "error": ""}

    t0 = time.monotonic()
    attempts = 0
    last_err = ""
    while (time.monotonic() - t0) < INTERACTION_WARMUP_TIMEOUT:
        attempts += 1
        if on_progress:
            on_progress({"stage": "provisioning", "attempt": attempts,
                         "at_ms": int((time.monotonic() - t0) * 1000)})
        r = _call_interaction(INTERACTION_WARMUP_TEXT, None, {"type": "remote"})
        if r.get("environment_id"):
            return {"env_id": r["environment_id"], "attempts": attempts,
                    "elapsed_ms": int((time.monotonic() - t0) * 1000), "error": ""}
        last_err = r.get("error", "")
        if not _setup_pending(r):
            break                      # a real error — don't spin on it
        time.sleep(INTERACTION_WARMUP_INTERVAL)

    return {"env_id": "", "attempts": attempts,
            "elapsed_ms": int((time.monotonic() - t0) * 1000),
            "error": last_err or "warmup_timeout: sandbox never became ready"}


def run_interaction(model: str, request_name: str = "default",
                    turns: int | None = None, on_progress=None) -> dict:
    """Replay the scenario over the Interactions API. Turn 1 sends system + first
    question and opens the interaction; each later turn sends only the new question
    with previous_interaction_id, so the server keeps the history.

    Target is `model` (plain Gemini model, no sandbox — fast) unless
    INTERACTION_AGENT is set, in which case the agent runs in a remote sandbox that
    is warmed up first. on_progress fires per turn with {stage:'interaction', turn,
    turns} (and {stage:'provisioning'} during agent warmup)."""
    system, steps, source = load_request(request_name)
    if turns:
        steps = steps[:max(1, min(turns, len(steps)))]
    n = len(steps)

    # Initialization stage (agent mode only): get a ready sandbox before any
    # question is asked, so provisioning latency isn't charged to turn 1.
    if _agent_mode() and INTERACTION_ENV_ID:
        warmup = {"env_id": INTERACTION_ENV_ID, "attempts": 0, "elapsed_ms": 0,
                  "error": "", "skipped": True, "reason": "INTERACTION_ENV_ID set"}
    else:
        warmup = warmup_environment(on_progress)
        warmup.setdefault("skipped", False)

    records = []
    prev_id: str | None = None
    env_id: str | None = warmup.get("env_id") or None
    for k, q in enumerate(steps, start=1):
        if on_progress:
            on_progress({"stage": "interaction", "turn": k, "turns": n})
        # System instruction placement isn't documented; prepend it to the first
        # turn's text so the model is primed once, then rely on server-side state.
        text = f"{system}\n\n{q}" if (k == 1 and system) else q
        # Agent mode reuses the warmed sandbox by id; model mode has no environment.
        environment = env_id if env_id else {"type": "remote"}

        if is_mock():
            r = _mock_interaction(k, text, prev_id)
        else:
            # Forward each SSE event to the UI so a stall is attributable to the
            # stage it happens in (provisioning vs the model/agent working).
            def _ev(et, at_ms, _k=k):
                if on_progress:
                    on_progress({"stage": "interaction", "turn": _k, "turns": n,
                                 "event": et or "", "at_ms": at_ms})
            r = _call_interaction(text, prev_id, environment, model=model, on_event=_ev)

        if r.get("id"):
            prev_id = r["id"]            # chain the next turn onto this interaction
        if r.get("environment_id"):
            env_id = r["environment_id"]

        records.append({
            "turn": k, "question": q,
            "response_text": r.get("text", ""),
            "interaction_id": r.get("id"),
            "environment_id": r.get("environment_id"),
            # `request` is the body we POSTed, as an object; `response_events` is the
            # SSE stream parsed into objects. The *_raw fields keep the exact bytes
            # (the response is an SSE stream, not a JSON document).
            "request": r.get("request", {}),
            "request_raw": r.get("request_json", ""),
            "response_events": r.get("response_events", []),
            "response_raw": r.get("response_json", ""),
            # Timing: elapsed_ms is the whole turn; first_event_ms is how long the
            # server took before its first SSE event (i.e. provisioning + queueing).
            "elapsed_ms": r.get("elapsed_ms"),
            "first_event_ms": r.get("first_event_ms"),
            "event_types": r.get("event_types", []),
            "reused_env": bool(env_id),   # true = sandbox came from warmup, no provisioning
            "error": r.get("error", ""),
        })

    return {
        "params": {"mode": "interaction", "turns": n,
                   "target": "agent" if _agent_mode() else "model",
                   "model": "" if _agent_mode() else model,
                   "agent": INTERACTION_AGENT,
                   "background": _background_default(),
                   "tools": _tools(), "tool_choice": INTERACTION_TOOL_CHOICE,
                   "endpoint": interactions_url(),
                   "request_source": source, "warmup": warmup},
        "interaction_records": records,
    }
