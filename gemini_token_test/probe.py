"""Capability probe for the Interactions API.

Answers, empirically, the questions the docs leave open before any performance
comparison is built on top of them:

  1. Does a plain Gemini *model* interaction work at all, and on which host?
     The Vertex/GEAP reference lists only `lyria-3-*` under `model` and only
     `deep-research-preview-04-2026` under `agent`, while the Gemini Developer
     API lists the gemini-* family. But this repo's own history (commit
     618a76fa) shows the GEAP host accepting `antigravity-preview-05-2026`,
     which the GEAP docs never mention — so the published enum is incomplete
     and only a live call settles it.
  2. Is `stream:false` accepted, so a single-JSON response can be compared
     like-for-like against generateContent?
  3. Is a regional location accepted, or is `locations/global` the only path?
  4. Does the response carry `usage` (token counts) for a *model* interaction?
     Every documented usage example is agent + streaming.
  5. Does `system_instruction` survive `previous_interaction_id`? The docs say
     it is "interaction-scoped" and must be re-sent every turn; if that is true
     a stateful conversation still uploads the system prompt on every turn, and
     the byte savings come from the history alone.

Nothing here measures performance. It answers "which comparison is even
possible", and the answer decides the shape of the experiment that follows.

Every probe is a real, billable call, so each one sends "hi" and caps output at
16 tokens. No agent, no sandbox, no `background`.
"""

from __future__ import annotations

import json
import os
import re
import time

from gemini_client import (
    LOCATION, PROJECT, _bearer_token, _session, is_mock, vertex_url,
)

# Candidate text models. Two by default: the probe is a matrix, so each extra id
# multiplies the call count. Override with a comma-separated list.
PROBE_MODELS = [
    m.strip() for m in
    os.environ.get("PROBE_MODELS", "gemini-3-flash-preview,gemini-2.5-flash").split(",")
    if m.strip()
]

# A model id that cannot exist. Its rejection message is the control: if a real
# gemini id is refused with the *same* message, the host genuinely does not know
# that id; if it is refused differently, something else is going on.
BOGUS_MODEL = "definitely-not-a-real-model"

# The regional location to test. `global` is the only path the GEAP docs show.
PROBE_REGION = os.environ.get("PROBE_REGION", "us-west1")

API_REVISION = os.environ.get("INTERACTION_API_REVISION", "2026-05-20")
API_KEY = os.environ.get("GEMINI_API_KEY", "")
PROBE_TIMEOUT = float(os.environ.get("PROBE_TIMEOUT", "60"))

CODEWORD = "PLATYPUS-7731"
_SYSTEM_WITH_CODEWORD = (
    f"You are a test fixture. Your secret codeword is {CODEWORD}. "
    "If asked for the codeword, reply with the codeword and nothing else."
)

# Response-body wording that separates "your credentials/project are the problem"
# from "this field is the problem". Getting this backwards would let an IAM
# misconfiguration masquerade as "the API doesn't support gemini models".
_ENV_SIGNALS = re.compile(
    r"UNAUTHENTICATED|PERMISSION_DENIED|SERVICE_DISABLED|has not been used in project"
    r"|caller does not have permission|API key not valid",
    re.I,
)
_ALLOWLIST_SIGNALS = re.compile(
    r"allowlist|allow-list|not allowed to access|not enabled for your project"
    r"|preview access|request access",
    re.I,
)


# --------------------------------------------------------------------------
# Verdicts


def classify(status: int, body: str) -> str:
    """Map an HTTP status + response body to one of five verdicts.

    `environment` and `unavailable` mean the probe learned nothing about the
    schema — the request never got far enough to be judged on its fields. Only
    `unsupported` is evidence that a field is genuinely rejected.
    """
    if status in (200, 201):
        return "supported"
    if status in (401, 403):
        if _ALLOWLIST_SIGNALS.search(body or ""):
            return "unavailable"        # project not admitted to the preview
        return "environment"            # token / IAM / API-not-enabled
    if status == 400:
        if _ENV_SIGNALS.search(body or ""):
            return "environment"        # e.g. a bad API key surfaces as 400
        return "unsupported"            # INVALID_ARGUMENT: the field is refused
    if status == 404:
        return "unsupported"            # no such resource path (e.g. a region)
    if status == 429:
        return "environment"            # quota, not schema
    return "error"


# --------------------------------------------------------------------------
# Targets


def _targets() -> list[dict]:
    """The (host, url, auth) combinations worth probing, in priority order."""
    out = [
        {
            "name": "vertex-global",
            "url": f"https://aiplatform.googleapis.com/v1beta1/projects/{PROJECT}"
                   f"/locations/global/interactions",
            "auth": "adc",
            "note": "GEAP host, locations/global — what the current code uses.",
        },
        {
            "name": f"vertex-{PROBE_REGION}",
            "url": f"https://aiplatform.googleapis.com/v1beta1/projects/{PROJECT}"
                   f"/locations/{PROBE_REGION}/interactions",
            "auth": "adc",
            "note": "Regional location. Only worth having if it answers 200.",
        },
    ]
    if API_KEY:
        out.append({
            "name": "devapi",
            "url": "https://generativelanguage.googleapis.com/v1beta/interactions",
            "auth": "apikey",
            "note": "Gemini Developer API. GA, and the only surface whose docs "
                    "list gemini-* under `model`.",
        })
    else:
        out.append({
            "name": "devapi",
            "url": "https://generativelanguage.googleapis.com/v1beta/interactions",
            "auth": "apikey",
            "skipped": "GEMINI_API_KEY not set",
            "note": "Set GEMINI_API_KEY to probe the Developer API.",
        })
    return out


def _headers(auth: str) -> dict:
    h = {"Content-Type": "application/json"}
    if auth == "adc":
        h["Authorization"] = f"Bearer {_bearer_token()}"
        h["Api-Revision"] = API_REVISION
    else:
        h["x-goog-api-key"] = API_KEY
    return h


def _input(text: str) -> list:
    return [{"type": "user_input", "content": [{"type": "text", "text": text}]}]


# --------------------------------------------------------------------------
# One call


def _usage_from_events(events: list) -> dict:
    """Pull `interaction.usage` out of whichever event carries it."""
    for ev in reversed(events):
        inter = ev.get("interaction") if isinstance(ev, dict) else None
        if isinstance(inter, dict) and isinstance(inter.get("usage"), dict):
            return inter["usage"]
    return {}


def _text_from(obj) -> str:
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


def _parse_sse(raw: str) -> list:
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


def _call(url: str, auth: str, body: dict) -> dict:
    """POST one interaction body. Never raises; the failure lands in the result.

    Streaming and non-streaming responses are normalized to the same shape, so
    the caller can compare them without caring which came back.
    """
    payload = json.dumps(body)
    t0 = time.monotonic()

    def ms() -> int:
        return int((time.monotonic() - t0) * 1000)

    try:
        headers = _headers(auth)
    except Exception as exc:
        return {"status": 0, "verdict": "environment", "elapsed_ms": ms(),
                "error": f"auth_failed: {exc}", "body": "", "usage": {}, "text": "",
                "request": body}

    try:
        resp = _session().post(url, data=payload, headers=headers,
                               timeout=PROBE_TIMEOUT, stream=bool(body.get("stream")))
    except Exception as exc:
        return {"status": 0, "verdict": "error", "elapsed_ms": ms(),
                "error": f"request_failed: {exc}", "body": "", "usage": {}, "text": "",
                "request": body}

    raw = resp.text                     # consumes the stream if there is one
    verdict = classify(resp.status_code, raw)

    usage, text, interaction_id = {}, "", None
    if verdict == "supported":
        if body.get("stream"):
            events = _parse_sse(raw)
            usage = _usage_from_events(events)
            text = _text_from(events)
            for ev in events:
                inter = ev.get("interaction") if isinstance(ev, dict) else None
                if isinstance(inter, dict) and inter.get("id"):
                    interaction_id = inter["id"]
        else:
            try:
                data = resp.json()
                usage = data.get("usage") or {}
                text = _text_from(data)
                interaction_id = data.get("id")
            except Exception as exc:
                verdict = "error"
                text = f"parse_failed: {exc}"

    return {
        "status": resp.status_code,
        "verdict": verdict,
        "elapsed_ms": ms(),
        "error": "" if verdict == "supported" else raw[:400],
        "body": raw[:400],
        "usage": usage,
        "text": text[:400],
        "interaction_id": interaction_id,
        "request": body,
    }


def _model_body(model: str, stream: bool, system: str = "",
                prev: str | None = None, text: str = "hi") -> dict:
    body = {
        "model": model,
        "stream": stream,
        "store": True,
        "input": _input(text),
        "generation_config": {"max_output_tokens": 16},
    }
    if system:
        body["system_instruction"] = system
    if prev:
        body["previous_interaction_id"] = prev
    return body


# --------------------------------------------------------------------------
# generateContent control


def _generate_content(target: str, model: str) -> dict:
    """Does this host serve this model over the *old* API? Establishes that a
    model id is valid for the host before its interactions rejection is read as
    'interactions does not support gemini'."""
    body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {"maxOutputTokens": 16}}
    if target == "devapi":
        if not API_KEY:
            return {"status": 0, "verdict": "skipped", "error": "GEMINI_API_KEY not set"}
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent")
        auth = "apikey"
    else:
        url = vertex_url(model)
        auth = "adc"

    t0 = time.monotonic()
    try:
        resp = _session().post(url, data=json.dumps(body), headers=_headers(auth),
                               timeout=PROBE_TIMEOUT)
    except Exception as exc:
        return {"status": 0, "verdict": "error", "error": f"request_failed: {exc}"}
    return {"status": resp.status_code,
            "verdict": classify(resp.status_code, resp.text),
            "elapsed_ms": int((time.monotonic() - t0) * 1000),
            "url": url,
            "error": "" if resp.status_code in (200, 201) else resp.text[:400]}


# --------------------------------------------------------------------------
# Probe sequence


def _probe_target(target: dict) -> dict:
    """Run the checks for one host, cheapest and most decisive first.

    Order matters: the bogus-model control runs before any real model, so an
    auth or allowlist failure is diagnosed once instead of being mistaken for
    N separate 'model unsupported' results.
    """
    name, url, auth = target["name"], target["url"], target["auth"]
    out: dict = {"target": name, "url": url, "note": target.get("note", ""),
                 "checks": {}, "models": {}}

    if target.get("skipped"):
        out["verdict"] = "skipped"
        out["reason"] = target["skipped"]
        return out

    if auth == "adc" and not PROJECT:
        out["verdict"] = "skipped"
        out["reason"] = "GOOGLE_CLOUD_PROJECT not set"
        return out

    control = _call(url, auth, _model_body(BOGUS_MODEL, stream=False))
    out["checks"]["control_bogus_model"] = control
    if control["verdict"] in ("environment", "unavailable"):
        # The host never judged the body. Stop: everything after this would be
        # the same failure wearing a different hat.
        out["verdict"] = control["verdict"]
        out["reason"] = control["body"]
        return out
    out["control_message"] = control["body"]

    supported_model = None
    for model in PROBE_MODELS:
        entry = {
            "interactions_stream": _call(url, auth, _model_body(model, stream=True)),
            "interactions_nonstream": _call(url, auth, _model_body(model, stream=False)),
            "generate_content": _generate_content(name, model),
        }
        s, n = entry["interactions_stream"], entry["interactions_nonstream"]
        entry["usage_reported"] = bool(s.get("usage")) or bool(n.get("usage"))
        # A rejection whose wording matches the bogus model's, once the ids are
        # blanked out, means "unknown id" — not "gemini models are refused on
        # principle". A *different* rejection is worth reading by hand.
        entry["same_message_as_bogus"] = (
            n["verdict"] == "unsupported"
            and _blank_id(n["body"], model) == _blank_id(control["body"], BOGUS_MODEL)
        )
        out["models"][model] = entry
        if supported_model is None and (s["verdict"] == "supported"
                                        or n["verdict"] == "supported"):
            supported_model = model

    out["checks"]["region"] = {"probed": name != "devapi", "verdict": _region_verdict(out)}

    if supported_model:
        out["checks"]["system_instruction"] = _probe_system_instruction(
            url, auth, supported_model)
    else:
        out["checks"]["system_instruction"] = {
            "verdict": "skipped", "reason": "no model interaction succeeded"}

    out["verdict"] = "supported" if supported_model else "unsupported"
    out["supported_model"] = supported_model
    return out


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _blank_id(body: str, model: str) -> str:
    """Normalized body with the model id replaced, so two rejection messages can
    be compared for wording rather than for which id they name."""
    return _normalize(body).replace(_normalize(model), "<model>")


def _region_verdict(out: dict) -> str:
    """A target that answered anything other than 404 for every model at least
    exists as a resource path."""
    verdicts = {e["interactions_nonstream"]["verdict"] for e in out["models"].values()}
    if not verdicts:
        return "unknown"
    statuses = {e["interactions_nonstream"]["status"] for e in out["models"].values()}
    return "unsupported" if statuses == {404} else "supported"


def _probe_system_instruction(url: str, auth: str, model: str) -> dict:
    """Two turns. Turn 1 carries a codeword in `system_instruction` and stores the
    interaction. Turn 2 chains via `previous_interaction_id` and sends *no*
    system_instruction, then asks for the codeword.

    If the answer contains the codeword, the server kept the system prompt and a
    stateful client need not re-upload it every turn. If not, the docs are right
    and the system prompt is per-turn traffic — which is most of the byte budget
    once the prompt is 12K characters.
    """
    turn1 = _call(url, auth, _model_body(model, stream=False,
                                         system=_SYSTEM_WITH_CODEWORD,
                                         text="Say ready."))
    if turn1["verdict"] != "supported":
        return {"verdict": "inconclusive", "reason": "turn 1 failed",
                "accepted": turn1["verdict"] != "unsupported", "turn1": turn1}
    if not turn1.get("interaction_id"):
        return {"verdict": "inconclusive", "reason": "no interaction id returned",
                "accepted": True, "turn1": turn1}

    turn2 = _call(url, auth, _model_body(
        model, stream=False, prev=turn1["interaction_id"],
        text="Reply with only the codeword from your system instruction."))

    persisted = CODEWORD.lower() in (turn2.get("text") or "").lower()
    return {
        "verdict": "persisted" if persisted else "per_turn",
        "accepted": True,               # the field itself was not rejected
        "codeword": CODEWORD,
        "turn1": turn1,
        "turn2": turn2,
        "meaning": ("system_instruction survives previous_interaction_id; it need "
                    "not be re-sent" if persisted else
                    "system_instruction must be re-sent every turn (matches the docs); "
                    "a stateful arm still uploads the whole system prompt per turn"),
    }


# --------------------------------------------------------------------------
# Mock


def _mock() -> dict:
    """Offline stand-in matching what the published docs predict, so the UI and
    the tests can run with no credentials. Deliberately shows Vertex refusing a
    gemini model and the Developer API accepting one."""
    def call(status, verdict, usage=None, text=""):
        return {"status": status, "verdict": verdict, "elapsed_ms": 0,
                "error": "" if verdict == "supported" else "(mock) rejected",
                "body": "(mock)", "usage": usage or {}, "text": text,
                "interaction_id": "mock_1" if verdict == "supported" else None,
                "request": {}}

    model = PROBE_MODELS[0]
    vertex = {
        "target": "vertex-global", "url": "(mock)", "note": "", "verdict": "unsupported",
        "supported_model": None, "control_message": "(mock) unknown model",
        "checks": {"control_bogus_model": call(400, "unsupported"),
                   "region": {"probed": True, "verdict": "supported"},
                   "system_instruction": {"verdict": "skipped",
                                          "reason": "no model interaction succeeded"}},
        "models": {model: {"interactions_stream": call(400, "unsupported"),
                           "interactions_nonstream": call(400, "unsupported"),
                           "generate_content": {"status": 200, "verdict": "supported"},
                           "usage_reported": False, "same_message_as_bogus": True}},
    }
    dev = {
        "target": "devapi", "url": "(mock)", "note": "", "verdict": "supported",
        "supported_model": model, "control_message": "(mock) unknown model",
        "checks": {"control_bogus_model": call(400, "unsupported"),
                   "region": {"probed": False, "verdict": "supported"},
                   "system_instruction": {"verdict": "per_turn", "accepted": True,
                                          "codeword": CODEWORD,
                                          "meaning": "(mock) re-send every turn"}},
        "models": {model: {
            "interactions_stream": call(200, "supported", {"total_tokens": 20}),
            "interactions_nonstream": call(200, "supported", {"total_tokens": 20}),
            "generate_content": {"status": 200, "verdict": "supported"},
            "usage_reported": True, "same_message_as_bogus": False}},
    }
    return {"mock": True, "targets": [vertex, dev],
            "env": {"project": PROJECT or "(unset)", "location": LOCATION,
                    "api_key": False, "models": PROBE_MODELS},
            "conclusion": _conclude([vertex, dev])}


# --------------------------------------------------------------------------
# Conclusion


def _conclude(targets: list[dict]) -> dict:
    """Turn the matrix into the one sentence that decides the next step."""
    by = {t["target"]: t for t in targets}
    vertex = [t for n, t in by.items() if n.startswith("vertex")]
    vertex_ok = [t for t in vertex if t.get("supported_model")]
    dev = by.get("devapi", {})
    dev_ok = bool(dev.get("supported_model"))

    # A target that was skipped or that failed on auth never judged the request
    # body, so it says nothing about which fields the API supports. If no target
    # got past that point, the only honest answer is "nothing was measured" —
    # reporting "unsupported" here would turn a missing credential into a claim
    # about the API.
    blocked = [t["target"] for t in targets
               if t.get("verdict") in ("environment", "skipped")]
    if not vertex_ok and not dev_ok and len(blocked) == len(targets):
        reasons = "; ".join(f"{t['target']}: {t.get('reason') or t.get('verdict')}"
                            for t in targets)
        return {"next_step": "fix_environment",
                "summary": "No target was actually probed — every one was skipped or "
                           f"blocked before the request body was judged. {reasons}",
                "blocked": blocked}

    env_blocked = [t["target"] for t in targets if t.get("verdict") == "environment"]
    if env_blocked and not vertex_ok and not dev_ok:
        return {"next_step": "fix_environment",
                "summary": f"Blocked before any schema question: {', '.join(env_blocked)}. "
                           "Fix credentials / IAM / API enablement, then re-run.",
                "blocked": env_blocked}

    if vertex_ok:
        t = vertex_ok[0]
        sysc = t["checks"].get("system_instruction", {})
        return {"next_step": "compare_on_vertex",
                "summary": f"{t['target']} runs model interactions "
                           f"({t['supported_model']}). All arms stay on Vertex/ADC.",
                "host": t["target"], "model": t["supported_model"],
                "system_instruction": sysc.get("verdict")}

    if dev_ok:
        sysc = dev["checks"].get("system_instruction", {})
        return {"next_step": "compare_on_devapi",
                "summary": "No model interaction on Vertex. The Developer API runs "
                           f"{dev['supported_model']}. Move every arm there so host, "
                           "auth, and network path stay identical across arms — or "
                           "keep Vertex and drop the interaction arm.",
                "host": "devapi", "model": dev["supported_model"],
                "system_instruction": sysc.get("verdict")}

    unprobed = [t["target"] for t in targets if t.get("verdict") == "skipped"]
    tail = (f" Not probed: {', '.join(unprobed)} — the finding covers only the hosts "
            "that answered." if unprobed else " Response bodies are attached.")
    return {"next_step": "no_comparison_possible",
            "summary": "No host accepted a plain model interaction. A stateless-vs-"
                       "stateful text comparison cannot be built as specified; that "
                       "is the finding." + tail,
            "unprobed": unprobed}


# --------------------------------------------------------------------------
# Entry point


def probe_interactions() -> dict:
    """Run the whole matrix. Safe to call with no credentials: every target
    reports why it was skipped rather than raising."""
    if is_mock():
        return _mock()

    targets = [_probe_target(t) for t in _targets()]
    return {
        "mock": False,
        "env": {"project": PROJECT or "(unset)", "location": LOCATION,
                "api_key": bool(API_KEY), "models": PROBE_MODELS,
                "api_revision": API_REVISION, "region_probed": PROBE_REGION},
        "targets": targets,
        "conclusion": _conclude(targets),
    }
