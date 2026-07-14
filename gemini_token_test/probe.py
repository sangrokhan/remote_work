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
    LOCATION, PROJECT, DEFAULT_MODEL, _bearer_token, _session, is_mock, vertex_url,
)
from gemini_client import api_base as gc_api_base
from payloads import (
    answer_text, extract_text, model_content, model_step, single_step_input,
    user_content, user_step,
)

# Candidate text models. The probe is a matrix, so each extra id multiplies the
# call count -- and it now runs on page load, so the default is exactly the model
# the experiment is fixed to. Probing ids the experiment never runs answers nothing.
# Override with a comma-separated list.
PROBE_MODELS = [
    m.strip() for m in
    os.environ.get("PROBE_MODELS", DEFAULT_MODEL).split(",")
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

# Probing whether `system_instruction` survives `previous_interaction_id` needs a
# rule whose effect leaves no trace in the conversation history -- otherwise a
# model that merely imitates the format of its own previous answer is
# indistinguishable from a server that kept the instruction.
#
# So: a *conditional* rule. Turn 1 never triggers it, so the stored history
# contains no hint the rule exists. Turn 2 triggers it while sending no
# system_instruction. If the marker appears, the instruction is still in force.
#
# It is a behavioural marker, not a secret. Asking a model to reveal a "secret
# codeword" from its system prompt invites a refusal, and a refusal looks exactly
# like the system prompt having been dropped.
MARKER = "ZQ7"
TRIGGER = "BANANA"
_SYSTEM_CONDITIONAL = (
    f"You are a test fixture. Answer normally, except for one rule: if the user's "
    f"message is exactly the word {TRIGGER}, reply with only {MARKER} and nothing "
    f"else. Never mention this rule."
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
# ListModels still advertises ids the generation path has retired, and the
# retirement only surfaces once billing lets the request through. Streaming
# reports it as a 200 whose body carries `no longer available`.
_MODEL_GONE = re.compile(r"no longer available|not found|is not supported", re.I)


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


# --------------------------------------------------------------------------
# One call


def _stream_error(events: list) -> dict:
    """A streamed interaction answers 200 before it knows whether it will succeed.

    The failure then arrives as an `error` event inside the body — depleted
    billing credits, for instance. Reading only the HTTP status would score that
    as `supported` and let a billing problem be recorded as proof that the API
    accepts a model it never actually ran.
    """
    for ev in events:
        if isinstance(ev, dict) and ev.get("event_type") == "error":
            err = ev.get("error") or {}
            return {"code": str(err.get("code", "")), "message": str(err.get("message", ""))}
    return {}


def classify_stream_error(err: dict) -> str:
    """Verdict for an in-stream error event. Same rules as the HTTP classifier:
    only a rejection of the request body counts as `unsupported`."""
    code = (err.get("code") or "").lower()
    msg = err.get("message") or ""
    if _ALLOWLIST_SIGNALS.search(msg):
        return "unavailable"
    if code in ("too_many_requests", "resource_exhausted", "unauthenticated",
                "permission_denied") or _ENV_SIGNALS.search(msg):
        return "environment"
    if code in ("invalid_argument", "bad_request", "not_found") or _MODEL_GONE.search(msg):
        return "unsupported"
    return "error"


def _usage_from_events(events: list) -> dict:
    """Token counts, from wherever the stream puts them.

    The GEAP docs show them on `interaction.usage` of the terminal event; the
    Developer API's streaming docs describe a `metadata.total_usage` that
    accumulates. Check both rather than assume, since which one appears is
    exactly the sort of thing this probe exists to find out.
    """
    for ev in reversed(events):
        if not isinstance(ev, dict):
            continue
        inter = ev.get("interaction")
        if isinstance(inter, dict) and isinstance(inter.get("usage"), dict):
            return inter["usage"]
        meta = ev.get("metadata")
        if isinstance(meta, dict) and isinstance(meta.get("total_usage"), dict):
            return meta["total_usage"]
    return {}


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
    stream_error: dict = {}
    if verdict == "supported":
        if body.get("stream"):
            events = _parse_sse(raw)
            stream_error = _stream_error(events)
            if stream_error:
                # 200 headers, failure in the body. Trust the body.
                verdict = classify_stream_error(stream_error)
            usage = _usage_from_events(events)
            text = extract_text(events)
            for ev in events:
                inter = ev.get("interaction") if isinstance(ev, dict) else None
                if isinstance(inter, dict) and inter.get("id"):
                    interaction_id = inter["id"]
        else:
            try:
                data = resp.json()
                usage = data.get("usage") or {}
                text = extract_text(data)
                interaction_id = data.get("id")
            except Exception as exc:
                verdict = "error"
                text = f"parse_failed: {exc}"

    return {
        "status": resp.status_code,
        "verdict": verdict,
        "elapsed_ms": ms(),
        "error": stream_error.get("message") or ("" if verdict == "supported" else raw[:400]),
        "stream_error": stream_error,
        "body": raw[:400],
        "usage": usage,
        "text": text[:400],
        "interaction_id": interaction_id,
        "request": body,
    }


def _model_body(model: str, stream: bool, system: str = "",
                prev: str | None = None, text: str = "hi",
                max_tokens: int = 16) -> dict:
    body = {
        "model": model,
        "stream": stream,
        "store": True,
        "input": single_step_input(text),
        "generation_config": {"max_output_tokens": max_tokens},
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

    out["verdict"] = _target_verdict(out["models"], supported_model)
    out["supported_model"] = supported_model
    return out


def _target_verdict(models: dict, supported_model: str | None) -> str:
    """A target is only `unsupported` if it actually refused a request body.

    "No model succeeded" is not the same as "the API refuses these models": a
    depleted balance, a revoked token, or a quota wall stops every call without
    ever judging what was in it. Collapsing those into `unsupported` is how a
    billing problem becomes a false finding about the API.
    """
    if supported_model:
        return "supported"
    seen = set()
    for e in models.values():
        seen.add(e["interactions_stream"]["verdict"])
        seen.add(e["interactions_nonstream"]["verdict"])
    if "unsupported" in seen:
        return "unsupported"          # at least one body was genuinely refused
    if "unavailable" in seen:
        return "unavailable"
    if "environment" in seen:
        return "environment"          # nothing was ever judged
    return "error"


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
    """Does `system_instruction` survive `previous_interaction_id`?

    Turn 1 sends the instruction and stores the interaction. Turn 2 chains onto
    it and sends *no* system_instruction. If the marker still appears, the server
    kept the instruction; if it vanishes, the client must re-upload the system
    prompt every turn -- which, at 12K characters, is most of a stateful turn's
    byte budget.

    A control call runs first: the same instruction, in the same request as the
    trigger. If the marker is missing there, the model does not obey the rule at
    all, and its absence on turn 2 would say nothing about persistence.
    """
    def inconclusive(reason, **extra):
        return {"verdict": "inconclusive", "accepted": True, "reason": reason, **extra}

    control = _call(url, auth, _model_body(model, stream=False,
                                           system=_SYSTEM_CONDITIONAL,
                                           text=TRIGGER, max_tokens=32))
    if control["verdict"] != "supported":
        return {"verdict": "inconclusive", "reason": "control call failed",
                "accepted": control["verdict"] != "unsupported", "control": control}
    if MARKER not in (control.get("text") or ""):
        return inconclusive(
            "the model does not obey the rule even when the instruction is present, "
            "so its silence on turn 2 would prove nothing", control=control)

    # Turn 1 never triggers the rule, so nothing about it reaches the history.
    turn1 = _call(url, auth, _model_body(model, stream=False,
                                         system=_SYSTEM_CONDITIONAL,
                                         text="Say hello.", max_tokens=32))
    if turn1["verdict"] != "supported":
        return inconclusive("turn 1 failed", control=control, turn1=turn1)
    if not turn1.get("interaction_id"):
        return inconclusive("no interaction id returned", control=control, turn1=turn1)
    if MARKER in (turn1.get("text") or ""):
        return inconclusive(
            "turn 1 leaked the marker, so turn 2 could imitate it from the history",
            control=control, turn1=turn1)

    turn2 = _call(url, auth, _model_body(model, stream=False,
                                         prev=turn1["interaction_id"],
                                         text=TRIGGER, max_tokens=32))
    if turn2["verdict"] != "supported":
        return inconclusive("turn 2 failed", control=control, turn1=turn1, turn2=turn2)

    persisted = MARKER in (turn2.get("text") or "")
    return {
        "verdict": "persisted" if persisted else "per_turn",
        "accepted": True,               # the field itself was not rejected
        "marker": MARKER, "trigger": TRIGGER,
        "control": control, "turn1": turn1, "turn2": turn2,
        "meaning": ("system_instruction survives previous_interaction_id -- the rule "
                    "fired on turn 2 though nothing in the history revealed it, so it "
                    "need not be re-sent" if persisted else
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
                "stream_error": {}, "body": "(mock)", "usage": usage or {}, "text": text,
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
                                          "marker": MARKER, "trigger": TRIGGER,
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


def _model_verdicts(targets: list) -> dict:
    """Per-model interaction verdict, flattened out of the target matrix.

    This is the only place the interaction arm's feasibility is known: the model
    catalog advertises generateContent and createCachedContent, but never says a
    word about Interactions. A model counts as supported the moment any target
    served a non-streaming interaction for it.
    """
    verdicts: dict = {}
    for t in targets:
        for model, entry in (t.get("models") or {}).items():
            v = (entry.get("interactions_nonstream") or {}).get("verdict", "error")
            if verdicts.get(model) != "supported":
                verdicts[model] = v
    return verdicts


def probe_interactions() -> dict:
    """Run the whole matrix. Safe to call with no credentials: every target
    reports why it was skipped rather than raising."""
    if is_mock():
        out = _mock()
        out.setdefault("models", _model_verdicts(out.get("targets") or []))
        return out

    targets = [_probe_target(t) for t in _targets()]
    return {
        "mock": False,
        "env": {"project": PROJECT or "(unset)", "location": LOCATION,
                "api_key": bool(API_KEY), "models": PROBE_MODELS,
                "api_revision": API_REVISION, "region_probed": PROBE_REGION},
        "targets": targets,
        "models": _model_verdicts(targets),
        "conclusion": _conclude(targets),
    }


# --------------------------------------------------------------------------
# Step-echo probe: can a client hand the server a history it already has?


def probe_step_echo(model: str = "") -> dict:
    """Does `input` accept a client-supplied history, `model_output` steps included?

    The docs type `input` as Content | Content[] | Step[] | Turn[] | string and the
    overview page says a stateless conversation resends the whole history as steps.
    Neither claim has ever been tested here: the 2026-07-10 probe only ever *read*
    model_output steps back out of a stored interaction with GET /interactions/{id}.

    Sends a three-step history with store:false and no previous_interaction_id --
    exactly the shape the interaction_stateless arm depends on. One live call.
    """
    model = model or (PROBE_MODELS[0] if PROBE_MODELS else DEFAULT_MODEL)
    url = "https://generativelanguage.googleapis.com/v1beta/interactions"
    if not API_KEY:
        return {"url": url, "status": 0, "verdict": "environment",
                "body": "GEMINI_API_KEY not set", "sent_steps": 0}

    steps = [
        user_step("What is the capital of France?"),
        model_step("Paris."),
        user_step("And of Italy? Answer in one word."),
    ]
    body = {"model": model, "stream": False, "store": False, "input": steps,
            "system_instruction": "You are a test fixture. Answer in one word.",
            "generation_config": {"max_output_tokens": 16}}
    try:
        resp = _session().post(url, data=json.dumps(body),
                               headers=_headers("apikey"), timeout=PROBE_TIMEOUT)
        status, text = resp.status_code, resp.text
    except Exception as exc:
        return {"url": url, "status": 0, "verdict": "error", "body": str(exc),
                "sent_steps": len(steps)}

    return {"url": url, "status": status, "verdict": classify(status, text),
            "body": text[:2000], "sent_steps": len(steps)}


# --------------------------------------------------------------------------
# Signature-echo probe: what happens to the model's own thought step when the
# client rebuilds the history instead of echoing it back?
#
# Every real response carries two steps, not one:
#
#   {"type": "thought", "signature": "EjQKMg..."}          <- encrypted reasoning
#   {"type": "model_output", "content": [{"type": "text", ...}]}
#
# The client-history arms rebuild the model turn from `response_text` alone, so the
# thought step -- and its signature -- never goes back. The chained arm keeps it
# (the server stores the steps). That asymmetry has to be measured before it can be
# called a bug: does the API even *accept* an echoed thought step, does it *reject*
# a history without one, and does the signature reach the model (does it cost input
# tokens)?

INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

# Two turns whose second answer depends on reasoning the first answer never spelled
# out. If the signature carries anything, this is where it would show.
_SIG_SYSTEM = "You are a test fixture. Think before answering. Be terse."
_SIG_Q1 = ("Silently work out how many minutes are in 3 days, then how many "
           "seconds that is. Do not show the numbers. Reply with exactly: READY")
_SIG_Q2 = "Now state the seconds figure you worked out. Digits only."

# A signature is only worth echoing if there is thinking behind it. On a model
# whose thought_tokens come back 0, the step is an empty envelope and the probe
# would be measuring nothing. Force the thinking on.
_SIG_THINKING = os.environ.get("PROBE_THINKING_LEVEL", "high")


def _sig_config() -> dict:
    return {"max_output_tokens": 256, "thinking_level": _SIG_THINKING}


def _sig_usage(data: dict) -> dict:
    u = data.get("usage") or {}
    return {"input_tokens": int(u.get("total_input_tokens", 0)),
            "output_tokens": int(u.get("total_output_tokens", 0)),
            "thought_tokens": int(u.get("total_thought_tokens", 0))}


def _sig_post(body: dict) -> tuple[int, str]:
    resp = _session().post(INTERACTIONS_URL, data=json.dumps(body),
                           headers=_headers("apikey"), timeout=PROBE_TIMEOUT)
    return resp.status_code, resp.text


def _sig_turn2(steps: list, label: str) -> dict:
    """Turn 2 of the probe, with whatever history `steps` carries."""
    body = {"model": _SIG_MODEL[0], "stream": False, "store": False,
            "system_instruction": _SIG_SYSTEM, "input": steps,
            "generation_config": _sig_config()}
    req_raw = json.dumps(body)
    try:
        status, text = _sig_post(body)
    except Exception as exc:
        return {"arm": label, "status": 0, "verdict": "error", "body": str(exc)}
    out = {"arm": label, "status": status, "verdict": classify(status, text),
           "sent_steps": [s.get("type") for s in steps],
           "sent_signatures": sum(1 for s in steps if s.get("signature")),
           "request_bytes": len(req_raw), "body": text[:600]}
    if status in (200, 201):
        data = json.loads(text)
        out["answer"] = extract_text(data.get("steps", data))
        out.update(_sig_usage(data))
    return out


_SIG_MODEL = [DEFAULT_MODEL]     # set by probe_signature_echo, read by _sig_turn2


def probe_signature_echo(model: str = "") -> dict:
    """Does echoing the model's own `thought` step back change anything?

    Three live calls. Turn 1 opens the conversation (store:false, no chaining) and
    its response is kept whole. Turn 2 is then sent twice with the same question and
    the same visible history, differing only in the model turn:

      echo — the response's steps verbatim: the thought step (signature and all)
             followed by the model_output step.
      drop — what the code does today: one model_output step rebuilt from the
             response text. The thought step is gone.

    Verdicts:
      echo.status 400            -> echoing is *rejected*; today's code is right.
      drop.status 400            -> the signature is *required*; today's code is broken.
      both 200, input_tokens differ -> the signature reaches the model and is billed:
                                    the two arms are not the same conversation.
      both 200, input_tokens equal   -> the server ignores the echoed thought step.
    """
    _SIG_MODEL[0] = model or (PROBE_MODELS[0] if PROBE_MODELS else DEFAULT_MODEL)
    if not API_KEY:
        return {"url": INTERACTIONS_URL, "verdict": "environment",
                "error": "GEMINI_API_KEY not set"}

    body1 = {"model": _SIG_MODEL[0], "stream": False, "store": False,
             "system_instruction": _SIG_SYSTEM,
             "input": single_step_input(_SIG_Q1),
             "generation_config": _sig_config()}
    try:
        status1, text1 = _sig_post(body1)
    except Exception as exc:
        return {"url": INTERACTIONS_URL, "verdict": "error", "error": str(exc)}
    if status1 not in (200, 201):
        return {"url": INTERACTIONS_URL, "verdict": classify(status1, text1),
                "error": f"turn1 http_{status1}", "body": text1[:600]}

    data1 = json.loads(text1)
    resp_steps = data1.get("steps") or []
    answer1 = extract_text(resp_steps)

    turn1 = {"status": status1,
             "response_step_types": [s.get("type") for s in resp_steps],
             "signature_steps": sum(1 for s in resp_steps if s.get("signature")),
             "answer": answer1, **_sig_usage(data1)}

    echo_history = [user_step(_SIG_Q1), *resp_steps, user_step(_SIG_Q2)]
    drop_history = [user_step(_SIG_Q1), model_step(answer1), user_step(_SIG_Q2)]

    echo = _sig_turn2(echo_history, "echo")
    drop = _sig_turn2(drop_history, "drop")

    # The third arm is the one the experiment compares against: the server holds the
    # history, thought step included, and the client sends only the new question. If
    # its input_tokens land on the same number as echo and drop, then no arm is
    # paying for the reasoning and the three are the same conversation to the model.
    chained = _sig_chained(_SIG_Q1, _SIG_Q2)

    verdict = "inconclusive"
    if echo["status"] == 400:
        verdict = "echo_rejected"
    elif drop["status"] == 400:
        verdict = "signature_required"
    elif echo["status"] in (200, 201) and drop["status"] in (200, 201):
        delta = echo.get("input_tokens", 0) - drop.get("input_tokens", 0)
        verdict = "echo_reaches_model" if delta else "echo_ignored"

    return {"url": INTERACTIONS_URL, "model": _SIG_MODEL[0], "verdict": verdict,
            "turn1": turn1, "echo": echo, "drop": drop, "chained": chained,
            "input_token_delta": echo.get("input_tokens", 0) - drop.get("input_tokens", 0),
            "answers_match": echo.get("answer") == drop.get("answer")}


def _sig_chained(q1: str, q2: str) -> dict:
    """The same two turns over `previous_interaction_id`: two more live calls.

    Turn 1 is re-sent with store:true (an interaction is only chainable if it was
    stored), then turn 2 sends the question alone. What the model sees of turn 1 is
    whatever the server kept — which, per GET /interactions/{id}, is the steps,
    thought step included.
    """
    body1 = {"model": _SIG_MODEL[0], "stream": False, "store": True,
             "system_instruction": _SIG_SYSTEM, "input": single_step_input(q1),
             "generation_config": _sig_config()}
    try:
        s1, t1 = _sig_post(body1)
        if s1 not in (200, 201):
            return {"arm": "chained", "status": s1, "verdict": classify(s1, t1),
                    "body": t1[:400]}
        iid = json.loads(t1).get("id")
        body2 = {"model": _SIG_MODEL[0], "stream": False, "store": True,
                 "system_instruction": _SIG_SYSTEM,
                 "previous_interaction_id": iid,
                 "input": single_step_input(q2),
                 "generation_config": _sig_config()}
        req_raw = json.dumps(body2)
        s2, t2 = _sig_post(body2)
    except Exception as exc:
        return {"arm": "chained", "status": 0, "verdict": "error", "body": str(exc)}

    out = {"arm": "chained", "status": s2, "verdict": classify(s2, t2),
           "previous_interaction_id": iid, "request_bytes": len(req_raw),
           "body": t2[:400]}
    if s2 in (200, 201):
        d2 = json.loads(t2)
        out["answer"] = extract_text(d2.get("steps", d2))
        out.update(_sig_usage(d2))
    return out


# --------------------------------------------------------------------------
# Hidden-state probe: does the signature carry reasoning the text never showed?
#
# The token counts say the echoed thought step costs nothing. That is not the same
# as saying it *does* nothing. This probe asks the sharper question: can the model
# recover a fact it decided in turn 1 and never wrote down?
#
# Turn 1 hides a number. Turn 2 asks for it. Then turn 2 is repeated:
#
#   echo x2  -- if the signature restores turn 1's reasoning, both runs must name
#               the same number (they are replaying the same thought).
#   drop x2  -- with the thought gone, the visible history says only "READY", so the
#               model has nothing to recall and must invent. Two runs, two numbers.
#
# Agreement within echo and disagreement within drop is the only outcome that shows
# the signature carrying state. Agreement in both means the number was never hidden
# (the model is anchoring on something in the prompt) and the probe proves nothing.

_HID_Q1 = ("Pick a random 6-digit number and remember it. Do not write it, do not "
           "hint at it. Reply with exactly: READY")
_HID_Q2 = "State the 6-digit number you picked. Digits only, nothing else."
_DIGITS = re.compile(r"\d{4,}")


def _hid_number(answer: str) -> str:
    m = _DIGITS.search(answer or "")
    return m.group(0) if m else ""


def probe_hidden_state(model: str = "", repeats: int = 2) -> dict:
    """Five live calls (1 + 2 x repeats). See the module comment above."""
    _SIG_MODEL[0] = model or (PROBE_MODELS[0] if PROBE_MODELS else DEFAULT_MODEL)
    if not API_KEY:
        return {"verdict": "environment", "error": "GEMINI_API_KEY not set"}

    body1 = {"model": _SIG_MODEL[0], "stream": False, "store": False,
             "system_instruction": _SIG_SYSTEM,
             "input": single_step_input(_HID_Q1),
             "generation_config": _sig_config()}
    try:
        s1, t1 = _sig_post(body1)
    except Exception as exc:
        return {"verdict": "error", "error": str(exc)}
    if s1 not in (200, 201):
        return {"verdict": classify(s1, t1), "error": f"turn1 http_{s1}",
                "body": t1[:400]}
    d1 = json.loads(t1)
    resp_steps = d1.get("steps") or []
    answer1 = extract_text(resp_steps)

    echo_history = [user_step(_HID_Q1), *resp_steps, user_step(_HID_Q2)]
    drop_history = [user_step(_HID_Q1), model_step(answer1), user_step(_HID_Q2)]

    echo_runs = [_sig_turn2(echo_history, f"echo{i}") for i in range(repeats)]
    drop_runs = [_sig_turn2(drop_history, f"drop{i}") for i in range(repeats)]
    echo_nums = [_hid_number(r.get("answer", "")) for r in echo_runs]
    drop_nums = [_hid_number(r.get("answer", "")) for r in drop_runs]

    echo_same = len(set(echo_nums)) == 1 and all(echo_nums)
    drop_same = len(set(drop_nums)) == 1 and all(drop_nums)
    if echo_same and not drop_same:
        verdict = "signature_carries_state"
    elif echo_same and drop_same:
        verdict = "inconclusive_both_stable"     # nothing was actually hidden
    elif not echo_same and not drop_same:
        verdict = "signature_carries_nothing"    # echo remembers no better than drop
    else:
        verdict = "inconclusive"

    return {"model": _SIG_MODEL[0], "verdict": verdict,
            "turn1": {"answer": answer1,
                      "thought_tokens": _sig_usage(d1)["thought_tokens"],
                      "signature_steps": sum(1 for s in resp_steps if s.get("signature"))},
            "echo_numbers": echo_nums, "drop_numbers": drop_nums,
            "echo_consistent": echo_same, "drop_consistent": drop_same}


# --------------------------------------------------------------------------
# Latency probe: which field costs the seconds?
#
# The comparison run puts `interaction` (store:true + previous_interaction_id) at a
# ~4.7 s median and `interaction_stateless` (store:false, client-side history) at
# ~2.2 s -- on the same endpoint, same model, same conversation. But those two arms
# differ in three ways at once (who stores, who chains, what the payload carries), so
# the run cannot say which of the three buys the delay.
#
# This probe holds the conversation fixed and varies one field at a time. Every cell
# asks the same second question with the same visible history and the same decoding
# settings; only the mechanism differs:
#
#   gen_stateless       generateContent, full history                (control)
#   client_nostore      interactions, client history, store:false    (the fast arm)
#   client_store        interactions, client history, store:true     <- store alone
#   chained_store       interactions, previous_interaction_id, store:true
#   chained_nostore     interactions, previous_interaction_id, store:false <- read alone
#
# client_nostore vs client_store isolates the cost of *writing* the interaction.
# client_store vs chained_store isolates the cost of *reading* the stored history.

_LAT_SYSTEM = "You are a test fixture. Be terse."
_LAT_Q1 = "Name one European capital city. One word."
_LAT_Q2 = "Name a different one. One word."


def _lat_config() -> dict:
    """Decoding is pinned: latency that tracks how much the model chose to think is
    not latency the endpoint is responsible for."""
    return {"max_output_tokens": 32, "thinking_level": "low"}


def _lat_time(fn) -> tuple[float, dict]:
    t0 = time.monotonic()
    out = fn()
    return (time.monotonic() - t0) * 1000, out


def _lat_interaction(history: list, prev_id: str | None, store: bool) -> dict:
    body = {"model": _SIG_MODEL[0], "stream": False, "store": store,
            "system_instruction": _LAT_SYSTEM, "input": history,
            "generation_config": _lat_config()}
    if prev_id:
        body["previous_interaction_id"] = prev_id
    ms, (status, text) = _lat_time(lambda: _sig_post(body))
    out = {"ms": int(ms), "status": status, "bytes": len(json.dumps(body))}
    if status in (200, 201):
        d = json.loads(text)
        out["output_tokens"] = _sig_usage(d)["output_tokens"]
        out["thought_tokens"] = _sig_usage(d)["thought_tokens"]
        out["input_tokens"] = _sig_usage(d)["input_tokens"]
    else:
        out["body"] = text[:200]
    return out


def _lat_generate(history: list) -> dict:
    """The generateContent control: same conversation, the other endpoint."""
    url = (f"{gc_api_base()}/models/{_SIG_MODEL[0]}:generateContent")
    # generateContent spells the same two settings differently: thinking_level lives
    # under thinkingConfig, and a flat `thinkingLevel` is a 400.
    body = {"contents": history,
            "systemInstruction": {"parts": [{"text": _LAT_SYSTEM}]},
            "generationConfig": {"maxOutputTokens": 32,
                                 "thinkingConfig": {"thinkingLevel": "low"}}}
    def call():
        r = _session().post(url, data=json.dumps(body),
                            headers=_headers("apikey"), timeout=PROBE_TIMEOUT)
        return r.status_code, r.text
    ms, (status, text) = _lat_time(call)
    out = {"ms": int(ms), "status": status, "bytes": len(json.dumps(body))}
    if status in (200, 201):
        u = json.loads(text).get("usageMetadata", {})
        out["input_tokens"] = int(u.get("promptTokenCount", 0))
        out["output_tokens"] = int(u.get("candidatesTokenCount", 0))
        out["thought_tokens"] = int(u.get("thoughtsTokenCount", 0))
    else:
        out["body"] = text[:200]
    return out


def _median(xs: list) -> int:
    xs = sorted(xs)
    if not xs:
        return 0
    mid = len(xs) // 2
    return int(xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2)


def probe_latency_matrix(model: str = "", repeats: int = 5) -> dict:
    """Where do the seconds go: `store`, `previous_interaction_id`, or neither?

    One stored turn 1 to chain onto, then `repeats` calls per cell. Each cell sends
    the same second question; only the state mechanism varies. Returns the per-cell
    median, plus the two differences the arms confound.
    """
    _SIG_MODEL[0] = model or (PROBE_MODELS[0] if PROBE_MODELS else DEFAULT_MODEL)
    if not API_KEY:
        return {"verdict": "environment", "error": "GEMINI_API_KEY not set"}

    # Turn 1, stored: gives the chained cells something to point at, and gives the
    # client-history cells the model's own steps to echo.
    body1 = {"model": _SIG_MODEL[0], "stream": False, "store": True,
             "system_instruction": _LAT_SYSTEM,
             "input": single_step_input(_LAT_Q1),
             "generation_config": _lat_config()}
    s1, t1 = _sig_post(body1)
    if s1 not in (200, 201):
        return {"verdict": classify(s1, t1), "error": f"turn1 http_{s1}",
                "body": t1[:300]}
    d1 = json.loads(t1)
    iid, steps1 = d1.get("id"), d1.get("steps") or []
    answer1 = answer_text(steps1)

    history = [user_step(_LAT_Q1), *steps1, user_step(_LAT_Q2)]
    gen_history = [user_content(_LAT_Q1), model_content(answer1), user_content(_LAT_Q2)]
    new_turn = single_step_input(_LAT_Q2)

    cells = {
        "gen_stateless":   lambda: _lat_generate(gen_history),
        "client_nostore":  lambda: _lat_interaction(history, None, False),
        "client_store":    lambda: _lat_interaction(history, None, True),
        "chained_store":   lambda: _lat_interaction(new_turn, iid, True),
        # There is no chained_nostore cell: the API rejects it outright --
        #   400 "store must be true when previous_interaction_id is set."
        # So a chained conversation cannot opt out of the write. Whatever `store`
        # costs, previous_interaction_id pays it on every turn, by construction.
    }

    runs = {name: [fn() for _ in range(repeats)] for name, fn in cells.items()}
    out = {}
    for name, rs in runs.items():
        ok = [r for r in rs if r["status"] in (200, 201)]
        out[name] = {
            "median_ms": _median([r["ms"] for r in ok]),
            "ms": [r["ms"] for r in rs],
            "errors": len(rs) - len(ok),
            "request_bytes": rs[0]["bytes"],
            "input_tokens": ok[0].get("input_tokens") if ok else None,
            "output_tokens": [r.get("output_tokens") for r in ok],
            "thought_tokens": [r.get("thought_tokens") for r in ok],
        }

    return {
        "model": _SIG_MODEL[0], "repeats": repeats, "cells": out,
        # What each field costs, once the other two are held still.
        "store_cost_ms": out["client_store"]["median_ms"] - out["client_nostore"]["median_ms"],
        "chaining_cost_ms": out["chained_store"]["median_ms"] - out["client_store"]["median_ms"],
        "endpoint_cost_ms": out["client_nostore"]["median_ms"] - out["gen_stateless"]["median_ms"],
        "chained_nostore": "rejected: store must be true when "
                           "previous_interaction_id is set (400)",
    }


# --------------------------------------------------------------------------
# Streaming probe: does `store` hide behind the first token?
#
# With stream:false, store:true costs ~1.8 s over store:false -- measured, constant,
# and independent of payload size. The published examples all stream, which reports
# time-to-first-token, not time-to-complete. So: is the write happening *before* the
# first token (in which case streaming hides nothing and the guide's numbers are
# measuring a different thing), or *after* the last one (in which case a streaming
# client never waits for it, and only our stream:false arms pay)?
#
# Measured three ways per cell: TTFT (first SSE byte), time to the last event, and
# the token counts, so a slower cell cannot be blamed on a longer answer.


def _stream_once(store: bool, question: str, system: str) -> dict:
    body = {"model": _SIG_MODEL[0], "stream": True, "store": store,
            "system_instruction": system,
            "input": single_step_input(question),
            "generation_config": _lat_config()}
    t0 = time.monotonic()
    ttft = None
    events = 0
    try:
        with _session().post(INTERACTIONS_URL, data=json.dumps(body),
                             headers=_headers("apikey"), timeout=PROBE_TIMEOUT,
                             stream=True) as resp:
            if resp.status_code not in (200, 201):
                return {"status": resp.status_code, "body": resp.text[:200]}
            for line in resp.iter_lines():
                if not line:
                    continue
                if ttft is None:
                    ttft = (time.monotonic() - t0) * 1000
                events += 1
            done = (time.monotonic() - t0) * 1000
    except Exception as exc:
        return {"status": 0, "body": str(exc)}
    return {"status": 200, "ttft_ms": int(ttft or 0), "done_ms": int(done),
            "events": events}


def probe_stream_ttft(model: str = "", repeats: int = 5) -> dict:
    """Where does the store cost land when the response streams?

    Four cells: {stream:false, stream:true} x {store:false, store:true}, same first
    turn, no previous_interaction_id, decoding pinned. If streaming's TTFT is the
    same with and without `store`, the write is not on the path to the first token --
    it is a tail cost that only a non-streaming client waits for.
    """
    _SIG_MODEL[0] = model or (PROBE_MODELS[0] if PROBE_MODELS else DEFAULT_MODEL)
    if not API_KEY:
        return {"verdict": "environment", "error": "GEMINI_API_KEY not set"}

    out: dict = {}
    for store in (False, True):
        key = f"stream_store_{str(store).lower()}"
        runs = [_stream_once(store, _LAT_Q1, _LAT_SYSTEM) for _ in range(repeats)]
        ok = [r for r in runs if r.get("status") == 200]
        out[key] = {"ttft_median_ms": _median([r["ttft_ms"] for r in ok]),
                    "done_median_ms": _median([r["done_ms"] for r in ok]),
                    "ttft_ms": [r.get("ttft_ms") for r in runs],
                    "done_ms": [r.get("done_ms") for r in runs],
                    "errors": len(runs) - len(ok)}

        key = f"blocking_store_{str(store).lower()}"
        b = [_lat_interaction(single_step_input(_LAT_Q1), None, store)
             for _ in range(repeats)]
        bok = [r for r in b if r["status"] in (200, 201)]
        out[key] = {"done_median_ms": _median([r["ms"] for r in bok]),
                    "done_ms": [r["ms"] for r in b],
                    "errors": len(b) - len(bok)}

    s = out["stream_store_true"]["ttft_median_ms"] - out["stream_store_false"]["ttft_median_ms"]
    d = out["stream_store_true"]["done_median_ms"] - out["stream_store_false"]["done_median_ms"]
    blk = out["blocking_store_true"]["done_median_ms"] - out["blocking_store_false"]["done_median_ms"]
    return {"model": _SIG_MODEL[0], "repeats": repeats, "cells": out,
            "store_cost_on_ttft_ms": s,
            "store_cost_on_stream_completion_ms": d,
            "store_cost_when_blocking_ms": blk}


# --- cached probe ---------------------------------------------------------
# The page probes on load, so an uncached probe would bill a matrix of live calls
# on every refresh. Hold the last result for a while; the button forces a re-run.

_CACHE: dict = {}


def _cache_ttl() -> float:
    return float(os.environ.get("PROBE_CACHE_TTL", "600"))


def clear_cache() -> None:
    _CACHE.clear()


def probe_cached(force: bool = False) -> dict:
    """The probe result, reusing the last one while it is still fresh."""
    now = time.monotonic()
    hit = _CACHE.get("result")
    if hit and not force and (now - _CACHE["at"]) < _cache_ttl():
        return {**hit, "cached": True, "age_seconds": int(now - _CACHE["at"])}
    result = probe_interactions()
    _CACHE["result"] = result
    _CACHE["at"] = now
    return {**result, "cached": False, "age_seconds": 0}


def interaction_verdicts() -> dict:
    """What the last probe said about each model. Empty until one has run -- an
    unprobed model's interaction support is unknown, never assumed."""
    return dict((_CACHE.get("result") or {}).get("models") or {})
