"""The Gemini Developer API, and the six ways of keeping a conversation going on it.

Ported from ``token_traffic/providers/gemini.py`` (DESIGN.md 5, A2) onto the
``aipt.backends.base.Backend`` protocol (connect/send_turn/close instead of
a single ``run_arm`` that owned the whole conversation -- see ``base.py``'s
module docstring for why the lifecycle was split).

Every arm asks the same questions of the same model; the only thing that varies is
who stores the context. The client resends the whole history (`stateless`,
`interaction_stateless`), the server keeps it (`interaction`, `interaction_inline`),
the prefix lives in an explicit cache (`cached`), or nothing is kept at all
(`nocontext`, the lower bound). What each choice costs on the wire is the experiment.

One host, one auth, one network path: generativelanguage.googleapis.com with an API
key.

Two rules this module exists to keep:

  1. When an arm keeps the history client-side, the model's turn goes back on the
     wire exactly as it came off it -- Gemini 3's `thought` step with its signature,
     or the parts carrying `thoughtSignature`. Rebuilding the turn from the answer
     text drops ~1 KB of upload per turn that a real client pays, and quietly turns
     the honest arm into a client nobody runs.
  2. Reasoning text is not the answer. A `thought` part never enters the transcript,
     never starts the TTFT clock, and never lands in a cache built from it.

Mock mode (TRAFFIC_MOCK=1 for the whole suite, or GEMINI_MOCK=1 for this backend
alone) makes no network call at all.

``connect``/``send_turn``/``close`` adaptation notes (what changed from
``run_arm``, and why):

  * ``stateless``, ``nocontext``, ``interaction``, ``interaction``-family arms map
    onto the lifecycle directly: each was already a per-turn loop carrying state
    (history / previous_interaction_id) forward turn to turn, so ``connect``
    initializes that state and ``send_turn`` advances it by exactly one step.
  * ``cached`` cannot: the original algorithm needs the *whole* conversation in
    advance -- it replays every question once to get real answers, then builds one
    cache per turn from the realized transcript, *before* any steady turn runs.
    ``send_turn`` only ever sees one question at a time, with no lookahead. The
    adaptation here builds the cache **online**: turn 1 has no cache yet and sends
    the system prompt with its question (as ``stateless`` would); after each turn
    finishes, a cache is built from the transcript accumulated *so far* and used by
    the *next* turn. This keeps the arm's cost shape (steady turns after the first
    reference a server-side cache instead of resending the prefix) while fitting a
    protocol that cannot see future turns. It is not the same schedule as the
    original two-pass ``_arm_cached`` (which this module keeps, unchanged, as
    ``run_arm`` below, for parity testing against the original fixture-driven
    tests) -- see ``GeminiBackend._send_turn_cached``.
"""

from __future__ import annotations

import json
import os
import secrets

from aipt.backends import base
from aipt.backends.public_ai import _cachebust as cachebust
from aipt.backends.public_ai import _call as call
from aipt.core import config, wire

NAME = "public_ai"
PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-3.1-flash-lite"

ARMS = ("stateless", "nocontext", "cached", "interaction", "interaction_inline",
        "interaction_stateless")
# nocontext answers each question with no history whatsoever. It is the floor the
# other arms are measured against, not a way anyone would run a chat, so a default
# run leaves it out of the headline.
HEADLINE_ARMS = ("stateless", "cached", "interaction", "interaction_inline",
                  "interaction_stateless")
# Arms that put the system prompt on the wire once and let somebody else keep it -- an
# explicit cache, or the stored first turn. A per-turn marker cannot vary on these: there
# is no per-turn send to vary.
PROMPT_SENT_ONCE_ARMS = ("cached", "interaction_inline", "nocontext")

CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "1800"))


# --- endpoints and auth -----------------------------------------------------
# Read at call time, never at import, so a test can point the host at a local server
# without reloading the module.

def api_host() -> str:
    return os.environ.get("GEMINI_API_HOST", "generativelanguage.googleapis.com")


def api_base() -> str:
    scheme = os.environ.get("GEMINI_API_SCHEME", "https")
    return f"{scheme}://{api_host()}/v1beta"


def api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "")


def auth_headers() -> dict:
    return {"Content-Type": "application/json", "x-goog-api-key": api_key()}


def generate_url(model: str) -> str:
    return f"{api_base()}/models/{model}:generateContent"


def stream_generate_url(model: str) -> str:
    return f"{api_base()}/models/{model}:streamGenerateContent?alt=sse"


def cache_url() -> str:
    return f"{api_base()}/cachedContents"


def interactions_url() -> str:
    return f"{api_base()}/interactions"


def is_mock() -> bool:
    """Shared parse, in core: a backend that reads TRAFFIC_MOCK its own way is a
    backend that can be live while another is synthetic, in a run filed as one
    thing."""
    return config.is_mock(PROVIDER)


def ready() -> tuple[bool, str]:
    if is_mock():
        return True, ""
    if not api_key():
        return False, "GEMINI_API_KEY not set (or run with TRAFFIC_MOCK=1)."
    return True, ""


# --- the two wire vocabularies ----------------------------------------------
# generateContent : {"role": "user"|"model", "parts": [{"text": ...}]}      Content
# interactions    : {"type": "user_input"|"model_output",
#                    "content": [{"type": "text", "text": ...}]}            Step
# They are not interchangeable, and one definition of each lives here.

def user_content(text: str) -> dict:
    return {"role": "user", "parts": [{"text": text}]}


def model_content(text: str) -> dict:
    return {"role": "model", "parts": [{"text": text}]}


def user_step(text: str) -> dict:
    return {"type": "user_input", "content": [{"type": "text", "text": text}]}


def model_step(text: str) -> dict:
    return {"type": "model_output", "content": [{"type": "text", "text": text}]}


def model_content_from_response(data: dict, fallback_text: str = "") -> dict:
    """The model's turn, exactly as generateContent returned it."""
    cands = (data or {}).get("candidates") or []
    content = cands[0].get("content") if cands else None
    if isinstance(content, dict) and content.get("parts"):
        return {"role": content.get("role", "model"), "parts": content["parts"]}
    return model_content(fallback_text)


def model_steps_from_response(data: dict, fallback_text: str = "") -> list:
    """The model's turn, exactly as the interactions endpoint returned it: the
    `thought` step with its signature, then the `model_output` step."""
    steps = (data or {}).get("steps")
    if isinstance(steps, list) and steps:
        return steps
    return [model_step(fallback_text)] if fallback_text else []


def extract_text(obj) -> str:
    """Every {"type": "text", "text": ...} leaf in a payload, in order."""
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


def answer_text(steps) -> str:
    """The answer, and only the answer: the text of the `model_output` steps."""
    return "".join(extract_text(s.get("content"))
                    for s in (steps if isinstance(steps, list) else [])
                    if isinstance(s, dict) and s.get("type") == "model_output")


# --- what call needs from us: text_of, and rebuild ---------------------

def gen_text(event: dict) -> str:
    """Answer text in one streamed generateContent chunk. A part flagged `thought` is
    the model's reasoning, not its answer, so it must not start the TTFT clock."""
    return "".join(part.get("text") or ""
                    for cand in event.get("candidates") or []
                    for part in (cand.get("content") or {}).get("parts") or []
                    if not part.get("thought"))


def gen_response(events: list) -> dict:
    """The chunks, reassembled into the body a blocking call would have returned."""
    parts, usage, role = [], {}, "model"
    for ev in events:
        for cand in ev.get("candidates") or []:
            content = cand.get("content") or {}
            role = content.get("role") or role
            parts.extend(content.get("parts") or [])
        if ev.get("usageMetadata"):
            usage = ev["usageMetadata"]
    return {"candidates": [{"content": {"role": role, "parts": parts}}],
            "usageMetadata": usage}


def interaction_text(event: dict) -> str:
    if event.get("event_type") != "step.delta":
        return ""
    return (event.get("delta") or {}).get("text") or ""


def interaction_response(events: list) -> dict:
    """The events, reassembled into the body a stream:false call would have returned."""
    steps: dict[int, dict] = {}
    order: list[int] = []
    iid, usage, status = "", {}, ""
    for ev in events:
        kind = ev.get("event_type")
        idx = ev.get("index")
        if kind == "step.start":
            steps[idx] = {"type": (ev.get("step") or {}).get("type", "")}
            order.append(idx)
        elif kind == "step.delta":
            step = steps.setdefault(idx, {"type": ""})
            delta = ev.get("delta") or {}
            if delta.get("signature"):
                step["signature"] = step.get("signature", "") + delta["signature"]
            if delta.get("text") is not None:
                content = step.setdefault("content", [{"type": "text", "text": ""}])
                content[0]["text"] += delta.get("text") or ""
        elif kind in ("interaction.created", "interaction.completed"):
            it = ev.get("interaction") or {}
            iid = it.get("id") or iid
            usage = it.get("usage") or usage
            status = it.get("status") or status
    return {"id": iid, "status": status, "usage": usage,
            "steps": [steps[i] for i in order if i in steps]}


# --- usage --------------------------------------------------------------------

def _usage_gen(response: dict) -> dict:
    u = (response or {}).get("usageMetadata") or {}
    prompt = int(u.get("promptTokenCount", 0))
    out = int(u.get("candidatesTokenCount", 0))
    return {
        "input_tokens": prompt,
        "cached_tokens": int(u.get("cachedContentTokenCount", 0)),
        "output_tokens": out,
        "reasoning_tokens": int(u.get("thoughtsTokenCount", 0)),
        "total_tokens": int(u.get("totalTokenCount", prompt + out)),
    }


def _usage_interaction(response: dict) -> dict:
    u = (response or {}).get("usage") or {}
    return {
        "input_tokens": int(u.get("total_input_tokens", 0)),
        "cached_tokens": int(u.get("total_cached_tokens", 0)),
        "output_tokens": int(u.get("total_output_tokens", 0)),
        "reasoning_tokens": int(u.get("total_thought_tokens", 0)),
        "total_tokens": int(u.get("total_tokens", 0)),
    }


# --- mock ---------------------------------------------------------------------

MOCK_REQ_SENT_BASE_MS = 20
MOCK_UPLOAD_MS_PER_KB = 2
MOCK_TTFB_MS = 200
MOCK_TTFT_MS = 300
MOCK_TTLT_MS = 800
MOCK_STORE_TAIL_MS = 1800
MOCK_HEADER_BYTES = 200


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _mock_signature(turn: int) -> str:
    return f"MOCKSIG{turn:03d}" + ("A" * 60)


def _mock_upload_ms(req_bytes: int) -> int:
    return MOCK_REQ_SENT_BASE_MS + (req_bytes // 1024) * MOCK_UPLOAD_MS_PER_KB


def _mock_exchange(body: dict, response: dict, text: str, measure: str,
                    store_tail_ms: int = 0) -> call.Exchange:
    request_json = json.dumps(body)
    response_json = json.dumps(response)
    req_bytes = len(request_json.encode("utf-8"))
    resp_bytes = len(response_json.encode("utf-8"))
    end = MOCK_TTLT_MS + store_tail_ms
    streamed = measure in ("latency", "both")
    return call.Exchange(
        status=200, error="",
        wire_sent=req_bytes + MOCK_HEADER_BYTES,
        wire_recv=resp_bytes + MOCK_HEADER_BYTES,
        req_payload_bytes=req_bytes, resp_payload_bytes=resp_bytes,
        req_sent_ms=_mock_upload_ms(req_bytes) if streamed else 0,
        ttfb_ms=MOCK_TTFB_MS if streamed else 0,
        ttft_ms=MOCK_TTFT_MS if streamed else 0,
        ttlt_ms=MOCK_TTLT_MS if streamed else 0,
        turn_end_ms=end if streamed else 0,
        elapsed_ms=end, text=text, response=response,
        request_json=request_json, response_json=response_json,
    )


def _mock_generate(body: dict, turn: int, measure: str,
                    cached_tokens: int = 0) -> call.Exchange:
    contents = body.get("contents") or []
    last_q = ""
    for c in reversed(contents):
        if c.get("role") == "user":
            last_q = "".join(p.get("text", "") for p in c.get("parts", []))[:40]
            break
    text = f"(mock answer to: {last_q}) " + ("lorem ipsum " * 20)
    sent = "".join(p.get("text", "") for c in contents for p in c.get("parts", []))
    cached = cached_tokens if body.get("cachedContent") else 0
    prompt = _approx_tokens(sent)
    response = {
        "candidates": [{"content": {"role": "model", "parts": [
            {"text": text, "thoughtSignature": _mock_signature(turn)}]}}],
        "usageMetadata": {"promptTokenCount": prompt, "candidatesTokenCount": 64,
                           "cachedContentTokenCount": cached, "thoughtsTokenCount": 0,
                           "totalTokenCount": prompt + 64 + cached},
    }
    return _mock_exchange(body, response, text, measure)


def _mock_interaction(body: dict, turn: int, measure: str) -> call.Exchange:
    text = (f"(mock interaction answer, turn {turn}, "
            f"prev={body.get('previous_interaction_id') or 'none'}) "
            + ("lorem ipsum " * 12))
    carried = body.get("system_instruction", "") + extract_text(body.get("input"))
    prompt = _approx_tokens(carried)
    steps = [{"signature": _mock_signature(turn), "type": "thought"},
             {"content": [{"text": text, "type": "text"}], "type": "model_output"}]
    response = {"id": f"mock_interaction_{turn:03d}", "status": "completed",
                "steps": steps,
                "usage": {"total_input_tokens": prompt, "total_cached_tokens": 0,
                          "total_output_tokens": 40, "total_thought_tokens": 0,
                          "total_tokens": prompt + 40}}
    tail = MOCK_STORE_TAIL_MS if body.get("store") else 0
    return _mock_exchange(body, response, text, measure, store_tail_ms=tail)


# --- one call ------------------------------------------------------------------

def _generate(model: str, contents: list, measure: str, turn: int,
              cached_content: str | None = None,
              cached_tokens_hint: int = 0) -> call.Exchange:
    """One generateContent turn: blocking for bytes, `:streamGenerateContent?alt=sse`
    for the marks. call.send decides which passes to make."""
    body: dict = {"contents": contents}
    if cached_content:
        body["cachedContent"] = cached_content
    if is_mock():
        return _mock_generate(body, turn, measure, cached_tokens_hint)
    return call.send(generate_url(model), auth_headers(), body,
                      measure=measure, text_of=gen_text,
                      stream_url=stream_generate_url(model),
                      stream_body=body, rebuild=gen_response)


def _interact(model: str, text: str, system: str, prev_id: str | None,
              measure: str, turn: int, store: bool = True,
              history: list | None = None) -> call.Exchange:
    """One interactions turn."""
    body: dict = {
        "model": model,
        "stream": False,
        "store": store,
        "input": list(history) if history is not None else [user_step(text)],
    }
    if system:
        body["system_instruction"] = system
    if prev_id and history is None:
        body["previous_interaction_id"] = prev_id
    if is_mock():
        return _mock_interaction(body, turn, measure)
    return call.send(interactions_url(), auth_headers(), body,
                      measure=measure, text_of=interaction_text,
                      stream_body={**body, "stream": True},
                      rebuild=interaction_response)


def _create_cache(model: str, contents: list, measure: str) -> tuple[call.Exchange, dict]:
    """Build one cachedContent holding `contents`. Returns (exchange, cache)."""
    body = {"model": f"models/{model}", "contents": contents,
            "ttl": f"{CACHE_TTL_SECONDS}s"}
    approx = _approx_tokens("".join(p.get("text", "") for c in contents
                                     for p in c.get("parts", [])))
    floor = int(os.environ.get("MIN_CACHE_TOKENS", "2048"))
    if approx < floor:
        exchange = call.Exchange(
            status=0, error=f"below_min ({approx} < {floor} tokens)",
            request_json=json.dumps(body), response={}, response_json="", text="")
        return exchange, {"name": None, "cached_tokens": 0}
    if is_mock():
        name = f"cachedContents/mock_{secrets.token_hex(4)}"
        response = {"name": name,
                    "usageMetadata": {"totalTokenCount": approx}}
        exchange = _mock_exchange(body, response, "", "bytes")
        return exchange, {"name": name, "cached_tokens": approx}
    exchange = call.send(cache_url(), auth_headers(), body,
                          measure="bytes", text_of=lambda _event: "")
    data = exchange.response or {}
    tokens = int((data.get("usageMetadata") or {}).get("totalTokenCount", approx))
    return exchange, {"name": data.get("name"), "cached_tokens": tokens}


def _delete_cache(name: str) -> None:
    """Best-effort teardown. Not a turn and not a measurement, so it goes straight at
    the session rather than through call.send, which exists to record one."""
    if not name or is_mock() or "mock_" in name:
        return
    try:
        wire.session().delete(f"{api_base()}/{name}",
                               headers={"x-goog-api-key": api_key()}, timeout=60)
    except Exception:
        pass


# --- record helper ---------------------------------------------------------

def _tick(on_progress, arm: str, phase: str, turn: int, turns: int) -> None:
    base.progress(on_progress, NAME, arm, phase, turn, turns)


def _rec(arm, phase, turn, question, measure, exchange, usage, extra=None) -> dict:
    """One row, backend-tagged ``public_ai`` with ``engine: gemini`` in ``extra`` --
    the turns CSV is shared across public_ai's two engines (gemini/openai) and the
    engine name is what tells them apart within one backend."""
    from aipt.backends.record import turn_record
    merged_extra = {"engine": PROVIDER}
    if extra:
        merged_extra.update(extra)
    return turn_record(NAME, arm, phase, turn, question, measure,
                        exchange, usage, extra=merged_extra)


def _cache_usage(cache: dict) -> dict:
    return {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
            "reasoning_tokens": 0, "total_tokens": 0}


# --- the arms (full-conversation replay, kept for parity tests) -------------

def _arm_stateless(model, system, steps, measure, on_progress) -> list[dict]:
    recs, history = [], []
    n = len(steps)
    for k, q in enumerate(steps, start=1):
        _tick(on_progress, "stateless", "steady", k, n)
        history.append(user_content(q))
        prefix = [user_content(cachebust.per_turn(system, k))] if system else []
        ex = _generate(model, prefix + history, measure, k)
        history.append(model_content_from_response(ex.response, ex.text))
        recs.append(_rec("stateless", "steady", k, q, measure, ex, _usage_gen(ex.response)))
    return recs


def _arm_nocontext(model, system, steps, measure, on_progress) -> list[dict]:
    recs, n = [], len(steps)
    for k, q in enumerate(steps, start=1):
        _tick(on_progress, "nocontext", "steady", k, n)
        contents = ([user_content(cachebust.per_turn(system, k)), user_content(q)]
                    if (k == 1 and system) else [user_content(q)])
        ex = _generate(model, contents, measure, k)
        recs.append(_rec("nocontext", "steady", k, q, measure, ex, _usage_gen(ex.response)))
    return recs


def _arm_cached(model, system, steps, measure, on_progress) -> list[dict]:
    """The prefix lives in an explicit cache, so turn k >= 2 sends only its question."""
    n = len(steps)
    recs: list[dict] = []

    transcript, history = [], ([user_content(system)] if system else [])
    for k, q in enumerate(steps, start=1):
        _tick(on_progress, "cached", "cachegen", k, n)
        history.append(user_content(q))
        ex = _generate(model, list(history), "bytes", k)
        history.append(model_content_from_response(ex.response, ex.text))
        transcript.append(ex.text)
        recs.append(_rec("cached", "cachegen", k, q, measure, ex,
                          _usage_gen(ex.response),
                          {"kind": "transcript", "billed": True}))

    off = 1 if system else 0
    caches = []
    for k in range(1, n + 1):
        _tick(on_progress, "cached", "cachegen", k, n)
        ex, cache = _create_cache(model, history[:off + 2 * k], measure)
        caches.append(cache)
        recs.append(_rec("cached", "cachegen", k, "", measure, ex,
                          _cache_usage(cache),
                          {"kind": "cache_create", "billed": False,
                           "cache_tokens": cache["cached_tokens"],
                           "cache_id": cache["name"],
                           "skipped": cache["name"] is None}))

    wire.reset_session()

    try:
        for k, q in enumerate(steps, start=1):
            _tick(on_progress, "cached", "steady", k, n)
            cache = caches[k - 2] if k >= 2 else None
            cache_id = cache["name"] if cache else None
            if cache_id:
                contents = [user_content(q)]
            else:
                contents = ([user_content(system)] if system else []) + [user_content(q)]
            ex = _generate(model, contents, measure, k, cached_content=cache_id,
                            cached_tokens_hint=cache["cached_tokens"] if cache else 0)
            recs.append(_rec("cached", "steady", k, q, measure, ex,
                              _usage_gen(ex.response), {"cache_id": cache_id}))
    finally:
        _tick(on_progress, "cached", "teardown", n, n)
        if os.environ.get("KEEP_CACHE") != "1":
            for cache in caches:
                _delete_cache(cache.get("name"))
    return recs


def _arm_interactions(arm, model, system, steps, measure, on_progress, *,
                       inline_system: bool = False,
                       client_history: bool = False) -> list[dict]:
    recs, n = [], len(steps)
    prev_id: str | None = None
    history: list = []
    for k, q in enumerate(steps, start=1):
        _tick(on_progress, arm, "steady", k, n)
        if inline_system:
            text = (f"{cachebust.per_turn(system, k)}\n\n{q}"
                    if (k == 1 and system) else q)
            sys_instruction = ""
        else:
            text, sys_instruction = q, cachebust.per_turn(system, k)

        if client_history:
            history.append(user_step(text))
            sent_history, store = list(history), False
        else:
            sent_history, store = None, True

        ex = _interact(model, text, sys_instruction, prev_id, measure, k,
                        store=store, history=sent_history)
        steps_back = model_steps_from_response(ex.response, ex.text)
        text_out = answer_text(steps_back) or ex.text
        iid = (ex.response or {}).get("id")

        if client_history:
            history.extend(steps_back)
        elif iid:
            prev_id = iid

        recs.append(_rec(arm, "steady", k, q, measure, ex,
                          _usage_interaction(ex.response),
                          {"interaction_id": iid, "response_text": text_out}))
    return recs


def run_arm(arm: str, model: str, system: str, steps: list[str],
            measure: str = "bytes", on_progress=None) -> list[dict]:
    """Replay the scenario over one arm: one record per turn, plus a `cachegen`
    record per prep call for the cached arm.

    Kept for parity testing against the original ``token_traffic`` behaviour and
    for scenario-replay style callers that already have the full step list up
    front. New client code should prefer :class:`GeminiBackend`'s
    connect/send_turn/close lifecycle (DESIGN.md 4.5).
    """
    model = model or DEFAULT_MODEL
    if arm == "stateless":
        return _arm_stateless(model, system, steps, measure, on_progress)
    if arm == "nocontext":
        return _arm_nocontext(model, system, steps, measure, on_progress)
    if arm == "cached":
        return _arm_cached(model, system, steps, measure, on_progress)
    if arm == "interaction":
        return _arm_interactions("interaction", model, system, steps, measure,
                                  on_progress)
    if arm == "interaction_inline":
        return _arm_interactions("interaction_inline", model, system, steps, measure,
                                  on_progress, inline_system=True)
    if arm == "interaction_stateless":
        return _arm_interactions("interaction_stateless", model, system, steps,
                                  measure, on_progress, client_history=True)
    raise ValueError(f"unknown gemini arm: {arm!r} (known: {', '.join(ARMS)})")


# --- Backend protocol: connect / send_turn / close --------------------------

class GeminiBackend:
    """``aipt.backends.base.Backend`` over the Gemini Developer API.

    See the module docstring for how each arm's state maps onto
    connect/send_turn/close, and for the ``cached`` arm's online-caching
    adaptation (the one behavioural difference from ``run_arm``'s two-pass
    ``_arm_cached``).
    """

    NAME = NAME
    DEFAULT_MODEL = DEFAULT_MODEL
    ARMS = ARMS
    HEADLINE_ARMS = HEADLINE_ARMS
    transport = base.DEFAULT_TRANSPORT

    def __init__(self) -> None:
        self._arm: str | None = None
        self._model: str = DEFAULT_MODEL
        self._system: str = ""
        self._turn = 0
        # stateless / cached
        self._history: list = []
        # interaction family
        self._prev_id: str | None = None
        self._interaction_history: list = []
        # cached: prefix-cache built after each turn, referenced by the next one
        self._cache_id: str | None = None
        self._cache_tokens = 0
        self._cache_history: list = []
        self._caches_built: list[str] = []

    def ready(self) -> tuple[bool, str]:
        return ready()

    def api_host(self) -> str:
        return api_host()

    def connect(self, arm: str, model: str, system: str) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown gemini arm: {arm!r} (known: {', '.join(ARMS)})")
        self._arm = arm
        self._model = model or DEFAULT_MODEL
        self._system = system or ""
        self._turn = 0
        self._history = []
        self._prev_id = None
        self._interaction_history = []
        self._cache_id = None
        self._cache_tokens = 0
        self._cache_history = [user_content(self._system)] if self._system else []
        self._caches_built = []

    def send_turn(self, turn: int, question: str, measure: str, on_progress=None):
        if self._arm is None:
            raise RuntimeError("send_turn called before connect")
        self._turn = turn
        base.progress(on_progress, NAME, self._arm, "steady", turn, turn)

        if self._arm == "stateless":
            ex = self._send_turn_stateless(turn, question, measure)
        elif self._arm == "nocontext":
            ex = self._send_turn_nocontext(turn, question, measure)
        elif self._arm == "cached":
            ex = self._send_turn_cached(turn, question, measure)
        elif self._arm == "interaction":
            ex = self._send_turn_interaction(turn, question, measure, inline_system=False,
                                               client_history=False)
        elif self._arm == "interaction_inline":
            ex = self._send_turn_interaction(turn, question, measure, inline_system=True,
                                               client_history=False)
        elif self._arm == "interaction_stateless":
            ex = self._send_turn_interaction(turn, question, measure, inline_system=False,
                                               client_history=True)
        else:  # pragma: no cover -- guarded in connect()
            raise ValueError(f"unknown gemini arm: {self._arm!r}")
        return ex

    def close(self) -> None:
        if self._arm == "cached" and self._caches_built:
            if os.environ.get("KEEP_CACHE") != "1":
                for name in self._caches_built:
                    _delete_cache(name)
        self._arm = None

    # -- per-arm single-turn steps -------------------------------------------

    def _send_turn_stateless(self, turn: int, question: str, measure: str):
        self._history.append(user_content(question))
        prefix = ([user_content(cachebust.per_turn(self._system, turn))]
                   if self._system else [])
        ex = _generate(self._model, prefix + self._history, measure, turn)
        self._history.append(model_content_from_response(ex.response, ex.text))
        return ex

    def _send_turn_nocontext(self, turn: int, question: str, measure: str):
        contents = ([user_content(cachebust.per_turn(self._system, turn)),
                     user_content(question)]
                    if (turn == 1 and self._system) else [user_content(question)])
        return _generate(self._model, contents, measure, turn)

    def _send_turn_cached(self, turn: int, question: str, measure: str):
        """Online-caching adaptation of ``_arm_cached`` -- see module docstring."""
        if self._cache_id:
            contents = [user_content(question)]
        else:
            contents = ([user_content(self._system)] if self._system else
                        []) + [user_content(question)]
        ex = _generate(self._model, contents, measure, turn,
                        cached_content=self._cache_id,
                        cached_tokens_hint=self._cache_tokens)

        # Grow the transcript this turn contributed, then build (or rebuild) the
        # cache that the *next* turn will reference.
        self._cache_history.append(user_content(question))
        self._cache_history.append(
            model_content_from_response(ex.response, ex.text))
        _, cache = _create_cache(self._model, list(self._cache_history), measure)
        if cache.get("name"):
            self._cache_id = cache["name"]
            self._cache_tokens = cache["cached_tokens"]
            self._caches_built.append(cache["name"])
        return ex

    def _send_turn_interaction(self, turn: int, question: str, measure: str, *,
                                inline_system: bool, client_history: bool):
        if inline_system:
            text = (f"{cachebust.per_turn(self._system, turn)}\n\n{question}"
                    if (turn == 1 and self._system) else question)
            sys_instruction = ""
        else:
            text, sys_instruction = question, cachebust.per_turn(self._system, turn)

        if client_history:
            self._interaction_history.append(user_step(text))
            sent_history, store = list(self._interaction_history), False
        else:
            sent_history, store = None, True

        ex = _interact(self._model, text, sys_instruction, self._prev_id, measure,
                        turn, store=store, history=sent_history)
        steps_back = model_steps_from_response(ex.response, ex.text)
        iid = (ex.response or {}).get("id")

        if client_history:
            self._interaction_history.extend(steps_back)
        elif iid:
            self._prev_id = iid
        return ex


#: Module-level singleton -- the client only ever needs one live connection at a
#: time (see aipt.backends.base.Backend's docstring on preferring a singleton).
BACKEND = GeminiBackend()
