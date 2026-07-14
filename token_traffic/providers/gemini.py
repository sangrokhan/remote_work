"""The Gemini Developer API, and the six ways of keeping a conversation going on it.

Every arm asks the same questions of the same model; the only thing that varies is
who stores the context. The client resends the whole history (`stateless`,
`interaction_stateless`), the server keeps it (`interaction`, `interaction_inline`),
the prefix lives in an explicit cache (`cached`), or nothing is kept at all
(`nocontext`, the lower bound). What each choice costs on the wire is the experiment.

One host, one auth, one network path: generativelanguage.googleapis.com with an API
key. Vertex/ADC is not an option here and must not come back -- it is a different
host on a different route with different latency, and it does not serve plain-model
Interactions at all, so half the arms would be measuring a network the other half
never touched.

Two rules this module exists to keep:

  1. When an arm keeps the history client-side, the model's turn goes back on the
     wire exactly as it came off it -- Gemini 3's `thought` step with its signature,
     or the parts carrying `thoughtSignature`. Rebuilding the turn from the answer
     text drops ~1 KB of upload per turn that a real client pays, and quietly turns
     the honest arm into a client nobody runs.
  2. Reasoning text is not the answer. A `thought` part never enters the transcript,
     never starts the TTFT clock, and never lands in a cache built from it.

Mock mode (TRAFFIC_MOCK=1 for the whole suite, or GEMINI_MOCK=1 for this provider
alone) makes no network call at all. Its responses are shaped like the real ones: a
thought step carrying a realistic-length signature, usage that grows with what the
payload actually carries, and a stored interaction holding the stream open ~1.8 s
after its last token while the server persists it. A mock run that let the stored
arms look free would be worse than no mock run.
"""

from __future__ import annotations

import json
import os
import secrets

from core import call, config, record, wire
from providers import base

NAME = "gemini"
DEFAULT_MODEL = "gemini-3.1-flash-lite"

ARMS = ("stateless", "nocontext", "cached", "interaction", "interaction_inline",
        "interaction_stateless")
# nocontext answers each question with no history whatsoever. It is the floor the
# other arms are measured against, not a way anyone would run a chat, so a default
# run leaves it out of the headline.
HEADLINE_ARMS = ("stateless", "cached", "interaction", "interaction_inline",
                 "interaction_stateless")

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
    """Shared parse, in core: a provider that reads TRAFFIC_MOCK its own way is a
    provider that can be live while the other is synthetic, in a run filed as one
    thing."""
    return config.is_mock(NAME)


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
# They are not interchangeable, and one definition of each lives here: two copies of
# a wire shape drifting apart would silently change what an arm sends, which is the
# one thing the experiment measures.

def user_content(text: str) -> dict:
    return {"role": "user", "parts": [{"text": text}]}


def model_content(text: str) -> dict:
    return {"role": "model", "parts": [{"text": text}]}


def user_step(text: str) -> dict:
    return {"type": "user_input", "content": [{"type": "text", "text": text}]}


def model_step(text: str) -> dict:
    return {"type": "model_output", "content": [{"type": "text", "text": text}]}


def model_content_from_response(data: dict, fallback_text: str = "") -> dict:
    """The model's turn, exactly as generateContent returned it.

    The candidate's `content` already is a Content: role `model`, parts carrying the
    text and the `thoughtSignature`. Echo it whole. Falls back to a rebuilt turn only
    when the call errored and there is no candidate to echo.
    """
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
    """The answer, and only the answer: the text of the `model_output` steps.

    With thinking summaries on, a `thought` step carries text of its own. Collecting
    every text leaf would staple the model's reasoning to its answer and put it in
    the transcript.
    """
    return "".join(extract_text(s.get("content"))
                   for s in (steps if isinstance(steps, list) else [])
                   if isinstance(s, dict) and s.get("type") == "model_output")


# --- what core.call needs from us: text_of, and rebuild ---------------------

def gen_text(event: dict) -> str:
    """Answer text in one streamed generateContent chunk. A part flagged `thought` is
    the model's reasoning, not its answer, so it must not start the TTFT clock."""
    return "".join(part.get("text") or ""
                   for cand in event.get("candidates") or []
                   for part in (cand.get("content") or {}).get("parts") or []
                   if not part.get("thought"))


def gen_response(events: list) -> dict:
    """The chunks, reassembled into the body a blocking call would have returned.

    The parts are kept as they arrived -- including the empty-text part that carries
    the thoughtSignature -- because that list *is* the model turn a client-side
    history echoes back. usageMetadata is cumulative, so the last one wins.
    """
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
    """The events, reassembled into the body a stream:false call would have returned.

    Steps are rebuilt by index: `step.start` declares the type, `step.delta` appends a
    signature (thought) or a text block (model_output). The completed event carries
    the id and the usage but *not* the steps (measured, docs/interactions-api-fields),
    so this is the only place the steps a client-side history must echo exist at all.
    """
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
# The two endpoints report the same five numbers under different names. Gemini calls
# reasoning "thoughts"; the record calls it reasoning_tokens, because a chart that
# reads one provider's vocabulary cannot compare two providers.

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
# Fixed timings, so a mock run's latency chart is stable, and shaped like the real
# thing, so an arm that pays a tail can still be told apart from one that does not.

MOCK_REQ_SENT_BASE_MS = 20
MOCK_UPLOAD_MS_PER_KB = 2           # what a bigger history really does buy: upload
MOCK_TTFB_MS = 200
MOCK_TTFT_MS = 300
MOCK_TTLT_MS = 800
# What store:true adds after the answer is already out. The real thing measured
# ~1.8 s (docs/interactions-api-fields.md); the mock keeps the shape so a mock run
# cannot pretend the stored arms are free.
MOCK_STORE_TAIL_MS = 1800
MOCK_HEADER_BYTES = 200             # request line + headers, roughly, on both sides


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _mock_signature(turn: int) -> str:
    """Stand-in for the opaque base64 signature a real thought step carries. Roughly
    the real length, because that blob is most of what echoing a model turn costs."""
    return f"MOCKSIG{turn:03d}" + ("A" * 60)


def _mock_upload_ms(req_bytes: int) -> int:
    return MOCK_REQ_SENT_BASE_MS + (req_bytes // 1024) * MOCK_UPLOAD_MS_PER_KB


def _mock_exchange(body: dict, response: dict, text: str, measure: str,
                   store_tail_ms: int = 0) -> call.Exchange:
    """A synthetic Exchange with the marks a real one would carry.

    A blocking pass has no TTFT to report -- the answer arrives all at once -- so
    `bytes` leaves the marks at zero exactly as core.call would.
    """
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
    # Only what the client actually put in `contents` is billed as input: a prefix
    # that lives in the cache is billed as cached, and a history that was never sent
    # is billed to nobody. That is the whole comparison, so the mock must count what
    # the payload carries and not what the conversation contains.
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
    # The billed input is the system prompt (re-sent every turn -- the server does not
    # keep it) plus every text leaf the `input` field actually carries. For the
    # chained arm that is one question; for the client-history arm it is the whole
    # conversation, and the curve has to bend, or the mock hides the arm's entire cost.
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
    for the marks. core.call decides which passes to make."""
    body: dict = {"contents": contents}
    if cached_content:
        body["cachedContent"] = cached_content
    if is_mock():
        # The hint never goes on the wire -- it stays out of `body` and out of the
        # recorded request -- it only tells the mock how big the prefix the cache is
        # holding for this turn would have been.
        return _mock_generate(body, turn, measure, cached_tokens_hint)
    return call.send(generate_url(model), auth_headers(), body,
                     measure=measure, text_of=gen_text,
                     stream_url=stream_generate_url(model),
                     stream_body=body, rebuild=gen_response)


def _interact(model: str, text: str, system: str, prev_id: str | None,
              measure: str, turn: int, store: bool = True,
              history: list | None = None) -> call.Exchange:
    """One interactions turn.

    Two ways to give the server a conversation. Chained: `input` is the new question
    alone and `previous_interaction_id` points at the interaction holding everything
    before it. Client-side (`history`): `input` is the whole conversation as a Step[]
    and nothing is stored, so there is no previous interaction to point at.

    `system_instruction` rides every turn either way -- it is interaction-scoped and
    the server does not keep it (measured, docs/interactions-api-fields.md), which is
    why the chained arm saves far less on the wire than it looks like it should.
    """
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
    """Build one cachedContent holding `contents`. Returns (exchange, cache).

    A cache build is a plain POST: nothing streams out of it, so it is always sent as
    a single blocking call whatever the run's `measure` is -- there is no first token
    to time. Skips (name None) below the minimum cacheable size rather than eating a
    400, and says so in the record.
    """
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
    the session rather than through core.call, which exists to record one."""
    if not name or is_mock() or "mock_" in name:
        return
    try:
        wire.session().delete(f"{api_base()}/{name}",
                              headers={"x-goog-api-key": api_key()}, timeout=60)
    except Exception:
        pass


# --- the arms -------------------------------------------------------------------

def _tick(on_progress, arm: str, phase: str, turn: int, turns: int) -> None:
    base.progress(on_progress, NAME, arm, phase, turn, turns)


def _rec(arm, phase, turn, question, measure, exchange, usage, extra=None) -> dict:
    return record.turn_record(NAME, arm, phase, turn, question, measure,
                              exchange, usage, extra)


def _cache_usage(cache: dict) -> dict:
    """A cache build is billed no input tokens: no answer came out of it.

    It used to report its size as `input_tokens`, which put a *size* in the column that
    everywhere else means *tokens billed as input for an answer*. core.metrics then
    summed the arm's prep rows and produced 19071 -- two transcript calls' real input
    tokens (4479 + 4762) added to two caches' sizes (4659 + 5171). A number describing
    nothing, in a column a reader is entitled to trust.

    So the billed columns are zero, which is true, and the size goes in the record's
    `cache_tokens` extra, which is where a size belongs.
    """
    return {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
            "reasoning_tokens": 0, "total_tokens": 0}


def _arm_stateless(model, system, steps, measure, on_progress) -> list[dict]:
    """The whole history every turn: system + q1..qk + the model's own a1..a(k-1).

    The model's turn is echoed exactly as generateContent sent it -- parts, and the
    thoughtSignature riding along with the text. Rebuilding it from the answer text
    would drop ~1 KB of upload a turn that a real client pays, from the one arm whose
    entire purpose is to pay for its own history.
    """
    recs, history = [], ([user_content(system)] if system else [])
    n = len(steps)
    for k, q in enumerate(steps, start=1):
        _tick(on_progress, "stateless", "steady", k, n)
        history.append(user_content(q))
        ex = _generate(model, list(history), measure, k)
        history.append(model_content_from_response(ex.response, ex.text))
        recs.append(_rec("stateless", "steady", k, q, measure, ex, _usage_gen(ex.response)))
    return recs


def _arm_nocontext(model, system, steps, measure, on_progress) -> list[dict]:
    """The new question only; the system prompt rides the first turn alone. Nobody
    keeps the conversation, so the model answers each question cold. Not a strategy --
    the floor everything else is measured against."""
    recs, n = [], len(steps)
    for k, q in enumerate(steps, start=1):
        _tick(on_progress, "nocontext", "steady", k, n)
        contents = ([user_content(system), user_content(q)]
                    if (k == 1 and system) else [user_content(q)])
        ex = _generate(model, contents, measure, k)
        recs.append(_rec("nocontext", "steady", k, q, measure, ex, _usage_gen(ex.response)))
    return recs


def _arm_cached(model, system, steps, measure, on_progress) -> list[dict]:
    """The prefix lives in an explicit cache, so turn k >= 2 sends only its question.

    Three stages. Prep replays the conversation once to get the model's real answers
    and builds cache k from the system prompt plus the first k real Q&A pairs -- a
    cache of a conversation that never happened would measure nothing. Every prep call
    is recorded under the `cachegen` phase and is never folded into the totals: it is
    setup, not traffic, and since each build re-uploads the whole prefix, n turns of
    it cost O(n^2) and would drown everything the arm exists to show. Then the steady
    turns, which are the measurement. Then the caches are deleted, because a cache
    left behind is a bill nobody is watching.
    """
    n = len(steps)
    recs: list[dict] = []

    # The transcript. These are real generateContent calls, so they are recorded --
    # an API call the run made and did not report is a hole in the evidence -- but
    # they are prep, and phased accordingly.
    transcript, history = [], ([user_content(system)] if system else [])
    for k, q in enumerate(steps, start=1):
        _tick(on_progress, "cached", "cachegen", k, n)
        history.append(user_content(q))
        ex = _generate(model, list(history), "bytes", k)
        history.append(model_content_from_response(ex.response, ex.text))
        transcript.append(ex.text)
        # A real generateContent call: real inference, real input tokens, real money.
        # `billed` is what keeps it from being summed together with the cache builds
        # beside it, whose token columns are zeros and whose size is a size.
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

    # Prep left a keep-alive connection open. The steady turns are what the pcap is
    # for, and a capture that opens onto an established connection has no SYN in it
    # and prep's own FIN in it instead.
    wire.reset_session()

    try:
        for k, q in enumerate(steps, start=1):
            _tick(on_progress, "cached", "steady", k, n)
            cache = caches[k - 2] if k >= 2 else None
            cache_id = cache["name"] if cache else None
            if cache_id:
                contents = [user_content(q)]        # the prefix is server-side
            else:
                # Turn 1 has no prior cache to point at, so it sends the system prompt
                # with its question, exactly as stateless would.
                contents = ([user_content(system)] if system else []) + [user_content(q)]
            ex = _generate(model, contents, measure, k, cached_content=cache_id,
                           cached_tokens_hint=cache["cached_tokens"] if cache else 0)
            recs.append(_rec("cached", "steady", k, q, measure, ex,
                             _usage_gen(ex.response), {"cache_id": cache_id}))
    finally:
        # Announce the teardown before doing any of it. The runner closes the capture
        # window on this event, so the DELETEs land outside the pcap: a capture that
        # contains them cannot be read as "this is what the arm's turns cost".
        _tick(on_progress, "cached", "teardown", n, n)
        # Even a run that blew up in the middle must not leave paid-for caches behind.
        if os.environ.get("KEEP_CACHE") != "1":
            for cache in caches:
                _delete_cache(cache.get("name"))
    return recs


def _arm_interactions(arm, model, system, steps, measure, on_progress, *,
                      inline_system: bool = False,
                      client_history: bool = False) -> list[dict]:
    """The Interactions API, in its three variants.

    Default (`interaction`): turn 1 opens the interaction, later turns pass
    previous_interaction_id and send only the new question -- but the system prompt
    goes up again every single turn, because system_instruction is interaction-scoped
    and the server does not store it.

    `inline_system`: the system prompt rides in the first user turn instead, which
    makes it part of the stored history, so every later turn sends only its question.
    The same content reaches the model; a different party stores it.

    `client_history` (`interaction_stateless`): store:false, no
    previous_interaction_id, and the whole conversation resent as a Step[] every turn.
    It holds the endpoint fixed and takes the server-side state away, so what is left
    between it and `interaction` is exactly what previous_interaction_id buys.
    """
    recs, n = [], len(steps)
    prev_id: str | None = None
    history: list = []
    for k, q in enumerate(steps, start=1):
        _tick(on_progress, arm, "steady", k, n)
        if inline_system:
            text = f"{system}\n\n{q}" if (k == 1 and system) else q
            sys_instruction = ""
        else:
            text, sys_instruction = q, system

        if client_history:
            history.append(user_step(text))
            sent_history, store = list(history), False
        else:
            sent_history, store = None, True

        ex = _interact(model, text, sys_instruction, prev_id, measure, k,
                       store=store, history=sent_history)
        steps_back = model_steps_from_response(ex.response, ex.text)
        # The answer is the model_output steps, never the thought step: with thinking
        # summaries on, the thought step carries text of its own, and stapling it to
        # the answer would put the model's reasoning into the transcript.
        text_out = answer_text(steps_back) or ex.text
        iid = (ex.response or {}).get("id")

        if client_history:
            # This arm's own answer, as the server sent it -- thought step, signature
            # and all. Not a step rebuilt from the text: that drops the signature and
            # under-reports what a real client uploads.
            history.extend(steps_back)
        elif iid:
            prev_id = iid                    # chain the next turn onto this one

        recs.append(_rec(arm, "steady", k, q, measure, ex,
                         _usage_interaction(ex.response),
                         {"interaction_id": iid, "response_text": text_out}))
    return recs


def run_arm(arm: str, model: str, system: str, steps: list[str],
            measure: str = "bytes", on_progress=None) -> list[dict]:
    """Replay the scenario over one arm: one record per turn, plus a `cachegen`
    record per prep call for the cached arm."""
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
