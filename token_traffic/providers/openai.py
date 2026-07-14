"""OpenAI: three arms over one conversation, and the gap between bytes and billing.

  chat_stateless       POST /v1/chat/completions
                       messages = [system, u1, a1, ..., uk]   -> uploads O(N^2)

  responses_stateless  POST /v1/responses, store=false
                       input    = [system, u1, a1, ..., uk]   -> uploads O(N^2)
                       The control arm: the same payload as chat_stateless on a
                       different endpoint, so any byte gap against
                       responses_stateful is about server-side state and not about
                       which endpoint the bytes went through.

  responses_stateful   POST /v1/conversations once, seeded with the system prompt
                       POST /v1/responses, conversation=conv_..., input = [uk]
                       -> uploads O(N). The server holds the history.

What the experiment is about: on the stateful arm the uploaded bytes collapse
while `usage.input_tokens` does not. OpenAI's own documentation says why — all
previous input tokens for responses in the chain are billed as input tokens. The
bytes can already be saved; the billing does not follow. That gap is the finding,
and a mock run must not be able to pretend it away.

Measurement choices that are not incidental:

  - the `requests` library, never the official SDK. The SDK rides on httpx, and
    core.wire's socket counter is an http.client/urllib3 subclass that cannot
    attach to it. Going through core.call keeps every byte on a counted socket.
  - the system prompt is byte-identical on every turn *of an arm*. OpenAI's prompt
    cache matches on an exact prefix, so a timestamp or a turn counter that moved
    between turns would miss on every turn and turn cached_tokens — and every
    number derived from it — into noise. Between *arms* the prompt is deliberately
    not identical: core.cachebust puts a per-(run, arm) marker in front of it, so
    an arm cannot be answered from the prefix its neighbour left warm.
  - `prompt_cache_key` is pinned per arm and per run. It is a routing hint, not a
    namespace: unkeyed, whether an identical prefix hits depends on which node the
    call lands on (measured live: 2/5 unkeyed, 4/5 keyed), but a distinct key never
    stopped one arm from reading another's cache — `responses_stateful` was billed
    4224 cached tokens on its own turn 1, off `responses_stateless`'s prefix, with
    the keys already distinct. Isolation comes from the prefix (core.cachebust);
    the key only keeps a run's own turns landing on the node holding their cache.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from core import cachebust, config
from core.call import Exchange, send
from core.record import turn_record
from providers import base

if TYPE_CHECKING:
    import sys

    from providers.base import Provider

    # This module *is* the provider: the protocol's functions are module-level, so
    # the module object is what a runner holds. Stated here so a drift between the
    # protocol and these functions is a type error rather than a runtime surprise.
    _conforms: Provider = sys.modules[__name__]  # type: ignore[assignment]


NAME = "openai"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-nano")

ARMS = ("chat_stateless", "responses_stateless", "responses_stateful")
HEADLINE_ARMS = ARMS

MAX_OUTPUT_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "400"))
# Empty for a non-reasoning model: the parameter must not be sent at all, or the
# call 400s. Only a reasoning model that accepts an effort level should set this.
REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "")
TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "180"))


def base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def api_host() -> str:
    """The host a capture filters on. Derived from base_url so that pointing the
    provider at a proxy or a local fake also points the pcap at it."""
    from urllib.parse import urlparse
    return urlparse(base_url()).hostname or "api.openai.com"


def mock_mode() -> bool:
    """TRAFFIC_MOCK is the suite-wide switch; OPENAI_MOCK turns this one provider
    into a mock while the other still talks to its API. The parse is core's, and
    shared: two providers reading the same variable differently is how half a run gets
    billed while the other half is synthetic."""
    return config.is_mock(NAME)


def ready() -> tuple[bool, str]:
    if mock_mode():
        return True, "mock mode: no call leaves the process"
    if not os.environ.get("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY is not set"
    return True, ""


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
        "Content-Type": "application/json",
    }


def _cache_key(arm: str) -> str:
    """Distinct per arm, and per run when cache-busting is on.

    Rotating it with the run matters even though the prefix already differs: an
    un-rotated key routes this run's calls at the node holding *last* run's prefix,
    which is a node with nothing of ours on it. Keeping the key with the prefix keeps
    an arm's own turns landing where its own turn 1 left its cache.
    """
    tag = cachebust.tag(NAME, arm)
    return f"tt-{tag}-{arm}" if tag else f"tt-{NAME}-{arm}"


# ---------------------------------------------------------------- request bodies

# Obfuscation pads the streamed deltas with random characters to normalize payload
# sizes — a side-channel mitigation, and it is ON by default. It is also why the
# bytes in this suite come from a blocking pass and only the marks come from a
# streamed one: padded frames make a streamed byte count a measurement of OpenAI's
# padding policy rather than of the conversation. We switch it off wherever we
# stream anyway, so that what does come back is at least the model's own bytes.
_CHAT_STREAM_OPTS = {"include_obfuscation": False, "include_usage": True}
# Responses puts usage on response.completed, so it has no include_usage.
_RESP_STREAM_OPTS = {"include_obfuscation": False}


def _chat_body(model: str, system: str, history: list[dict], question: str,
               arm: str, *, stream: bool = False) -> dict:
    body: dict = {
        "model": model,
        "messages": ([{"role": "system", "content": system}]
                     + history
                     + [{"role": "user", "content": question}]),
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
        "stream": stream,
        "prompt_cache_key": _cache_key(arm),
    }
    if stream:
        # Without include_usage the chat endpoint sends no usage object at all in a
        # stream, and every token number in the record would come back zero.
        body["stream_options"] = dict(_CHAT_STREAM_OPTS)
    if REASONING_EFFORT:
        body["reasoning_effort"] = REASONING_EFFORT
    return body


def _responses_body(model: str, items: list[dict], arm: str, *, store: bool,
                    conversation: str | None = None, stream: bool = False) -> dict:
    body: dict = {
        "model": model,
        "input": items,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "stream": stream,
        "store": store,
        "prompt_cache_key": _cache_key(arm),
    }
    if stream:
        body["stream_options"] = dict(_RESP_STREAM_OPTS)
    if conversation:
        body["conversation"] = conversation
    if REASONING_EFFORT:
        body["reasoning"] = {"effort": REASONING_EFFORT}
    return body


def _streamed(body: dict) -> dict:
    return {**body, "stream": True,
            **({"stream_options": dict(_CHAT_STREAM_OPTS)} if "messages" in body
               else {"stream_options": dict(_RESP_STREAM_OPTS)})}


# ------------------------------------------------------------------ reading back

def _chat_text_of(event: dict) -> str:
    """The answer text in one chat SSE frame, and nothing else.

    The first frame carries role with content "" — not a token, and it must not
    start the TTFT clock.
    """
    choices = event.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("delta") or {}).get("content") or ""


def _responses_text_of(event: dict) -> str:
    """The answer text in one Responses SSE frame, and nothing else.

    A reasoning summary arrives as response.reasoning_summary_text.delta. It is not
    the answer: it must not enter the transcript and must not start the TTFT clock.
    """
    if event.get("type") == "response.output_text.delta":
        return event.get("delta") or ""
    return ""


def _rebuild_chat(events: list) -> dict:
    """The blocking body a streamed chat call never sent.

    Usage rides a trailing frame with choices: [] that arrives *after*
    finish_reason, so it must be picked up from wherever it lands, not from the
    frame that ended the answer.
    """
    text, usage, rid = [], {}, ""
    for ev in events:
        text.append(_chat_text_of(ev))
        if ev.get("usage"):
            usage = ev["usage"]
        rid = rid or ev.get("id", "")
    return {
        "id": rid,
        "choices": [{"message": {"role": "assistant", "content": "".join(text)}}],
        "usage": usage,
    }


def _rebuild_responses(events: list) -> dict:
    """response.completed carries the whole Response object, output items included —
    reasoning items among them. That is what the client-side arms must echo back, so
    the rebuilt body is that object, not a message reassembled from the deltas."""
    for ev in events:
        if ev.get("type") == "response.completed":
            return ev.get("response") or {}
    text = "".join(_responses_text_of(ev) for ev in events)
    return {
        "id": "",
        "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": text}]}],
        "usage": {},
    }


def _usage(data: dict, *, chat: bool) -> dict:
    """The two endpoints name the same numbers differently. One shape out, so no
    metric downstream has to care which endpoint produced it."""
    u = data.get("usage") or {}
    if chat:
        out = {
            "input_tokens": u.get("prompt_tokens", 0),
            "output_tokens": u.get("completion_tokens", 0),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
        }
    else:
        out = {
            "input_tokens": u.get("input_tokens", 0),
            "output_tokens": u.get("output_tokens", 0),
            "cached_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens", 0),
            "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0),
        }
    out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
    return out


def _chat_text(data: dict) -> str:
    choices = data.get("choices") or [{}]
    return (choices[0].get("message") or {}).get("content") or ""


def _responses_text(data: dict) -> str:
    text = ""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text += part.get("text", "")
    return text


def _echo_items(data: dict) -> list[dict]:
    """What an arm that keeps the history client-side must put back on the wire.

    Every output item, verbatim — a reasoning item included, with whatever opaque
    id or encrypted content it came with. Rebuilding the turn from the answer text
    would drop the reasoning item, and the arm would under-report what a real client
    uploads (base.Provider, rule 1). If the model returned nothing but a message,
    this is exactly the assistant turn the old client rebuilt by hand.
    """
    return list(data.get("output") or [])


# ------------------------------------------------------------------------- arms

def run_arm(arm, model, system, steps, measure, on_progress=None) -> list[dict]:
    """Replay the conversation on one arm.

    `steps` are the questions. `on_progress(record)` is called with each finished
    record; everything a caller could want to print is already in it.

    The stateful arm's conversation create is a prep record with phase "setup": its
    bytes are counted and reported, not hidden, but core.metrics keeps prep out of
    the totals because it is setup, not traffic.

    One caveat this arm cannot design away: `measure="both"` sends the turn twice,
    and on responses_stateful both passes carry `conversation=`, which OpenAI
    appends to (store=false is not allowed alongside a conversation). So `both`
    writes each turn into the server-side history twice and inflates input_tokens
    from the next turn on. Use `bytes` or `latency` on that arm, or read its token
    series knowing it is doubled.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")

    records: list[dict] = []
    history: list[dict] = []
    conversation = ""

    n = len(steps)
    if arm == "responses_stateful":
        base.progress(on_progress, NAME, arm, "setup", 1, n)
        rec, conversation = _create_conversation(system, measure)
        records.append(rec)

    for k, question in enumerate(steps, start=1):
        base.progress(on_progress, NAME, arm, "steady", k, n)

        if arm == "chat_stateless":
            url = f"{base_url()}/chat/completions"
            body = _chat_body(model, system, history, question, arm)
            text_of, rebuild = _chat_text_of, _rebuild_chat
        elif arm == "responses_stateless":
            url = f"{base_url()}/responses"
            items = ([{"role": "system", "content": system}]
                     + history
                     + [{"role": "user", "content": question}])
            body = _responses_body(model, items, arm, store=False)
            text_of, rebuild = _responses_text_of, _rebuild_responses
        else:
            url = f"{base_url()}/responses"
            # Only the new question. The system prompt and every prior turn already
            # live on the server; resending them is the thing this arm exists not to do.
            body = _responses_body(model, [{"role": "user", "content": question}],
                                   arm, store=True, conversation=conversation)
            text_of, rebuild = _responses_text_of, _rebuild_responses

        x = _send(url, body, measure=measure, text_of=text_of, rebuild=rebuild)
        chat = arm == "chat_stateless"
        usage = _usage(x.response, chat=chat)

        rec = turn_record(
            provider=NAME, arm=arm, phase="steady", turn=k, question=question,
            measure=measure, exchange=x, usage=usage,
            extra={"conversation": conversation, "url": url},
        )
        records.append(rec)

        if arm == "chat_stateless":
            # Chat has no reasoning items to echo: an assistant message is the
            # whole of what the server produced.
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": x.text})
        elif arm == "responses_stateless":
            # Rule 1: the model's turn goes back on the wire exactly as it came off
            # it. That is what makes this arm's upload the real cost of holding the
            # history client-side rather than a flattering reconstruction of it.
            history.append({"role": "user", "content": question})
            history.extend(_echo_items(x.response))
        # The stateful arm keeps no history: the server appended both messages for it.

    return records


def _create_conversation(system: str, measure: str) -> tuple[dict, str]:
    """The stateful arm's one-time upload of the system prompt.

    Counted and reported as a prep record rather than pretended free: this arm buys
    its flat per-turn upload with an upload here, and a comparison that hid it would
    be arguing with itself.

    Sent blocking whatever the run's `measure` is, and `measure` is not passed on. This
    is not a turn: nothing streams out of /v1/conversations, there is no first token to
    time, and the endpoint rejects the parameter outright --

        400 invalid_request_error: Unknown parameter: 'stream'.

    -- which is how every `latency` and `both` run of this arm died before its first
    question, with no conversation id to chain the turns onto. The run's `measure` is
    still what the record is filed under: it says which pass the *turns* were measured
    by, and a reader comparing prep bytes across runs has to see it.
    """
    url = f"{base_url()}/conversations"
    body = {"items": [{"type": "message", "role": "system", "content": system}]}
    x = _send(url, body, measure="bytes", text_of=lambda _e: "", rebuild=lambda _e: {})
    rec = turn_record(
        provider=NAME, arm="responses_stateful", phase="setup", turn=0,
        question="", measure=measure, exchange=x,
        usage={"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
               "reasoning_tokens": 0, "total_tokens": 0},
        extra={"conversation": x.response.get("id", ""), "url": url},
    )
    return rec, x.response.get("id", "")


def _send(url: str, body: dict, *, measure: str, text_of, rebuild) -> Exchange:
    if mock_mode():
        return _mock_send(url, body, measure)
    return send(url, _headers(), body, measure=measure, text_of=text_of,
                stream_body=_streamed(body), rebuild=rebuild, timeout=TIMEOUT)


# ------------------------------------------------------------------------- mock

_MOCK_REPLY = "ack"
# A fixed per-request framing cost. Fixed on purpose: the uplink delta between two
# arms then *is* the payload delta, with nothing of the mock's own invention in it.
_MOCK_REQ_OVERHEAD = 280
_MOCK_RESP_OVERHEAD = 210

# The server-side conversations, and what each of them holds. This is the whole
# reason mock mode cannot flatter the stateful arm: the client stops uploading the
# history, but the server keeps it, and keeps billing for it.
_MOCK_CONVERSATIONS: dict[str, int] = {}
_MOCK_SEQ = {"conv": 0, "resp": 0}


def reset_mock() -> None:
    _MOCK_CONVERSATIONS.clear()
    _MOCK_SEQ.update(conv=0, resp=0)


def _mock_chars(items) -> int:
    total = 0
    for it in items:
        c = it.get("content", it)
        total += len(c) if isinstance(c, str) else len(json.dumps(c))
    return total


def _mock_respond(url: str, body: dict) -> dict:
    """One synthetic reply, shaped like the real one, with usage that grows with
    whatever the payload actually carries."""
    if url.endswith("/conversations"):
        _MOCK_SEQ["conv"] += 1
        conv_id = f"conv_{_MOCK_SEQ['conv']}"
        _MOCK_CONVERSATIONS[conv_id] = _mock_chars(body.get("items", []))
        return {"id": conv_id, "object": "conversation"}

    _MOCK_SEQ["resp"] += 1
    uploaded = body.get("input") or body.get("messages") or []
    chars = _mock_chars(uploaded)

    conv_id = body.get("conversation")
    if conv_id:
        # The finding, in the mock: the client uploaded one question, and OpenAI
        # bills every prior input token in the chain anyway. input_tokens is
        # computed from the FULL server-side history, never from what came up the
        # wire — a mock that billed the upload would erase the result.
        _MOCK_CONVERSATIONS[conv_id] += chars
        chars = _MOCK_CONVERSATIONS[conv_id]
        _MOCK_CONVERSATIONS[conv_id] += len(_MOCK_REPLY)

    in_tokens = max(chars // 4, 1)
    reasoning = 24 if body.get("reasoning") or body.get("reasoning_effort") else 0

    if url.endswith("/chat/completions"):
        return {
            "id": f"chatcmpl_{_MOCK_SEQ['resp']}",
            "choices": [{"message": {"role": "assistant", "content": _MOCK_REPLY},
                         "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": in_tokens,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": reasoning},
            },
        }

    output: list[dict] = []
    if reasoning:
        # A reasoning item, opaque and expensive to carry — which is the point: an
        # arm holding the history client-side has to echo this back verbatim.
        output.append({"type": "reasoning", "id": f"rs_{_MOCK_SEQ['resp']}",
                       "summary": [], "encrypted_content": "x" * 256})
    output.append({"type": "message", "role": "assistant", "status": "completed",
                   "content": [{"type": "output_text", "text": _MOCK_REPLY}]})
    return {
        "id": f"resp_{_MOCK_SEQ['resp']}",
        "output": output,
        "usage": {
            "input_tokens": in_tokens,
            "output_tokens": 5,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": reasoning},
        },
    }


def _mock_send(url: str, body: dict, measure: str) -> Exchange:
    """A turn that never leaves the process, measured the way core.call measures one.

    `both` runs the request twice against the real API, and on the stateful arm both
    passes are appended to the server-side conversation. The mock appends twice too:
    the point of a mock here is to reproduce what the numbers will look like, and
    that includes the ways they can be wrong.
    """
    payload = json.dumps(body).encode()
    passes = 2 if measure == "both" else 1
    for _ in range(passes):
        data = _mock_respond(url, body)
    resp = json.dumps(data).encode()

    text = (_chat_text(data) if url.endswith("/chat/completions")
            else _responses_text(data))

    if measure == "bytes":
        marks = dict(req_sent_ms=0, ttfb_ms=0, ttft_ms=0, ttlt_ms=0, turn_end_ms=0)
    else:
        # Shaped like a real stream: the request goes out, the server thinks, tokens
        # arrive, and the connection stays open a moment past the last one.
        sent = 1 + len(payload) // 20_000
        ttfb = sent + 30
        ttft = ttfb + 20
        ttlt = ttft + 2 * len(text)
        marks = dict(req_sent_ms=sent, ttfb_ms=ttfb, ttft_ms=ttft, ttlt_ms=ttlt,
                     turn_end_ms=ttlt + 15)

    return Exchange(
        status=200,
        error="",
        wire_sent=len(payload) + _MOCK_REQ_OVERHEAD,
        wire_recv=len(resp) + _MOCK_RESP_OVERHEAD,
        req_payload_bytes=len(payload),
        resp_payload_bytes=len(resp),
        elapsed_ms=marks["turn_end_ms"],
        text=text,
        response=data,
        request_json=payload.decode(),
        response_json=resp.decode(),
        **marks,
    )
