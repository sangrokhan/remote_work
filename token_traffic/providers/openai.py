"""OpenAI: four ways to keep one conversation going, and the gap between bytes and
billing.

  chat_stateless        POST /v1/chat/completions
                        messages = [system, u1, a1, ..., uk]   -> uploads O(N^2)

  responses_stateless   POST /v1/responses, store=false
                        input    = [system, u1, a1, ..., uk]   -> uploads O(N^2)
                        The control arm: the same payload as chat_stateless on a
                        different endpoint, so any byte gap against the server-state
                        arms is about server-side state and not about which
                        endpoint the bytes went through.

  responses             POST /v1/responses, store=true, previous_response_id=...
                        instructions = system   (every turn)
                        input        = [uk]
                        -> uploads O(1) per turn, but a *big* O(1): the history
                        lives on the server and the system prompt does not.
                        `instructions` is top-level and is not stored, so it must
                        be resent with every request (OpenAI, migrate-to-responses).
                        The idiomatic Responses loop -- previous_response_id is the
                        continuation OpenAI's own docs reach for first.

  responses_inline      POST /v1/conversations, no items       (~200 B)
                        POST /v1/responses, conversation=conv_..., store=true
                        turn 1: input = [system, u1]           <- the prompt is
                        turn k: input = [uk]                      stored here
                        -> uploads O(1), and a small one. Same content reaches the
                        model as `responses`; a different party stores the prompt,
                        so the gap between the two arms is the system prompt itself.

These map onto the Gemini arms one for one, which is the point -- `responses` is
`interaction` (server holds the history, the system prompt goes up every turn),
`responses_inline` is `interaction_inline` (the system prompt is stored with the
history), `responses_stateless` is `interaction_stateless`. A finding that holds on one
vendor and not the other is only visible if the arms line up.

What the experiment is about: on every server-state arm the uploaded bytes collapse while
`usage.input_tokens` does not. OpenAI's own documentation says why — all previous input
tokens for responses in the chain are billed as input tokens. Measured: the inline arm
uploaded 1176 B on turn 1 against chat_stateless's 21866 B, and both were billed 4338
input tokens. Exactly the same. There are ways to stop uploading the history and none of
them stops paying for it. That gap is the finding, and a mock run must not be able to
pretend it away.

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
    stopped one arm from reading another's cache — `responses_inline` was billed
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

ARMS = ("chat_stateless", "responses_stateless", "responses", "responses_inline")
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
                    conversation: str | None = None, previous: str | None = None,
                    instructions: str = "", stream: bool = False) -> dict:
    """One /v1/responses request.

    `instructions` is the top-level system prompt, and it is a different thing from a
    system message in `input`: an input item is appended to the server-side history and
    is therefore stored once, while `instructions` is not stored at all and has to be
    resent with every request (OpenAI, migrate-to-responses). That distinction is the
    whole difference between `responses` and `responses_inline`, so the two must not be
    reachable through the same field -- a system message sent as `input` on every turn of
    the `responses` arm would be appended on every turn, and turn k's input_tokens would
    count k copies of the system prompt.
    """
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
    if instructions:
        body["instructions"] = instructions
    if conversation:
        body["conversation"] = conversation
    if previous:
        body["previous_response_id"] = previous
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

    `responses_inline` opens with a conversation create -- a prep record with phase
    "setup": its ~200 bytes are counted and reported, not hidden, but core.metrics keeps
    prep out of the totals because it is setup, not traffic. The system prompt itself does
    not go up here; it rides turn 1, inside the measured window, where a reader can see it.

    One caveat `responses_inline` cannot design away: `measure="both"` sends the turn
    twice, and both passes carry `conversation=`, which OpenAI appends to (store=false is
    not allowed alongside a conversation). So `both` writes each turn into the server-side
    history twice and inflates input_tokens from the next turn on. core.runner refuses it.
    `responses` is safe with `both`: each pass branches from the same previous_response_id
    rather than appending to a shared object, and the chain follows the blocking pass --
    the streamed pass is an orphan branch that costs money and corrupts nothing.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")

    records: list[dict] = []
    history: list[dict] = []
    conversation = ""
    previous = ""

    n = len(steps)
    if arm == "responses_inline":
        base.progress(on_progress, NAME, arm, "setup", 1, n)
        # No items. `items` is optional on POST /v1/conversations, so the create is a bare
        # container and the system prompt is uploaded once, inside turn 1, where the
        # measurement can see it.
        rec, conversation = _create_conversation(arm)
        records.append(rec)

    for k, question in enumerate(steps, start=1):
        base.progress(on_progress, NAME, arm, "steady", k, n)
        url = f"{base_url()}/responses"

        if arm == "chat_stateless":
            url = f"{base_url()}/chat/completions"
            body = _chat_body(model, system, history, question, arm)
        elif arm == "responses_stateless":
            items = ([{"role": "system", "content": system}]
                     + history
                     + [{"role": "user", "content": question}])
            body = _responses_body(model, items, arm, store=False)
        elif arm == "responses":
            # The server holds the history; it does not hold the system prompt.
            # `instructions` is not stored, so it goes up again on every single turn --
            # the same bill Gemini's `interaction` arm pays for the same reason.
            body = _responses_body(model, [{"role": "user", "content": question}],
                                   arm, store=True, previous=previous or None,
                                   instructions=system)
        else:   # responses_inline
            # Turn 1 carries the system prompt as an input *item*, so the server stores
            # it with the history and no later turn resends it. Not `instructions`:
            # that would not be stored, and this arm would silently become `responses`.
            items = ([{"role": "system", "content": system}] if k == 1 else [])
            items += [{"role": "user", "content": question}]
            body = _responses_body(model, items, arm, store=True,
                                   conversation=conversation)

        chat = arm == "chat_stateless"
        text_of = _chat_text_of if chat else _responses_text_of
        rebuild = _rebuild_chat if chat else _rebuild_responses

        x = _send(url, body, measure=measure, text_of=text_of, rebuild=rebuild)
        usage = _usage(x.response, chat=chat)

        rec = turn_record(
            provider=NAME, arm=arm, phase="steady", turn=k, question=question,
            measure=measure, exchange=x, usage=usage,
            extra={"conversation": conversation, "previous_response_id": previous,
                   "url": url},
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
        elif arm == "responses":
            previous = (x.response or {}).get("id") or previous
        # responses_inline keeps no history: the server appended both messages.

    return records


def _create_conversation(arm: str) -> tuple[dict, str]:
    """Open an empty server-side conversation and return (setup_record, conversation_id).

    `items` is optional on POST /v1/conversations, so this bare create is a legal
    ~200-byte call. The system prompt does not go up here: `responses_inline` lets it ride
    turn 1, inside the measured window, where a reader can see what it costs.

    Nothing is billed here, and the zeros in this record are measured, not assumed: the
    endpoint runs no inference and its response carries no usage object at all --

        {"id": "conv_...", "object": "conversation", "created_at": ..., "metadata": {}}

    -- so 0 tokens is the honest number and `kind` is what says it means "not billed"
    rather than "not sent".

    Sent blocking whatever the run's `measure` is. This is not a turn: nothing streams
    out of /v1/conversations, there is no first token to time, and the endpoint rejects
    the parameter outright --

        400 invalid_request_error: Unknown parameter: 'stream'.

    -- which is how every `latency` and `both` run of this arm died before its first
    question, with no conversation id to chain the turns onto.
    """
    url = f"{base_url()}/conversations"
    body: dict = {}
    x = _send(url, body, measure="bytes", text_of=lambda _e: "", rebuild=lambda _e: {})
    rec = turn_record(
        provider=NAME, arm=arm, phase="setup", turn=0,
        question="", measure="bytes", exchange=x,
        usage={"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
               "reasoning_tokens": 0, "total_tokens": 0},
        extra={"kind": "conversation_create",
               "billed": False,
               "conversation": x.response.get("id", ""), "url": url},
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

# The server-side state, and what each piece of it holds. This is the whole reason mock
# mode cannot flatter the server-state arms: the client stops uploading the history, but
# the server keeps it, and keeps billing for it.
#   _MOCK_CONVERSATIONS  conversation id -> chars stored in it
#   _MOCK_CHAINS         response id     -> chars of history behind it
# Two structures because they are two mechanisms: a conversation is a shared container
# that every turn appends to, a chain is a linked list that every turn branches from.
_MOCK_CONVERSATIONS: dict[str, int] = {}
_MOCK_CHAINS: dict[str, int] = {}
_MOCK_SEQ = {"conv": 0, "resp": 0}


def reset_mock() -> None:
    _MOCK_CONVERSATIONS.clear()
    _MOCK_CHAINS.clear()
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
    resp_id = f"resp_{_MOCK_SEQ['resp']}"
    uploaded = body.get("input") or body.get("messages") or []
    # `instructions` is billed on the turn that sends it and is never stored: it rides
    # up again next turn because the server did not keep it. Adding it to the stored
    # history instead would make the chained arm's input_tokens grow by a system prompt
    # per turn -- a curve the real API does not produce.
    instructions = body.get("instructions") or ""
    chars = _mock_chars(uploaded) + len(instructions)

    conv_id = body.get("conversation")
    if conv_id:
        # The finding, in the mock: the client uploaded one question, and OpenAI
        # bills every prior input token in the chain anyway. input_tokens is
        # computed from the FULL server-side history, never from what came up the
        # wire — a mock that billed the upload would erase the result.
        _MOCK_CONVERSATIONS[conv_id] += _mock_chars(uploaded)
        chars = _MOCK_CONVERSATIONS[conv_id] + len(instructions)
        _MOCK_CONVERSATIONS[conv_id] += len(_MOCK_REPLY)
    elif body.get("store"):
        # The chain. Turn k branches off turn k-1's stored response, so what it is
        # billed for is everything behind that response plus what it uploaded now --
        # the same total the conversation arms pay, reached down a different road.
        behind = _MOCK_CHAINS.get(body.get("previous_response_id") or "", 0)
        stored = behind + _mock_chars(uploaded)
        chars = stored + len(instructions)
        _MOCK_CHAINS[resp_id] = stored + len(_MOCK_REPLY)

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
        "id": resp_id,
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

    `both` runs the request twice against the real API, and on `responses_inline` both
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
