"""One OpenAI call per arm, with real wire-byte counting.

Three arms, one model, one fixture:

  chat_stateless       POST /v1/chat/completions
                       messages = [system, u1, a1, ..., uk]        -> uploads O(N^2)

  responses_stateless  POST /v1/responses, store=false
                       input    = [system, u1, a1, ..., uk]        -> uploads O(N^2)
                       Control arm: same payload as chat_stateless on a different
                       endpoint, so any byte gap vs responses_stateful is about
                       *server-side state*, not about which endpoint we used.

  responses_stateful   POST /v1/conversations (once, seeded with the system prompt)
                       POST /v1/responses, conversation=conv_..., input = [uk]
                       -> uploads O(N). The server already holds the history.

The point of the experiment: on the stateful arm the bytes collapse while
`usage.input_tokens` does not. OpenAI's own docs say so — "all previous input
tokens for responses in the chain are billed as input tokens". Bytes can already
be saved; billing does not follow. That gap is the thing we are measuring.

Deliberate measurement choices:
  - non-streaming everywhere. Streaming SSE deltas carry `include_obfuscation`
    padding by default, which normalizes payload sizes and would corrupt a byte
    measurement.
  - the `requests` library, not the official SDK. The SDK rides on httpx, which
    our socket counter (an http.client/urllib3 subclass) cannot attach to.
  - the system prompt is byte-identical on every turn. OpenAI's prompt cache
    matches on an exact prefix; a timestamp in there would kill every cache hit.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field

import env  # noqa: F401  — loads .env before the os.environ reads below
from wire import session, wire_counter

ARMS = ("chat_stateless", "responses_stateless", "responses_stateful")

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-nano")
DEFAULT_MAX_OUTPUT_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "400"))
# Empty for a non-reasoning model: the parameter must not be sent at all, or the
# call 400s. Only a reasoning model that accepts "none" should set this.
DEFAULT_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "")
TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "180"))


def base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def api_host() -> str:
    """Hostname only — what the packet capture filters on."""
    from urllib.parse import urlparse
    return urlparse(base_url()).hostname or "api.openai.com"


def api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return key


def auth_headers() -> dict:
    return {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
    }


@dataclass
class CallResult:
    """One HTTP exchange, normalized across both endpoints."""

    arm: str
    turn: int
    url: str = ""
    text: str = ""

    # bytes actually crossing the socket: headers + content-encoded body
    wire_sent: int = 0
    wire_recv: int = 0
    # decoded JSON body sizes (application layer)
    req_payload_bytes: int = 0
    resp_payload_bytes: int = 0

    # usage, normalized: chat calls them prompt_/completion_, responses input_/output_
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    latency_ms: int = 0      # request sent -> response fully read
    # Only a stream has a first token to time. On a non-streamed call these stay
    # 0 rather than echoing latency_ms: a total is not a TTFT, and a copied number
    # is the kind of lie that ends up on a chart.
    ttft_ms: int = 0         # -> first CONTENT token (not the role/created event)
    ttlt_ms: int = 0         # -> last token (finish_reason / response.completed)
    streamed: bool = False
    response_id: str = ""
    error: str = ""

    # The JSON bodies exactly as sent and received. Bodies only — never headers,
    # which carry the bearer token. This is what makes the result auditable: the
    # numbers say the stateless arm re-sent 21 kB, and these say what was in it.
    request_json: str = ""
    response_json: str = ""

    @property
    def billed_uncached_tokens(self) -> int:
        return max(self.input_tokens - self.cached_tokens, 0)

    def as_dict(self, *, bodies: bool = False) -> dict:
        d = asdict(self)
        if not bodies:
            # metrics rows stay small; the bodies are written once, separately
            d.pop("request_json", None)
            d.pop("response_json", None)
        d["billed_uncached_tokens"] = self.billed_uncached_tokens
        return d


@dataclass
class _Exchange:
    """One HTTP round trip, measured."""
    data: dict
    wire_sent: int
    wire_recv: int
    req_bytes: int
    resp_bytes: int
    latency_ms: int
    request_json: str
    response_json: str
    ttft_ms: int = 0
    ttlt_ms: int = 0
    text: str = ""          # assembled from the deltas, when streamed
    streamed: bool = False


def _iter_sse(resp):
    """Yield (event_name, parsed_json) per SSE frame.

    Chat sends bare `data:` frames; Responses prefixes each with `event: <type>`
    and also repeats the type inside the JSON. Handle both.
    """
    event = ""
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            event = ""
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        yield event or obj.get("type", ""), obj


def _post_stream(url: str, body: dict) -> _Exchange:
    """Stream the response, timing the first and last token.

    The byte counting still works: wire_counter wraps the whole read, and the
    socket tally keeps ticking as the chunks arrive. What is NOT comparable to a
    non-streamed call is resp_bytes — SSE framing, the per-chunk envelope, and (on
    Responses) the full Response object shipped again in created/completed all
    ride along. Upload bytes are comparable; download bytes are not.
    """
    payload = json.dumps(body).encode()
    sess = session()
    is_chat = url.endswith("/chat/completions")

    t0 = time.perf_counter()
    ttft = ttlt = 0.0
    chunks: list[str] = []
    text_parts: list[str] = []
    usage: dict = {}
    resp_id = ""
    resp_bytes = 0

    with wire_counter() as w:
        resp = sess.post(url, data=payload, headers=auth_headers(),
                         timeout=TIMEOUT, stream=True)
        if resp.status_code >= 400:
            raise RuntimeError(f"{resp.status_code} from {url}: {resp.text[:500]}")

        for event, obj in _iter_sse(resp):
            chunks.append(json.dumps(obj))
            resp_bytes += len(json.dumps(obj))

            if is_chat:
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        # the first chunk carries role with content "" — not a token
                        if not ttft:
                            ttft = time.perf_counter()
                        text_parts.append(piece)
                    if choices[0].get("finish_reason"):
                        ttlt = time.perf_counter()
                # usage rides a trailing chunk with choices: [] — AFTER finish_reason
                if obj.get("usage"):
                    usage = obj["usage"]
                resp_id = resp_id or obj.get("id", "")
            else:
                if event == "response.output_text.delta":
                    if not ttft:
                        ttft = time.perf_counter()
                    text_parts.append(obj.get("delta", "") or "")
                elif event == "response.completed":
                    ttlt = time.perf_counter()
                    r = obj.get("response") or {}
                    usage = r.get("usage") or {}
                    resp_id = resp_id or r.get("id", "")
                elif event == "response.created":
                    resp_id = resp_id or ((obj.get("response") or {}).get("id", ""))

    end = time.perf_counter()
    ttlt = ttlt or end          # no terminal event seen; fall back to the read end

    ms = lambda t: int((t - t0) * 1000)  # noqa: E731
    return _Exchange(
        data={"id": resp_id, "usage": usage},
        wire_sent=w.sent,
        wire_recv=w.recv,
        req_bytes=len(payload),
        resp_bytes=resp_bytes,
        latency_ms=ms(end),
        request_json=payload.decode(),
        response_json="\n".join(chunks),
        ttft_ms=ms(ttft) if ttft else 0,
        ttlt_ms=ms(ttlt),
        text="".join(text_parts),
        streamed=True,
    )


def _post(url: str, body: dict) -> _Exchange:
    payload = json.dumps(body).encode()
    sess = session()
    t0 = time.perf_counter()
    with wire_counter() as w:
        resp = sess.post(
            url,
            data=payload,
            headers=auth_headers(),
            timeout=TIMEOUT,
        )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code} from {url}: {resp.text[:500]}")
    return _Exchange(
        data=resp.json(),
        wire_sent=w.sent,
        wire_recv=w.recv,
        req_bytes=len(payload),
        resp_bytes=len(resp.content),
        latency_ms=latency_ms,
        request_json=payload.decode(),
        response_json=resp.text,
    )


def create_conversation(system: str) -> tuple[str, CallResult]:
    """Seed a server-side conversation with the system prompt.

    This is the stateful arm's one-time upload of the system prompt. We count it
    and report it, rather than pretending the bytes were free.
    """
    url = f"{base_url()}/conversations"
    body = {"items": [{"type": "message", "role": "system", "content": system}]}
    x = _post(url, body)
    res = CallResult(
        arm="responses_stateful",
        turn=0,  # turn 0 = setup, not a model call
        url=url,
        wire_sent=x.wire_sent,
        wire_recv=x.wire_recv,
        req_payload_bytes=x.req_bytes,
        resp_payload_bytes=x.resp_bytes,
        latency_ms=x.latency_ms,
        response_id=x.data.get("id", ""),
        request_json=x.request_json,
        response_json=x.response_json,
    )
    return x.data["id"], res


# Obfuscation pads streaming deltas with random characters to normalize payload
# sizes (a side-channel mitigation). It is ON by default, and it would make every
# streamed byte we report meaningless. Off, always.
_CHAT_STREAM_OPTS = {"include_obfuscation": False, "include_usage": True}
# Responses always puts usage on response.completed, so there is no include_usage.
_RESP_STREAM_OPTS = {"include_obfuscation": False}


def _chat_body(model: str, system: str, history: list[dict],
               cache_key: str | None = None, stream: bool = False) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + history,
        "max_completion_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "stream": stream,
    }
    if stream:
        # without include_usage the chat endpoint sends no usage object at all in
        # a stream, and every token number would come back zero
        body["stream_options"] = dict(_CHAT_STREAM_OPTS)
    if DEFAULT_REASONING_EFFORT:
        body["reasoning_effort"] = DEFAULT_REASONING_EFFORT
    if cache_key:
        body["prompt_cache_key"] = cache_key
    return body


def _responses_body(model: str, items: list[dict], *, store: bool,
                    conversation: str | None = None,
                    cache_key: str | None = None, stream: bool = False) -> dict:
    body: dict = {
        "model": model,
        "input": items,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "stream": stream,
        "store": store,
    }
    if stream:
        body["stream_options"] = dict(_RESP_STREAM_OPTS)
    if conversation:
        body["conversation"] = conversation
    if DEFAULT_REASONING_EFFORT:
        body["reasoning"] = {"effort": DEFAULT_REASONING_EFFORT}
    if cache_key:
        body["prompt_cache_key"] = cache_key
    return body


def _normalize_usage(u: dict, *, chat: bool) -> dict:
    """The two endpoints name the same four numbers differently. One shape out, so
    every downstream metric can stop caring which endpoint produced it."""
    if chat:
        return {
            "input_tokens": u.get("prompt_tokens", 0),
            "output_tokens": u.get("completion_tokens", 0),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
        }
    return {
        "input_tokens": u.get("input_tokens", 0),
        "output_tokens": u.get("output_tokens", 0),
        "cached_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens", 0),
        "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0),
    }


def _parse_chat(data: dict) -> tuple[str, dict]:
    text = data["choices"][0]["message"].get("content") or ""
    return text, _normalize_usage(data.get("usage") or {}, chat=True)


def _parse_responses(data: dict) -> tuple[str, dict]:
    text = ""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text += part.get("text", "")
    return text, _normalize_usage(data.get("usage") or {}, chat=False)


def call(arm: str, *, model: str, system: str, history: list[dict], question: str,
         turn: int, conversation: str | None = None,
         cache_key: str | None = None, stream: bool = False) -> CallResult:
    """Run one turn on one arm.

    `history` is the conversation so far as [{role, content}, ...], NOT including
    the system prompt and NOT including the current question. The stateless arms
    resend all of it; the stateful arm ignores it and sends only `question`,
    because the server is already holding the same thing.

    `cache_key` pins the prompt-cache routing. Without it, whether a call hits the
    cache depends on which node it lands on — measured live, an identical prefix
    hit 2/5 times unkeyed and 4/5 keyed. Unkeyed, cached_tokens is noise and so is
    every cost figure computed from it.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")

    user_msg = {"role": "user", "content": question}

    if arm == "chat_stateless":
        url = f"{base_url()}/chat/completions"
        body = _chat_body(model, system, history + [user_msg], cache_key=cache_key,
                          stream=stream)
        parse = _parse_chat
    elif arm == "responses_stateless":
        url = f"{base_url()}/responses"
        items = [{"role": "system", "content": system}] + history + [user_msg]
        body = _responses_body(model, items, store=False, cache_key=cache_key,
                               stream=stream)
        parse = _parse_responses
    else:  # responses_stateful
        if not conversation:
            raise ValueError("responses_stateful needs a conversation id")
        url = f"{base_url()}/responses"
        # only the new question. system + prior turns already live on the server.
        body = _responses_body(model, [user_msg], store=True,
                               conversation=conversation, cache_key=cache_key,
                               stream=stream)
        parse = _parse_responses

    if stream:
        x = _post_stream(url, body)
        # a stream hands us the text in pieces and the usage in a trailing frame;
        # both endpoints' usage objects still normalize through the same parsers
        text = x.text
        usage = _normalize_usage(x.data.get("usage") or {}, chat=(arm == "chat_stateless"))
    else:
        x = _post(url, body)
        text, usage = parse(x.data)

    return CallResult(
        arm=arm,
        turn=turn,
        url=url,
        text=text,
        wire_sent=x.wire_sent,
        wire_recv=x.wire_recv,
        req_payload_bytes=x.req_bytes,
        resp_payload_bytes=x.resp_bytes,
        latency_ms=x.latency_ms,
        ttft_ms=x.ttft_ms,
        ttlt_ms=x.ttlt_ms,
        streamed=x.streamed,
        response_id=x.data.get("id", ""),
        request_json=x.request_json,
        response_json=x.response_json,
        **usage,
    )
