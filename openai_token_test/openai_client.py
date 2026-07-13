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

    latency_ms: int = 0
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


def _chat_body(model: str, system: str, history: list[dict],
               cache_key: str | None = None) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + history,
        "max_completion_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "stream": False,
    }
    if DEFAULT_REASONING_EFFORT:
        body["reasoning_effort"] = DEFAULT_REASONING_EFFORT
    if cache_key:
        body["prompt_cache_key"] = cache_key
    return body


def _responses_body(model: str, items: list[dict], *, store: bool,
                    conversation: str | None = None,
                    cache_key: str | None = None) -> dict:
    body: dict = {
        "model": model,
        "input": items,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "stream": False,
        "store": store,
    }
    if conversation:
        body["conversation"] = conversation
    if DEFAULT_REASONING_EFFORT:
        body["reasoning"] = {"effort": DEFAULT_REASONING_EFFORT}
    if cache_key:
        body["prompt_cache_key"] = cache_key
    return body


def _parse_chat(data: dict) -> tuple[str, dict]:
    text = data["choices"][0]["message"].get("content") or ""
    u = data.get("usage") or {}
    usage = {
        "input_tokens": u.get("prompt_tokens", 0),
        "output_tokens": u.get("completion_tokens", 0),
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
    }
    return text, usage


def _parse_responses(data: dict) -> tuple[str, dict]:
    text = ""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text += part.get("text", "")
    u = data.get("usage") or {}
    usage = {
        "input_tokens": u.get("input_tokens", 0),
        "output_tokens": u.get("output_tokens", 0),
        "cached_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens", 0),
        "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0),
    }
    return text, usage


def call(arm: str, *, model: str, system: str, history: list[dict], question: str,
         turn: int, conversation: str | None = None,
         cache_key: str | None = None) -> CallResult:
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
        body = _chat_body(model, system, history + [user_msg], cache_key=cache_key)
        parse = _parse_chat
    elif arm == "responses_stateless":
        url = f"{base_url()}/responses"
        items = [{"role": "system", "content": system}] + history + [user_msg]
        body = _responses_body(model, items, store=False, cache_key=cache_key)
        parse = _parse_responses
    else:  # responses_stateful
        if not conversation:
            raise ValueError("responses_stateful needs a conversation id")
        url = f"{base_url()}/responses"
        # only the new question. system + prior turns already live on the server.
        body = _responses_body(model, [user_msg], store=True,
                               conversation=conversation, cache_key=cache_key)
        parse = _parse_responses

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
        response_id=x.data.get("id", ""),
        request_json=x.request_json,
        response_json=x.response_json,
        **usage,
    )
