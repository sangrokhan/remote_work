"""OpenAI: four ways to keep one conversation going, and the gap between bytes and
billing.

Ported from ``token_traffic/providers/openai.py`` (DESIGN.md 5, A2) onto the
``aipt.backends.base.Backend`` protocol.

  chat_stateless        POST /v1/chat/completions
                        messages = [system, u1, a1, ..., uk]   -> uploads O(N^2)

  responses_stateless   POST /v1/responses, store=false
                        input    = [system, u1, a1, ..., uk]   -> uploads O(N^2)

  responses             POST /v1/responses, store=true, previous_response_id=...
                        instructions = system   (every turn)
                        input        = [uk]
                        -> uploads O(1) per turn, but a *big* O(1).

  responses_inline      POST /v1/conversations, no items       (~200 B)
                        POST /v1/responses, conversation=conv_..., store=true
                        turn 1: input = [system, u1]           <- the prompt is
                        turn k: input = [uk]                      stored here

These map onto the Gemini arms one for one: `responses` <-> `interaction`,
`responses_inline` <-> `interaction_inline`, `responses_stateless` <->
`interaction_stateless`.

What the experiment is about: on every server-state arm the uploaded bytes collapse
while `usage.input_tokens` does not.

Measurement choices that are not incidental:

  - the `requests` library, never the official SDK.
  - the system prompt is byte-identical on every turn *of an arm*; cachebust puts a
    per-(run, arm) marker in front of it between arms.
  - `prompt_cache_key` is pinned per arm and per run.

``connect``/``send_turn``/``close`` adaptation notes (what changed from
``run_arm``, and why):

  * ``chat_stateless``, ``responses_stateless``, ``responses`` map onto the
    lifecycle directly -- each was already a per-turn loop carrying ``history`` or
    ``previous`` forward.
  * ``responses_inline`` opens with a conversation create. In ``run_arm`` this
    happened before the turn loop and produced a ``phase="setup"`` record inline
    with the steady ones; here it happens in ``connect`` and the setup record is
    stashed on the backend instance for the client to collect (see
    ``GeminiBackend``... err, ``OpenAIBackend.pending_setup_records``) since
    ``connect`` has no return channel in the Backend protocol.
"""

from __future__ import annotations

import json
import os

from aipt.backends import base
from aipt.backends.public_ai import _cachebust as cachebust
from aipt.backends.public_ai._call import Exchange, send
from aipt.core import config

NAME = "public_ai"
PROVIDER = "openai"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-nano")

ARMS = ("chat_stateless", "responses_stateless", "responses", "responses_inline")
HEADLINE_ARMS = ARMS
PROMPT_SENT_ONCE_ARMS = ("responses_inline",)

MAX_OUTPUT_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "400"))
REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "")
TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "180"))


def base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def api_host() -> str:
    """The host a capture filters on. Derived from base_url so that pointing the
    backend at a proxy or a local fake also points the pcap at it."""
    from urllib.parse import urlparse
    return urlparse(base_url()).hostname or "api.openai.com"


def mock_mode() -> bool:
    """TRAFFIC_MOCK is the suite-wide switch; OPENAI_MOCK turns this one engine
    into a mock while gemini still talks to its API. The parse is core's, and
    shared."""
    return config.is_mock(PROVIDER)


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
    """Distinct per arm, and per run when cache-busting is on."""
    tag = cachebust.tag(PROVIDER, arm)
    return f"tt-{tag}-{arm}" if tag else f"tt-{PROVIDER}-{arm}"


# ---------------------------------------------------------------- request bodies

_CHAT_STREAM_OPTS = {"include_obfuscation": False, "include_usage": True}
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
        body["stream_options"] = dict(_CHAT_STREAM_OPTS)
    if REASONING_EFFORT:
        body["reasoning_effort"] = REASONING_EFFORT
    return body


def _responses_body(model: str, items: list[dict], arm: str, *, store: bool,
                     conversation: str | None = None, previous: str | None = None,
                     instructions: str = "", stream: bool = False) -> dict:
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
    choices = event.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("delta") or {}).get("content") or ""


def _responses_text_of(event: dict) -> str:
    if event.get("type") == "response.output_text.delta":
        return event.get("delta") or ""
    return ""


def _rebuild_chat(events: list) -> dict:
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
    """What an arm that keeps the history client-side must put back on the wire."""
    return list(data.get("output") or [])


# --- record helper -----------------------------------------------------------

def _rec(arm, phase, turn, question, measure, exchange, usage, extra=None) -> dict:
    """One row, backend-tagged ``public_ai`` with ``engine: openai`` in ``extra``."""
    from aipt.backends.record import turn_record
    merged_extra = {"engine": PROVIDER}
    if extra:
        merged_extra.update(extra)
    return turn_record(NAME, arm, phase, turn, question, measure, exchange, usage,
                        extra=merged_extra)


# ------------------------------------------------------------------------- arms

def run_arm(arm, model, system, steps, measure, on_progress=None) -> list[dict]:
    """Replay the conversation on one arm.

    Kept for parity testing against the original ``token_traffic`` behaviour.
    New client code should prefer :class:`OpenAIBackend`'s connect/send_turn/close
    lifecycle (DESIGN.md 4.5).
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
        rec, conversation = _create_conversation(arm)
        records.append(rec)

    for k, question in enumerate(steps, start=1):
        base.progress(on_progress, NAME, arm, "steady", k, n)
        url = f"{base_url()}/responses"

        if arm == "chat_stateless":
            url = f"{base_url()}/chat/completions"
            body = _chat_body(model, cachebust.per_turn(system, k), history, question,
                               arm)
        elif arm == "responses_stateless":
            items = ([{"role": "system", "content": cachebust.per_turn(system, k)}]
                     + history
                     + [{"role": "user", "content": question}])
            body = _responses_body(model, items, arm, store=False)
        elif arm == "responses":
            body = _responses_body(model, [{"role": "user", "content": question}],
                                    arm, store=True, previous=previous or None,
                                    instructions=cachebust.per_turn(system, k))
        else:   # responses_inline
            items = ([{"role": "system", "content": cachebust.per_turn(system, k)}]
                     if k == 1 else [])
            items += [{"role": "user", "content": question}]
            body = _responses_body(model, items, arm, store=True,
                                    conversation=conversation)

        chat = arm == "chat_stateless"
        text_of = _chat_text_of if chat else _responses_text_of
        rebuild = _rebuild_chat if chat else _rebuild_responses

        x = _send(url, body, measure=measure, text_of=text_of, rebuild=rebuild)
        usage = _usage(x.response, chat=chat)

        rec = _rec(arm, "steady", k, question, measure, x, usage,
                   {"conversation": conversation, "previous_response_id": previous,
                    "url": url})
        records.append(rec)

        if arm == "chat_stateless":
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": x.text})
        elif arm == "responses_stateless":
            history.append({"role": "user", "content": question})
            history.extend(_echo_items(x.response))
        elif arm == "responses":
            previous = (x.response or {}).get("id") or previous

    return records


def _create_conversation(arm: str) -> tuple[dict, str]:
    """Open an empty server-side conversation and return (setup_record, conversation_id)."""
    url = f"{base_url()}/conversations"
    body: dict = {}
    x = _send(url, body, measure="bytes", text_of=lambda _e: "", rebuild=lambda _e: {})
    rec = _rec(arm, "setup", 0, "", "bytes", x,
               {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
                "reasoning_tokens": 0, "total_tokens": 0},
               {"kind": "conversation_create", "billed": False,
                "conversation": x.response.get("id", ""), "url": url})
    return rec, x.response.get("id", "")


def _send(url: str, body: dict, *, measure: str, text_of, rebuild) -> Exchange:
    if mock_mode():
        return _mock_send(url, body, measure)
    return send(url, _headers(), body, measure=measure, text_of=text_of,
                stream_body=_streamed(body), rebuild=rebuild, timeout=TIMEOUT)


# ------------------------------------------------------------------------- mock

_MOCK_REPLY = "ack"
_MOCK_REQ_OVERHEAD = 280
_MOCK_RESP_OVERHEAD = 210

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
    if url.endswith("/conversations"):
        _MOCK_SEQ["conv"] += 1
        conv_id = f"conv_{_MOCK_SEQ['conv']}"
        _MOCK_CONVERSATIONS[conv_id] = _mock_chars(body.get("items", []))
        return {"id": conv_id, "object": "conversation"}

    _MOCK_SEQ["resp"] += 1
    resp_id = f"resp_{_MOCK_SEQ['resp']}"
    uploaded = body.get("input") or body.get("messages") or []
    instructions = body.get("instructions") or ""
    chars = _mock_chars(uploaded) + len(instructions)

    conv_id = body.get("conversation")
    if conv_id:
        _MOCK_CONVERSATIONS[conv_id] += _mock_chars(uploaded)
        chars = _MOCK_CONVERSATIONS[conv_id] + len(instructions)
        _MOCK_CONVERSATIONS[conv_id] += len(_MOCK_REPLY)
    elif body.get("store"):
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


# --- Backend protocol: connect / send_turn / close --------------------------

class OpenAIBackend:
    """``aipt.backends.base.Backend`` over the OpenAI API.

    ``responses_inline`` needs a conversation-create call before any turn; since
    ``connect`` has no return channel, its setup record is stashed on
    ``pending_setup_records`` for the client to drain after ``connect()`` (mirrors
    how ``run_arm`` prepended it to the record list).
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
        self._history: list = []
        self._conversation: str = ""
        self._previous: str = ""
        #: Non-turn records produced outside send_turn (e.g. the responses_inline
        #: conversation create). The client should drain this after connect() and
        #: append to its own record stream; it is not part of the TurnExchange
        #: protocol, which is one exchange per send_turn call.
        self.pending_setup_records: list[dict] = []

    def ready(self) -> tuple[bool, str]:
        return ready()

    def api_host(self) -> str:
        return api_host()

    def connect(self, arm: str, model: str, system: str) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        self._arm = arm
        self._model = model or DEFAULT_MODEL
        self._system = system or ""
        self._history = []
        self._conversation = ""
        self._previous = ""
        self.pending_setup_records = []
        if arm == "responses_inline":
            rec, conversation = _create_conversation(arm)
            self._conversation = conversation
            self.pending_setup_records.append(rec)

    def send_turn(self, turn: int, question: str, measure: str, on_progress=None):
        if self._arm is None:
            raise RuntimeError("send_turn called before connect")
        base.progress(on_progress, NAME, self._arm, "steady", turn, turn)
        url = f"{base_url()}/responses"
        arm = self._arm

        if arm == "chat_stateless":
            url = f"{base_url()}/chat/completions"
            body = _chat_body(self._model, cachebust.per_turn(self._system, turn),
                               self._history, question, arm)
        elif arm == "responses_stateless":
            items = ([{"role": "system", "content": cachebust.per_turn(self._system, turn)}]
                     + self._history
                     + [{"role": "user", "content": question}])
            body = _responses_body(self._model, items, arm, store=False)
        elif arm == "responses":
            body = _responses_body(
                self._model, [{"role": "user", "content": question}], arm,
                store=True, previous=self._previous or None,
                instructions=cachebust.per_turn(self._system, turn))
        else:  # responses_inline
            items = ([{"role": "system", "content": cachebust.per_turn(self._system, turn)}]
                     if turn == 1 else [])
            items += [{"role": "user", "content": question}]
            body = _responses_body(self._model, items, arm, store=True,
                                    conversation=self._conversation)

        chat = arm == "chat_stateless"
        text_of = _chat_text_of if chat else _responses_text_of
        rebuild = _rebuild_chat if chat else _rebuild_responses
        x = _send(url, body, measure=measure, text_of=text_of, rebuild=rebuild)

        if arm == "chat_stateless":
            self._history.append({"role": "user", "content": question})
            self._history.append({"role": "assistant", "content": x.text})
        elif arm == "responses_stateless":
            self._history.append({"role": "user", "content": question})
            self._history.extend(_echo_items(x.response))
        elif arm == "responses":
            self._previous = (x.response or {}).get("id") or self._previous

        return x

    def close(self) -> None:
        self._arm = None


#: Module-level singleton -- mirrors GeminiBackend's.
BACKEND = OpenAIBackend()
