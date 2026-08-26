"""aipt.backends.local_llm.engine_adapter -- thin client for a standard
serving engine's OpenAI-compatible HTTP API (DESIGN.md 4.5, "확정된 설계
결정" row 1, and B4).

DESIGN.md's decision is explicit: **do not reimplement inference**. llama.cpp
(via its `llama-server` OpenAI-compatible HTTP server, or the
`llama-cpp-python` package's own OpenAI-compatible server mode) and vLLM
(via its OpenAI-compatible API server) both already speak the same
`POST /v1/chat/completions` wire format that ``aipt.backends.public_ai``
already knows how to time and byte-count (``aipt.core.wire``,
``aipt.core.streaming``). So this module is not an engine -- it never spawns
a server process and never loads a model -- it is a client that knows that
one wire format and nothing else about which engine answers it.

Actually starting an engine process (downloading a GGUF, `llama-server
--model ...`, or `vllm serve ...`) is explicitly out of scope for this
change (see the task's own framing and DESIGN.md B4's "엔진 선택은
huggingface-hub/llama-cpp/serving-llms-vllm 스킬 활용" -- that is a
follow-up operational concern, not something this adapter does). Instead the
engine's base URL is supplied externally, by convention
``LOCAL_LLM_ENGINE_URL`` (default ``http://127.0.0.1:8080``, the default
`llama-server` port) -- whoever stood the engine up (a human, a Compose
service, a later Dockerfile.local-llm) is responsible for it being reachable
there. ``LOCAL_LLM_ENGINE_KIND`` is purely a label (``"llama_cpp"`` /
``"vllm"``) carried through to the run record for readers -- both kinds are
driven through the identical OpenAI-compatible request/response shape, so
the adapter's own behaviour never branches on it.

``EngineAdapter.text_of``/``usage_of`` intentionally mirror the shape of the
callback contract ``aipt.backends.public_ai._call.send`` already takes
(``text_of(event) -> str``) so a caller could, in principle, hand this
adapter's callables to that same call-layer -- OpenAI-compatible chat
completions and OpenAI's own Responses API differ, but a blocking chat
completion and a streamed chat completion chunk share one shape
(``choices[0].delta.content`` while streaming, ``choices[0].message.content``
once complete), which is why one ``text_of`` handles both.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

#: Purely a label carried into run records -- see module docstring. The
#: adapter's request/response handling never branches on this value because
#: both engines are addressed through the identical OpenAI-compatible API.
EngineKind = Literal["llama_cpp", "vllm"]

_KNOWN_KINDS = ("llama_cpp", "vllm")

#: The default `llama-server` (llama.cpp's OpenAI-compatible HTTP server)
#: listen address. vLLM's OpenAI server defaults to :8000 -- a vLLM-backed
#: run is expected to set LOCAL_LLM_ENGINE_URL explicitly rather than rely
#: on this default silently being wrong for it.
DEFAULT_ENGINE_URL = "http://127.0.0.1:8080"
DEFAULT_MODEL = "local-model"


def engine_url() -> str:
    """The configured engine base URL (no trailing slash)."""
    return (os.environ.get("LOCAL_LLM_ENGINE_URL") or DEFAULT_ENGINE_URL).rstrip("/")


def engine_kind() -> EngineKind:
    kind = (os.environ.get("LOCAL_LLM_ENGINE_KIND") or "llama_cpp").strip().lower()
    return kind if kind in _KNOWN_KINDS else "llama_cpp"


def api_key() -> str:
    """Most local engines need no key; some deployments (an engine behind
    its own auth, or a shared dev box) still want one -- optional, unlike
    public_ai's ``GEMINI_API_KEY``/``OPENAI_API_KEY`` which are required."""
    return os.environ.get("LOCAL_LLM_API_KEY", "")


def default_model() -> str:
    return os.environ.get("LOCAL_LLM_MODEL", DEFAULT_MODEL)


def ready() -> tuple[bool, str]:
    """Whether this adapter is configured to try talking to an engine.

    Unlike public_ai's ``ready()`` (which needs a real API key), a local
    engine has a workable default (``DEFAULT_ENGINE_URL``) -- there is
    nothing to require up front. This deliberately does NOT probe the URL:
    a run against an engine that is not actually up yet must fail loudly at
    ``send_turn`` (with the actual connection error attached to that turn's
    record), not silently report "not ready" before a run even starts,
    which is a check no other backend in this codebase performs either
    (public_ai's ``ready()`` only checks a key is present, never that the
    provider is reachable).
    """
    url = engine_url()
    if not url:
        return False, "LOCAL_LLM_ENGINE_URL is empty"
    return True, f"targeting {url} ({engine_kind()})"


@dataclass
class EngineAdapter:
    """Builds requests for, and reads responses from, one OpenAI-compatible
    chat/completions endpoint. Holds no connection state of its own --
    ``aipt.core.wire``'s counting session is what actually opens sockets;
    this only knows how to shape a request and parse a reply.
    """

    base_url: str = ""
    #: "" (falsy) means "not given -- resolve engine_kind() at
    #: __post_init__ time", matching base_url/model/api_key_value below.
    #: Kept as EngineKind rather than str so a caller who *does* pass one
    #: still gets the closed-Literal type check; only the class-level
    #: default itself has to widen to satisfy that sentinel.
    kind: EngineKind = ""  # type: ignore[assignment]
    model: str = ""
    api_key_value: str = ""
    timeout: int = 120
    #: Extra body fields merged into every request (e.g. engine-specific
    #: sampling knobs). A plain dict, not validated here -- this adapter is
    #: a thin client, not a schema for every engine's sampling parameters.
    extra_body: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_url = (self.base_url or engine_url()).rstrip("/")
        self.kind = self.kind or engine_kind()
        self.model = self.model or default_model()
        self.api_key_value = self.api_key_value or api_key()

    def chat_completions_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key_value:
            headers["Authorization"] = f"Bearer {self.api_key_value}"
        return headers

    def build_body(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: dict | None = None,
    ) -> dict:
        """The OpenAI-compatible chat/completions request body.

        Both llama.cpp's `llama-server` and vLLM's OpenAI server accept
        this exact shape -- ``model``/``messages``/``stream`` plus the
        common sampling knobs -- so nothing here is engine-specific;
        ``extra``/``self.extra_body`` exist for anything that is.
        """
        body: dict = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if self.extra_body:
            body.update(self.extra_body)
        if extra:
            body.update(extra)
        return body

    @staticmethod
    def text_of(event: dict) -> str:
        """The answer text in one OpenAI-compatible chunk or completed
        response -- handles both shapes with one function, matching the
        ``text_of(event) -> str`` contract ``aipt.backends.public_ai._call``
        already uses (streamed chunks carry ``delta``, a blocking/completed
        response carries ``message``).
        """
        if not isinstance(event, dict):
            return ""
        choices = event.get("choices") or []
        if not choices:
            return ""
        choice = choices[0] or {}
        delta = choice.get("delta")
        if isinstance(delta, dict) and delta.get("content"):
            return delta.get("content") or ""
        message = choice.get("message")
        if isinstance(message, dict):
            return message.get("content") or ""
        # Some engines put the fully-formed text directly on the choice
        # (legacy /v1/completions-style `text` field) -- accept it too
        # rather than reporting an answer-less turn.
        return choice.get("text") or ""

    @staticmethod
    def usage_of(body: dict) -> dict:
        """Backend-neutral usage dict from an OpenAI-compatible response's
        ``usage`` block, in the shape ``aipt.backends.record.turn_record``
        expects (``input_tokens``/``output_tokens``/``total_tokens``)."""
        usage = (body or {}).get("usage") or {}
        return {
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }


__all__ = [
    "EngineAdapter",
    "EngineKind",
    "DEFAULT_ENGINE_URL",
    "DEFAULT_MODEL",
    "engine_url",
    "engine_kind",
    "api_key",
    "default_model",
    "ready",
]
