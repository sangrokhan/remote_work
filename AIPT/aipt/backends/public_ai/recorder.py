"""aipt.backends.public_ai.recorder -- capture real API traffic as fixtures.

DESIGN.md 5 B2: "Public AI backend 호출 시 request/response 원문을 B1 포맷으로
저장" -- when a live (non-mock) call goes out through :mod:`aipt.backends.public_ai`,
this module can capture its raw request/response bodies into a JSON fixture that
:mod:`aipt.backends.mock` (DESIGN.md 5 B3 -- replay) can later play back byte-for-byte.

Two things this module refuses to do:

  1. Store a secret. ``mask_secrets`` strips API keys out of headers and out of any
     string value that looks like one, in both the request headers dict and anywhere
     inside the request/response JSON body -- a Gemini ``x-goog-api-key`` header, an
     OpenAI ``Authorization: Bearer ...`` header, and any stray ``api_key``/``apiKey``
     field a future engine might put in its body. Masking happens before anything
     touches disk, never after.
  2. Guess at a fixture format nobody asked for. The record schema mirrors
     ``token_traffic/fixtures/perf.json`` in shape (a top-level ``system`` +
     ``steps``) and adds `turns`: exactly what a real call produced, keyed the way
     :mod:`aipt.backends.record` already keys everything else -- backend/arm/turn/
     phase -- so a mock replay layer built on this later does not have to invent a
     second vocabulary.

This module never calls Gemini/OpenAI itself. It is handed an already-made
:class:`aipt.backends.public_ai._call.Exchange` (or any duck-typed
``TurnExchange``) by the caller after the real call returned, and it turns that into
one recorded turn. Wiring this into ``GeminiBackend``/``OpenAIBackend`` automatically
(so every live call is captured with no extra call site) is left to the caller --
see :func:`recording_backend` for an opt-in wrapper that does exactly that.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

#: Header names (case-insensitive) whose values are replaced outright.
_SECRET_HEADER_NAMES = {
    "authorization",
    "x-goog-api-key",
    "api-key",
    "x-api-key",
}

#: JSON body key names (case-insensitive) whose *values* are masked wherever they
#: appear, however deep -- a future engine putting a key inline in its body (rather
#: than a header) must not be able to slip one onto disk just because this module
#: doesn't know its adapter yet.
_SECRET_BODY_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "x-goog-api-key",
    "secret",
    "token",
}

#: Values that look like a bearer/API key even under an innocuous key name --
#: `Bearer sk-...`, a long opaque token. Conservative on purpose: this only
#: catches obvious cases, it does not replace the key-name based masking above.
_BEARER_RE = re.compile(r"^Bearer\s+\S+", re.IGNORECASE)

_MASK = "***MASKED***"


def _mask_value(v: Any) -> Any:
    if isinstance(v, str) and (_BEARER_RE.match(v) or len(v) > 20):
        return _MASK
    return _MASK if v else v


def mask_secrets(obj: Any) -> Any:
    """Recursively mask anything shaped like a credential in headers/body/JSON.

    Safe to call on ``None``, headers dicts, or an already-parsed request/response
    body. Returns a new structure; the input is never mutated, so a caller that
    still needs the real value for the call itself is unaffected.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key_l = str(k).lower()
            if key_l in _SECRET_HEADER_NAMES or key_l in _SECRET_BODY_KEYS:
                out[k] = _MASK
            else:
                out[k] = mask_secrets(v)
        return out
    if isinstance(obj, list):
        return [mask_secrets(v) for v in obj]
    return obj


def mask_secrets_json(raw: str) -> str:
    """Mask secrets in a JSON string, returning JSON. Falls back to a flat mask of
    the whole string if it isn't valid JSON (never write an un-maskable blob that
    might contain a key as plain text)."""
    if not raw:
        return raw
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return _MASK if _BEARER_RE.search(raw or "") else raw
    return json.dumps(mask_secrets(parsed))


@dataclass
class RecordedTurn:
    """One captured (backend, arm, turn) exchange, fixture-ready."""

    backend: str
    engine: str          # "gemini" | "openai" -- which public_ai adapter made the call
    arm: str
    turn: int
    phase: str
    question: str
    measure: str
    request_headers: dict = field(default_factory=dict)
    request_json: Any = None
    response_json: Any = None
    response_text: str = ""
    status: int = 0
    error: str | None = None
    wire_sent: int = 0
    wire_recv: int = 0
    recorded_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "backend": self.backend,
            "engine": self.engine,
            "arm": self.arm,
            "turn": self.turn,
            "phase": self.phase,
            "question": self.question,
            "measure": self.measure,
            "request_headers": mask_secrets(self.request_headers),
            "request_json": mask_secrets(self.request_json),
            "response_json": mask_secrets(self.response_json),
            "response_text": self.response_text,
            "status": self.status,
            "error": self.error,
            "wire_sent": self.wire_sent,
            "wire_recv": self.wire_recv,
            "recorded_at": self.recorded_at,
        }


def record_turn(
    *,
    backend: str,
    engine: str,
    arm: str,
    turn: int,
    phase: str,
    question: str,
    measure: str,
    exchange,
    headers: dict | None = None,
) -> RecordedTurn:
    """Build a :class:`RecordedTurn` from a finished exchange.

    ``exchange`` is duck-typed against ``aipt.backends.record.TurnExchange`` (or the
    richer ``aipt.backends.public_ai._call.Exchange``) -- whatever the gemini/openai
    adapter already produced. ``request_json``/``response_json`` on the exchange are
    JSON *strings* (that's what ``_call.Exchange`` carries); this function parses
    them back to structured data so the fixture is readable JSON rather than a
    string-within-a-string, falling back to the raw string if parsing fails.
    """

    def _parse(raw):
        if raw is None:
            return None
        if isinstance(raw, (dict, list)):
            return raw
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return raw

    return RecordedTurn(
        backend=backend,
        engine=engine,
        arm=arm,
        turn=turn,
        phase=phase,
        question=question,
        measure=measure,
        request_headers=dict(headers or {}),
        request_json=_parse(getattr(exchange, "request_json", None)),
        response_json=_parse(getattr(exchange, "response_json", None)),
        response_text=getattr(exchange, "text", "") or "",
        status=getattr(exchange, "status", 0) or 0,
        error=getattr(exchange, "error", None) or None,
        wire_sent=getattr(exchange, "wire_sent", 0) or 0,
        wire_recv=getattr(exchange, "wire_recv", 0) or 0,
    )


class FixtureWriter:
    """Accumulates :class:`RecordedTurn` rows for one run and writes them as a
    fixture JSON file shaped like ``token_traffic/fixtures/perf.json`` (a top-level
    ``system``/``steps`` plus the recorded ``turns``), so a replay layer built on
    top of this (DESIGN.md 5 B3) can read both the original scenario fixture format
    and a recorded-traffic fixture without a format switch.
    """

    def __init__(self, system: str = "", steps: list[str] | None = None) -> None:
        self.system = system
        self.steps = list(steps or [])
        self._turns: list[RecordedTurn] = []

    def add(self, turn: RecordedTurn) -> None:
        self._turns.append(turn)

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "system": self.system,
            "steps": [{"text": s} for s in self.steps],
            "turns": [t.to_dict() for t in self._turns],
        }

    def write(self, path: str | os.PathLike) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return p


def recording_backend(backend, writer: FixtureWriter, *, engine: str):
    """Wrap a live ``Backend`` (GeminiBackend/OpenAIBackend instance) so every
    ``send_turn`` it makes is also captured into ``writer``.

    Returns a thin proxy exposing the same connect/send_turn/close surface -- the
    caller uses the proxy exactly like the backend it wraps, and gets a populated
    fixture as a side effect. Recording is opt-in and additive: nothing about
    ``GeminiBackend``/``OpenAIBackend`` themselves depends on this module, so a
    backend used directly (no recorder) behaves exactly as before.
    """

    class _RecordingProxy:
        NAME = backend.NAME
        DEFAULT_MODEL = backend.DEFAULT_MODEL
        ARMS = backend.ARMS
        HEADLINE_ARMS = backend.HEADLINE_ARMS
        transport = backend.transport

        def __init__(self) -> None:
            self._arm = None

        def ready(self):
            return backend.ready()

        def api_host(self):
            return backend.api_host()

        def connect(self, arm: str, model: str, system: str) -> None:
            self._arm = arm
            backend.connect(arm, model, system)

        def send_turn(self, turn: int, question: str, measure: str, on_progress=None):
            exchange = backend.send_turn(turn, question, measure, on_progress=on_progress)
            recorded = record_turn(
                backend=backend.NAME,
                engine=engine,
                arm=self._arm or "",
                turn=turn,
                phase="steady",
                question=question,
                measure=measure,
                exchange=exchange,
            )
            writer.add(recorded)
            return exchange

        def close(self) -> None:
            backend.close()

    return _RecordingProxy()
