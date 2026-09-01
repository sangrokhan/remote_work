"""The per-turn record every AIPT backend produces, and its schema version.

Generalized from ``token_traffic/core/record.py`` for the 3-backend
architecture (DESIGN.md 4.5 / 4.6, layer 2 -- ``turns.csv``). One record per
(backend, arm, turn, phase); the turns CSV is one row per record. All three
backends (``public_ai``, ``mock``, ``local_llm``) build their rows through
:func:`turn_record` and nowhere else, because a chart that puts a Gemini turn
next to a local-llama turn or a mock replay is only honest if the columns
mean the same thing regardless of which backend produced them.

Where this differs from the token_traffic original:

  * ``provider`` is renamed ``backend`` -- the neutral term across the three
    kinds of counterparty (public API / mock replay / local engine).
  * ``transport`` is new (DESIGN.md 4.5 B5): the connection kind the turn was
    carried over. Only ``"http1"`` is implemented; the field exists so a
    later QUIC/HTTP3 backend does not require a schema migration.
  * ``goodput_bps`` is new (DESIGN.md 4.6 B7), derived from wire bytes and
    the req_sent_ms..turn_end_ms window, left as 0 until the export layer
    (``aipt/export/turns.py``) fills it in from bytes actually on the wire.

Two fields exist purely so a run can be doubted:

  ``schema_version``  a run written under an older layout can be told apart
                       instead of being silently charted against a newer one.
  ``measure``          bytes from a streamed pass and bytes from a blocking
                        pass are not the same measurement. Averaging them is
                        the mistake this column exists to make visible.

``store_tail_ms`` is the gap between the last answer token and the server
finally letting go, floored at zero: a mark pinned by a failed call must not
be able to push ``ttlt`` past ``turn_end`` and produce a negative wait.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

SCHEMA_VERSION = 1


class TurnExchange(Protocol):
    """What every backend's per-turn call result must expose.

    Any object satisfying this (dataclass, SimpleNamespace, or a backend's
    own exchange type) can be handed to :func:`turn_record`. Backends are
    not required to subclass anything -- this is a structural (duck-typed)
    contract, matching the ``Protocol`` style already used for
    ``token_traffic.providers.base.Provider``.
    """

    wire_sent: int
    wire_recv: int
    req_payload_bytes: int
    resp_payload_bytes: int

    req_sent_ms: int
    ttfb_ms: int
    ttft_ms: int
    ttlt_ms: int
    turn_end_ms: int

    text: str
    request_json: Any
    response_json: Any
    error: str | None


@dataclass
class Exchange:
    """A concrete, minimal :class:`TurnExchange` implementation.

    Backends that don't already have their own exchange object (mock/local
    engines especially) can use this directly instead of hand-rolling a
    class that happens to satisfy the protocol.
    """

    wire_sent: int = 0
    wire_recv: int = 0
    req_payload_bytes: int = 0
    resp_payload_bytes: int = 0

    req_sent_ms: int = 0
    ttfb_ms: int = 0
    ttft_ms: int = 0
    ttlt_ms: int = 0
    turn_end_ms: int = 0

    text: str = ""
    request_json: Any = None
    response_json: Any = None
    error: str | None = None
    # Request-body leaf-hash dedup savings for this turn (docs/
    # engine_gateway_caching_seed.md), local_llm-only, 0 for every other
    # backend/when caching is off. See Gateway.send()'s docstring for how
    # this is computed from the same call rather than a separate baseline.
    cache_bytes_saved: int = 0


def _store_tail(exchange: TurnExchange) -> int:
    """The wait after the last answer token, or 0 when nothing measured it.

    A blocking pass has no ``ttlt``: it never saw a last token, only a
    finished body. So ``turn_end - ttlt`` is not the tail there, it is the
    whole call. A mark nobody took must read as absent, not as
    zero-and-therefore-subtractable.
    """
    if exchange.ttlt_ms <= 0:
        return 0
    return max(0, exchange.turn_end_ms - exchange.ttlt_ms)


def turn_record(
    backend: str,
    arm: str,
    phase: str,
    turn: int,
    question: str,
    measure: str,
    exchange: TurnExchange,
    usage: dict,
    *,
    transport: str = "http1",
    extra: dict | None = None,
) -> dict:
    """One row.

    ``usage`` is already backend-neutral -- the caller has translated its
    counterparty's usage block (Gemini/OpenAI billing fields, or a mock/local
    engine's own token count) into ``input_tokens`` / ``cached_tokens`` /
    ``output_tokens`` / ``reasoning_tokens`` / ``total_tokens`` before it gets
    here. ``phase`` is ``steady`` for the turns that count, or the name of a
    prep phase (``cachegen``, ``setup``) whose cost is real but is setup, not
    traffic, and must never be folded into an arm's totals.

    ``transport`` is the connection kind the turn rode on -- see the module
    docstring. It is not validated against a closed set here: the schema
    slot exists ahead of any backend actually offering more than
    ``"http1"``.
    """
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0) or (input_tokens + output_tokens)

    record = {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "arm": arm,
        "phase": phase,
        "turn": turn,
        "measure": measure,
        "transport": transport,

        "wire_sent": exchange.wire_sent,
        "wire_recv": exchange.wire_recv,
        "req_payload_bytes": exchange.req_payload_bytes,
        "resp_payload_bytes": exchange.resp_payload_bytes,

        "req_sent_ms": exchange.req_sent_ms,
        "ttfb_ms": exchange.ttfb_ms,
        "ttft_ms": exchange.ttft_ms,
        "ttlt_ms": exchange.ttlt_ms,
        "turn_end_ms": exchange.turn_end_ms,
        "store_tail_ms": _store_tail(exchange),

        "input_tokens": input_tokens,
        "cached_tokens": int(usage.get("cached_tokens", 0) or 0),
        "output_tokens": output_tokens,
        "reasoning_tokens": int(usage.get("reasoning_tokens", 0) or 0),
        "total_tokens": total_tokens,

        # Filled in by aipt/export/turns.py (DESIGN.md 4.6 B7); left at 0
        # here because computing it needs the export layer's byte/window
        # conventions, not just this one exchange.
        "goodput_bps": 0.0,

        # local_llm-only leaf-hash request dedup savings (docs/
        # engine_gateway_caching_seed.md) -- getattr rather than direct
        # attribute access because TurnExchange implementations that
        # predate this field (mock/public_ai's own exchange types) are
        # not required to carry it; 0 there reads the same as "caching
        # not applicable to this backend", matching how a mock/local_llm
        # backend's absent probe_rtt_* optional columns already read.
        "cache_bytes_saved": getattr(exchange, "cache_bytes_saved", 0),

        "question": question,
        "response_text": exchange.text,
        "request_raw": exchange.request_json,
        "response_raw": exchange.response_json,
        "error": exchange.error,
    }
    if extra:
        record.update(extra)
    return record
