"""What a Backend must supply, and how the client side finds one.

DESIGN.md 4.5 replaces the earlier "external_api lab / synthetic_mock lab"
split with a single client (cwnd/capture/stats export in ``aipt.core`` /
``aipt.export``) that talks to exactly one of three interchangeable
backends:

  * ``public_ai``  (``aipt.backends.public_ai``) -- Gemini / ChatGPT over the
    real network. Generalizes the former ``token_traffic/providers/*``.
  * ``mock``       (``aipt.backends.mock``) -- fixed or replayed JSON I/O,
    generalizes the former ``tcp_congestion`` mock server.
  * ``local_llm``  (``aipt.backends.local_llm``) -- a standard serving engine
    (llama.cpp/vLLM) behind an in-repo gateway; new in this design.

This module defines the protocol all three implement, generalized from
``token_traffic/providers/base.py``'s ``Provider`` protocol. Differences
from that ancestor, and why:

  * ``run_arm`` (one call that owns a whole conversation) is split into an
    explicit lifecycle -- ``connect`` / ``send_turn`` (called once per turn)
    / ``close`` -- because a mock backend has no connection to open and a
    local-llm backend's gateway connection is exactly the thing under
    measurement; a single opaque ``run_arm`` would hide that from the
    client's cwnd/capture instrumentation, which needs to know when the
    connection is up so it can start counting.
  * ``ARMS`` / ``HEADLINE_ARMS`` are unchanged in spirit (an arm is still a
    named calling convention -- e.g. "with cache" vs "no cache" -- and a
    default run still may exclude diagnostic-only arms).
  * ``transport`` is new (DESIGN.md 4.5, "확정된 설계 결정" table, and B5):
    an extension slot for a future HTTP/3-over-QUIC transport. It is a
    plain field, not a branch -- nothing in this module or in the backend
    stubs implements anything beyond ``"http1"`` yet. The slot exists so
    adding a transport later is a new value plus a new code path inside a
    backend, not a protocol change.

As with the provider registry, backends are looked up by name through
``aipt.backends.get()`` (see ``aipt/backends/__init__.py``) rather than by
importing an adapter module directly by name in client code -- a missing or
broken backend must only break the backend actually asked for.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from aipt.backends.record import TurnExchange

#: Wire transport a backend's connection rides on. Only "http1" is
#: implemented anywhere in AIPT today; "http3" is the QUIC extension point
#: reserved by DESIGN.md 4.5 B5 and is not wired into any backend yet --
#: passing it is not an error at the type level, but no backend is
#: obligated to honor it until that follow-up project lands.
Transport = Literal["http1", "http3"]

DEFAULT_TRANSPORT: Transport = "http1"


@runtime_checkable
class Backend(Protocol):
    """One counterparty (public API / mock / local engine), talked to
    through a uniform connect -> send_turn* -> close lifecycle.

    A concrete backend does not have to subclass this -- ``Protocol`` makes
    it a structural contract, matching the existing
    ``token_traffic.providers.base.Provider`` style. Prefer a module-level
    singleton or small class; the client only ever holds an object that
    satisfies this shape.
    """

    #: Registry name, matching the package name under aipt/backends/
    #: (e.g. "public_ai", "mock", "local_llm").
    NAME: str

    #: Model/engine identifier used when the caller doesn't pick one.
    DEFAULT_MODEL: str

    #: Every calling convention this backend can be run under.
    ARMS: tuple[str, ...]

    #: What a default run includes. An arm can be in ARMS and out of
    #: HEADLINE_ARMS because it is a diagnostic rather than something
    #: anyone would ship.
    HEADLINE_ARMS: tuple[str, ...]

    #: Transport this backend instance is configured for. Reserved for
    #: future http3/QUIC use; every backend today only implements "http1"
    #: and may raise if asked for anything else.
    transport: Transport

    def ready(self) -> tuple[bool, str]:
        """(ok, reason). False with an actionable reason, never a bare
        False: a run that dies on a missing key/model/socket must say
        which one."""

    def api_host(self) -> str:
        """The host (or host:port) this backend's connection targets.

        Capture needs it to filter tcpdump down to the traffic this run
        produced, and the client must not have to know which backend is
        running to build that filter. A mock backend that never leaves the
        host should still return whatever loopback address it binds.
        """
        ...

    def connect(self, arm: str, model: str, system: str) -> None:
        """Open whatever connection/session this arm needs.

        Called once before the first ``send_turn`` of a run. A backend
        without a persistent connection (e.g. a stateless mock) may make
        this a no-op, but must still accept the call -- the client always
        calls ``connect`` before ``send_turn`` and always calls ``close``
        after the last one, regardless of backend.
        """
        ...

    def send_turn(
        self, turn: int, question: str, measure: str, on_progress=None
    ) -> TurnExchange:
        """Send one turn and return its exchange (see
        ``aipt.backends.record.TurnExchange``).

        ``on_progress(event)`` is called before the call goes out, not with
        the finished exchange -- a UI has to be able to say "turn 3 of 10,
        in flight" while the call is still outstanding. Use
        ``aipt.backends.base.progress()`` to emit the event in the shape
        the client expects.
        """
        ...

    def close(self) -> None:
        """Release whatever ``connect`` opened.

        Must be safe to call even if ``connect`` was a no-op or if no turn
        was ever sent (e.g. the run failed ``ready()``).
        """
        ...


def progress(
    on_progress, backend: str, arm: str, phase: str, turn: int, turns: int
) -> None:
    """Announce the call about to be made, *before* making it.

    Every backend emits the same event shape, because the UI/instrumentation
    that renders it cannot know which backend is running. Without the
    announcement an arm sits at turn 0 of N for its whole run, and a stall is
    indistinguishable from progress.

    The event also carries the measurement window, which is why ``phase``
    must be exact and why the announcement must precede the call: the client
    opens capture on the first ``steady`` event and closes it on a
    ``teardown`` one, so a prep call (cache build, session setup) announced
    as prep stays outside the pcap, and an arm with neither phase is
    captured whole.
    """
    if on_progress:
        on_progress(
            {
                "backend": backend,
                "arm": arm,
                "phase": phase,
                "turn": turn,
                "turns": turns,
            }
        )
