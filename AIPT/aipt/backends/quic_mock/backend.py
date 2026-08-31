"""aipt.backends.quic_mock.backend -- QuicMockBackend, a Backend-protocol
implementation (aipt.backends.base.Backend) that talks QUIC instead of
plain HTTP/1.1-over-TCP, so the web UI's "Transport" picker
(_experiment_form.html) has something real to select between.

This is the first Backend-protocol integration of the QUIC idle-probe
work landed in earlier commits (aipt/backends/quic_mock/congestion.py,
server.py, spike_runner.py, experiment.py -- all standalone CLI/spike
tools until now). DESIGN.md section 7's step 3 ("UI 편입") is explicitly
deferred there until the algorithm shows a measured improvement -- this
backend does NOT default to "idle_probe"; it defaults to plain "reno"
(the same aioquic-stock algorithm the negative-result experiment used as
baseline) so a user selecting QUIC transport gets a fair baseline unless
they deliberately pick "idle_probe" from the dropdown, at which point
they are trying a still-unproven experimental algorithm on purpose.

Scope, deliberately narrower than the HTTP/1.1 MockBackend it sits next
to in the UI:
  * Server side (``_MockEchoProtocol``) replies with N bytes of filler
    for "dummy" input mode, or the record's answer text for "record" mode
    -- close to what ``aipt.backends.mock.server.Server`` does, but no
    ``inference_delay_ms`` server-side sleep yet (the client's own
    inference_delay_ms sleep between turns still applies, matching every
    other backend's "latency knob is client-side, not server-replayed"
    posture per DESIGN.md 4.5).
  * cwnd tracing: aioquic's congestion control lives entirely in
    userspace, so ``aipt.core.cwnd``'s netlink-based continuous monitor
    (built for kernel TCP sockets) cannot observe it at all -- there is
    no socket inode for netlink sock_diag to query. ``cwnd_result()``
    here reports only a final snapshot (``final_cwnd``), not the
    continuous per-2ms trace TCP mock runs get. This is a real capability
    gap, not a bug -- documented on the returned dict's own ``note``
    field so the UI/CSV export can say so rather than silently showing an
    empty chart.
  * Packet capture: still works unmodified -- ``aipt.core.capture``
    filters tcpdump by (host, port), and QUIC's UDP traffic is exactly as
    capturable as TCP's, just a different L4 protocol in the pcap.

The one connection is driven from a private asyncio event loop running on
a background thread (started in ``connect()``, stopped in ``close()``) --
mirroring ``MockBackend``'s own background-thread server, but here the
thread hosts the asyncio loop asyncio/aioquic need rather than a
``socketserver.ThreadingTCPServer``. Every ``send_turn()`` call is a
synchronous method (matching the ``Backend`` protocol, which every other
backend already implements synchronously) that hands work to that loop
via ``asyncio.run_coroutine_threadsafe`` and blocks on the result -- the
same bridge pattern ``aipt/web/routes_run.py``'s own docstring describes
for its sync-generator-to-async-SSE hand-off, just used in the other
direction here (sync caller, async worker).
"""
from __future__ import annotations

import asyncio
import os
import struct
import threading
import time
from typing import TYPE_CHECKING

from aioquic.asyncio import connect as quic_connect
from aioquic.asyncio import serve as quic_serve
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import QuicEvent, StreamDataReceived

from aipt.backends.base import Transport
from aipt.backends.record import Exchange

# Import side effect: registers "idle_probe" alongside aioquic's own
# built-in "reno"/"cubic" (aioquic.quic.congestion.reno/cubic, imported
# transitively the first time anything asks aioquic to construct one).
from aipt.backends.quic_mock import congestion as _idle_probe_cc  # noqa: F401

if TYPE_CHECKING:
    from aipt.backends.mock.records import ScenarioRecord

DEFAULT_ALGORITHM = "reno"

_CERT_LOCK = threading.Lock()
_CERT_PATHS: tuple[str, str] | None = None


def _ensure_cert() -> tuple[str, str]:
    """A self-signed cert/key pair, generated once per process and reused
    by every QuicMockBackend connection -- QUIC mandates TLS 1.3, and
    this is a loopback-only mock server, so a real CA chain would be
    pure overhead. Never do this for anything internet-facing (same
    caveat as docker/Dockerfile.quic_mock_server's build-time cert)."""
    global _CERT_PATHS
    with _CERT_LOCK:
        if _CERT_PATHS is None:
            import subprocess
            import tempfile

            d = tempfile.mkdtemp(prefix="aipt_quic_mock_cert_")
            cert, key = f"{d}/cert.pem", f"{d}/key.pem"
            subprocess.run(
                ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-keyout", key,
                 "-out", cert, "-days", "1", "-nodes", "-subj", "/CN=aipt-quic-mock"],
                check=True, capture_output=True, timeout=15,
            )
            _CERT_PATHS = (cert, key)
    return _CERT_PATHS


class _MockEchoProtocol(QuicConnectionProtocol):
    """Server side. Each turn's request is
    ``struct.pack(">II", desired_reply_bytes, inference_delay_ms) +
    question_text`` on its own stream (one stream per turn, matching
    HTTP/1.1 MockBackend's one-request-per-turn shape over its single
    keep-alive connection); the reply is ``desired_reply_bytes`` bytes of
    filler, or (if this protocol instance has ``self.record`` bound) the
    record's actual answer text for that turn, sized to its own byte
    length regardless of what the client asked for -- same "record
    answer wins over a byte-size request" precedence as
    ``aipt.backends.mock.server``'s ``_record_answer()``. If
    ``inference_delay_ms`` is nonzero, the reply is held back that long
    before being sent -- the QUIC-side equivalent of
    ``aipt.backends.mock.server``'s ``delay`` query param /
    ``time.sleep(delay_ms / 1000)`` (found missing 2026-08-31: user report
    that a QUIC mock run showed no inference delay at all, verified true
    -- MockBackend's TCP path applies ``inference_delay_ms`` server-side
    on every ``/inference-mock`` request regardless of input_mode, but
    this protocol had no delay field in its wire format at all).

    Backward compatible with an 8-byte-short (4-byte, no delay field)
    request -- an older/mismatched client is treated as delay_ms=0
    rather than a protocol error, since a length-prefix-only client
    (pre-delay-field) is otherwise a legitimate, if outdated, caller of
    this echo protocol.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._buffers: dict[int, bytearray] = {}
        self.record: "ScenarioRecord | None" = None
        self._turn_counter = 0

    def quic_event_received(self, event: QuicEvent) -> None:
        if not isinstance(event, StreamDataReceived):
            return
        buf = self._buffers.setdefault(event.stream_id, bytearray())
        buf.extend(event.data)
        if not event.end_stream:
            return
        data = bytes(buf)
        del self._buffers[event.stream_id]

        if len(data) >= 8:
            requested_len, delay_ms = struct.unpack(">II", data[:8])
        elif len(data) >= 4:
            requested_len, delay_ms = struct.unpack(">I", data[:4])[0], 0
        else:
            requested_len, delay_ms = 400, 0
        turn_idx = self._turn_counter
        self._turn_counter += 1

        body: bytes
        if self.record is not None and turn_idx < len(self.record.turns):
            body = self.record.turns[turn_idx].answer.encode()
        else:
            body = b"x" * requested_len

        if delay_ms > 0:
            # quic_event_received is a synchronous callback -- the sleep
            # has to happen on a scheduled task, not inline, or it would
            # block aioquic's whole event-processing loop (every other
            # connection/stream too, not just this turn) for delay_ms.
            asyncio.ensure_future(self._send_after_delay(event.stream_id, body, delay_ms))
        else:
            self._quic.send_stream_data(event.stream_id, body, end_stream=True)
            self.transmit()

    async def _send_after_delay(self, stream_id: int, body: bytes, delay_ms: int) -> None:
        await asyncio.sleep(delay_ms / 1000)
        self._quic.send_stream_data(stream_id, body, end_stream=True)
        self.transmit()


class _MockClientProtocol(QuicConnectionProtocol):
    """Client side. ``send_turn()`` waits for the *full* reply
    (accumulates fragments until ``end_stream=True``) -- the same
    correctness fix ``aipt.backends.quic_mock.experiment.ThroughputProtocol``
    made over the earlier ``spike_runner.ProbeAwareProtocol``, needed here
    for exactly the same reason: a record's answer text can be larger
    than one packet."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._buffers: dict[int, bytearray] = {}
        self._waiters: dict[int, asyncio.Future] = {}

    def quic_event_received(self, event: QuicEvent) -> None:
        if not isinstance(event, StreamDataReceived):
            return
        buf = self._buffers.setdefault(event.stream_id, bytearray())
        buf.extend(event.data)
        if event.end_stream:
            waiter = self._waiters.pop(event.stream_id, None)
            if waiter and not waiter.done():
                waiter.set_result(bytes(buf))
            del self._buffers[event.stream_id]

    async def send_turn(self, payload: bytes) -> bytes:
        stream_id = self._quic.get_next_available_stream_id()
        waiter = asyncio.get_event_loop().create_future()
        self._waiters[stream_id] = waiter
        self._quic.send_stream_data(stream_id, payload, end_stream=True)
        self.transmit()
        return await waiter


class QuicMockBackend:
    """``aipt.backends.base.Backend`` implementation, QUIC transport.

    Mirrors ``aipt.backends.mock.conversation.MockBackend``'s constructor
    shape (``record``/``mock_response_bytes``/``inference_delay_ms``/
    ``algorithm``/``label``) so ``aipt/web/routes_run.py``'s
    ``_build_backend()`` can construct whichever one it needs with the
    same call site, keyed only on the requested transport.
    """

    NAME = "mock"
    DEFAULT_MODEL = "mock-record"
    ARMS = ("dummy", "record")
    HEADLINE_ARMS = ("dummy", "record")

    #: Env vars pointing this backend at the standalone `quic-mock-server`
    #: Docker service (docker-compose.yml, DESIGN.md 4.7's L3 topology,
    #: DESIGN.md 7.2) instead of spawning an in-process aioquic server on
    #: loopback. Same "unset means spawn our own, exactly as before"
    #: contract as aipt.backends.mock.conversation.MockBackend's
    #: MOCK_SERVER_HOST/MOCK_SERVER_PORT (see that class's own docstring
    #: for the full story of why this matters -- a Wireshark capture of a
    #: web-UI Mock/QUIC run showed pure loopback traffic with zero netem
    #: effect, because there was no gateway hop in the path at all).
    _MOCK_SERVER_HOST_ENV = "QUIC_MOCK_SERVER_HOST"
    _MOCK_SERVER_PORT_ENV = "QUIC_MOCK_SERVER_PORT"

    def __init__(
        self,
        *,
        record: "ScenarioRecord | None" = None,
        host: str = "127.0.0.1",
        port: int = 0,
        mock_response_bytes: int = 400,
        inference_delay_ms: int = 0,
        algorithm: str | None = None,
        label: str = "quic-mock-conversation",
    ) -> None:
        self.record = record
        self._host = host
        self._port = port
        self.mock_response_bytes = mock_response_bytes
        self.inference_delay_ms = inference_delay_ms
        self.algorithm = algorithm or DEFAULT_ALGORITHM
        self.label = label
        self.transport: Transport = "http3"

        self.algorithm_requested = algorithm or ""
        self.algorithm_actual = ""
        self.algorithm_error = ""

        # External target (quic-mock-server container, reached via
        # gateway) if configured -- see _MOCK_SERVER_HOST_ENV's docstring
        # above. Read the same way MockBackend reads MOCK_SERVER_HOST/
        # MOCK_SERVER_PORT -- environment only, no constructor arg, so
        # routes_run.py's _build_backend() needs no special case.
        self._external_host = os.environ.get(self._MOCK_SERVER_HOST_ENV, "").strip()
        _external_port_raw = os.environ.get(self._MOCK_SERVER_PORT_ENV, "").strip()
        self._external_port = int(_external_port_raw) if _external_port_raw else None
        #: Whether connect() spawned its own in-process server (False when
        #: an external quic-mock-server was used instead) -- close() only
        #: tears down a server this instance actually started.
        self._owns_server = False

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server = None
        self._client_cm = None
        self._client: _MockClientProtocol | None = None
        self._cc = None  # aioquic congestion control instance, for cwnd_result()
        #: resolve_target() is idempotent -- connect() calls it too, so a
        #: second explicit call (routes_run.py, to learn api_host() before
        #: opening a capture window) never spawns a second server.
        self._target_resolved = False
        #: System prompt bound at connect() and the growing (question,
        #: answer) history built as send_turn() completes each turn --
        #: same cumulative-context accounting as MockBackend's own
        #: _request_body_text()/self._history (see that docstring for the
        #: full story: a record-mode run without this sends only each
        #: turn's own short question, never growing, which is not what a
        #: real stateless multi-turn client sends).
        self._system: str = ""
        self._history: list[tuple[str, str]] = []

    def ready(self) -> tuple[bool, str]:
        try:
            import aioquic  # noqa: F401
        except ImportError as exc:
            return False, f"aioquic not installed (optional [quic] extra): {exc}"
        return True, "quic mock backend has no external dependency"

    def api_host(self) -> str:
        return f"{self._host}:{self._port}"

    def resolve_target(self) -> None:
        """Start the private asyncio loop + (for the in-process case)
        the aioquic server -- everything short of the actual QUIC
        handshake -- so a caller can learn ``api_host()`` and open a
        packet capture *before* ``connect()`` performs the handshake.

        Split out of ``connect()`` (idempotent, safe to call first) for
        the same reason ``MockBackend.resolve_target()`` was: a Wireshark
        capture of a QUIC mock run showed zero Initial/Handshake packets
        because tcpdump only started after ``connect()`` had already
        finished the full TLS 1.3 handshake -- with no long-header
        packets in the pcap, Wireshark has nothing to identify the
        connection as QUIC by, and displays the (fully 1-RTT-encrypted)
        remainder as plain UDP. ``aipt/web/routes_run.py`` now calls this
        first, opens the capture, and only then calls ``connect()``.
        """
        if self._target_resolved:
            return
        fut = asyncio.run_coroutine_threadsafe(
            self._async_resolve_target(), self._ensure_loop()
        )
        fut.result(timeout=15)
        self._target_resolved = True

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._loop.run_forever, daemon=True, name=f"quic-mock:{self.label}"
            )
            self._thread.start()
        return self._loop

    def connect(self, arm: str, model: str, system: str) -> None:
        """``system`` accepted for protocol compatibility; also bound to
        ``self._system`` for turn-0 cumulative-context accounting -- see
        ``_request_body_text()``'s docstring (shared logic with
        ``MockBackend``, mirrored here since aioquic's async client
        protocol can't share that sync implementation directly)."""
        self._system = system or ""
        loop = self._ensure_loop()
        self.resolve_target()
        fut = asyncio.run_coroutine_threadsafe(self._async_connect(), loop)
        fut.result(timeout=15)

    async def _async_resolve_target(self) -> None:
        cert, key = _ensure_cert()

        if self._external_host and self._external_port:
            # External quic-mock-server (gateway-routed) -- see
            # _MOCK_SERVER_HOST_ENV's docstring. It's a generic
            # long-running process with no record bound to it (unlike the
            # in-process server branch below), so record answer text is
            # never requested from it -- only the byte length, matching
            # MockBackend's own external-server posture (DESIGN.md 4.5:
            # wire byte-size fidelity, not content fidelity, is what Mock
            # replay promises). _async_send_turn() already reports the
            # record's actual text client-side regardless of which branch
            # ran here, so this has no visible effect on response_text.
            self._host, self._port = self._external_host, self._external_port
            self._owns_server = False
        else:
            server_config = QuicConfiguration(is_client=False)
            server_config.load_cert_chain(cert, key)

            def _make_server_protocol(*args, **kwargs):
                proto = _MockEchoProtocol(*args, **kwargs)
                proto.record = self.record
                return proto

            self._server = await quic_serve(
                self._host, self._port, configuration=server_config,
                create_protocol=_make_server_protocol,
            )
            self._owns_server = True
            sock = self._server._transport.get_extra_info("socket")
            if sock is not None:
                bound_host, bound_port = sock.getsockname()[:2]
                self._host, self._port = bound_host, bound_port

    async def _async_connect(self) -> None:
        client_config = QuicConfiguration(
            is_client=True, congestion_control_algorithm=self.algorithm
        )
        client_config.verify_mode = False  # loopback self-signed cert, spike posture
        self._client_cm = quic_connect(
            self._host, self._port, configuration=client_config,
            create_protocol=_MockClientProtocol,
        )
        self._client = await self._client_cm.__aenter__()
        try:
            self._cc = self._client._quic._loss._cc
        except AttributeError:
            self._cc = None
        self.algorithm_actual = self.algorithm

    def _request_body_text(self, turn: int, question: str) -> str:
        """See ``MockBackend._request_body_text()``'s docstring -- same
        cumulative-context reconstruction, duplicated here rather than
        shared because the two backends' send_turn() call this from
        different sync/async contexts."""
        if self.record is None:
            return question
        parts: list[str] = []
        if self._system:
            parts.append(self._system)
        for prior_question, prior_answer in self._history:
            parts.append(prior_question)
            parts.append(prior_answer)
        parts.append(question)
        return "\n\n".join(parts)

    def send_turn(
        self, turn: int, question: str, measure: str, on_progress=None
    ) -> Exchange:
        if self._client is None or self._loop is None:
            raise RuntimeError("send_turn called before connect()")
        from aipt.backends.base import progress

        progress(
            on_progress, backend=self.NAME, arm="record" if self.record else "dummy",
            phase="steady", turn=turn, turns=len(self.record) if self.record else 0,
        )
        fut = asyncio.run_coroutine_threadsafe(
            self._async_send_turn(turn, question), self._loop
        )
        return fut.result(timeout=60)

    async def _async_send_turn(self, turn: int, question: str) -> Exchange:
        response_bytes = self.mock_response_bytes
        if self.record is not None and turn < len(self.record.turns):
            response_bytes = len(self.record.turns[turn].answer.encode())

        request_text = self._request_body_text(turn, question)
        payload = (
            struct.pack(">II", response_bytes, self.inference_delay_ms)
            + request_text.encode()
        )
        t0 = time.monotonic()
        try:
            reply = await self._client.send_turn(payload)
            error = None
        except Exception as exc:
            reply = b""
            error = str(exc)
        t1 = time.monotonic()
        req_ms = int((t1 - t0) * 1000)

        if self.record is not None and turn < len(self.record.turns):
            text = self.record.turns[turn].answer
        else:
            text = reply.decode("utf-8", errors="replace")

        if self.record is not None:
            # Grow history from the real (question, answer) pair, not
            # request_text (which already contains every earlier turn --
            # appending it again would double-count on the next request).
            # `text` here is the record's own canned answer (server-side
            # precedence, see _MockEchoProtocol), matching what a real
            # server would have actually returned for this turn.
            self._history.append((question, text))

        return Exchange(
            wire_sent=len(payload), wire_recv=len(reply),
            req_payload_bytes=len(payload), resp_payload_bytes=len(reply),
            req_sent_ms=0, ttfb_ms=req_ms, ttft_ms=req_ms, ttlt_ms=req_ms,
            turn_end_ms=req_ms, text=text,
            request_json={"question": question}, response_json=None, error=error,
        )

    def close(self) -> None:
        if self._loop is None:
            return
        fut = asyncio.run_coroutine_threadsafe(self._async_close(), self._loop)
        try:
            fut.result(timeout=10)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop.close()
        self._loop = None

    async def _async_close(self) -> None:
        if self._client_cm is not None:
            try:
                await self._client_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._client_cm = None
        if self._server is not None:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None

    def cwnd_result(self) -> dict:
        """Final-snapshot-only cwnd info -- see module docstring on why
        this cannot be a continuous trace the way
        ``aipt.core.cwnd.Monitor`` produces for kernel TCP: aioquic's
        congestion state lives in a plain Python object on this backend's
        own event loop thread, not something netlink sock_diag can query
        for an unrelated process. Empty dict before ``connect()``."""
        if self._cc is None:
            return {}
        return {
            "label": self.label,
            "samples": [],
            "final_cwnd": getattr(self._cc, "congestion_window", None),
            "idle_adjustments": len(getattr(self._cc, "idle_adjustments", [])),
            "note": (
                "QUIC congestion control runs in userspace (aioquic); "
                "aipt.core.cwnd's netlink monitor cannot observe it, so "
                "this is a final-value snapshot, not a continuous trace."
            ),
        }


__all__ = ["QuicMockBackend", "DEFAULT_ALGORITHM"]
