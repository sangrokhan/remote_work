"""aipt.backends.mock.conversation: multi-turn scenario over one keep-alive
socket, plus the ``Backend``-protocol wrapper (``MockBackend``) around it.

Migrated from ``tcp_congestion/tcp_congestion/conversation.py`` (DESIGN.md
5, A3), then extended (DESIGN.md 4.5) to satisfy
``aipt.backends.base.Backend`` so the client's cwnd/capture/export
instrumentation can drive this backend exactly like ``public_ai``/
``local_llm``, and (DESIGN.md 5 B1/B3) to serve scenario-record Q&A / replay
answers instead of only N-byte dummy padding.

Two layers, kept separate on purpose:

  * The **migrated** low-level pieces (``turn_prompt_size``/``build_turns``,
    ``set_congestion_algorithm``, the raw socket ``run()``) are unchanged
    in spirit from the original -- they compute/exercise cumulative-context
    byte growth over a single connection and are exactly what
    ``test_conversation.py``/``test_conversation_live.py`` (migrated to
    ``tests/backends/mock/``) already cover.
  * ``MockBackend`` is new: it satisfies ``connect``/``send_turn``/``close``
    by driving one ``aipt.backends.mock.server.Server`` + one keep-alive
    socket + one ``aipt.core.cwnd.Monitor`` for the connection's whole
    lifetime, same as the client expects from any backend. The natural gap
    *between* two ``send_turn`` calls (the client's own pacing) is the idle
    window the cwnd reset happens in -- ``MockBackend`` does not need to
    sleep for it itself; only the server-side ``inference_delay_ms`` knob
    (never a replayed real latency, per DESIGN.md 4.5) simulates think time.
"""

from __future__ import annotations

import os
import socket
import threading
import time

from aipt.backends.base import Transport
from aipt.backends.mock import probe
from aipt.backends.mock.records import ScenarioRecord
from aipt.backends.mock.server import Server
from aipt.backends.record import Exchange
from aipt.core import capture as capture_mod
from aipt.core import cwnd

# Linux-only sockopt (setsockopt(IPPROTO_TCP, TCP_CONGESTION, b"bbr")).
# socket.TCP_CONGESTION exists on CPython/Linux (value 13); fall back to
# the numeric value so this still imports on a non-Linux interpreter.
TCP_CONGESTION = getattr(socket, "TCP_CONGESTION", 13)


def set_congestion_algorithm(sock: socket.socket, algo: str) -> None:
    """Pin *sock* to congestion-control algorithm *algo* (e.g. "bbr").

    Must be set before connect() -- Linux only applies TCP_CONGESTION to
    the handshake and subsequent behaviour when set pre-connect.
    """
    sock.setsockopt(socket.IPPROTO_TCP, TCP_CONGESTION, algo.encode() + b"\x00")


def get_congestion_algorithm(sock: socket.socket) -> str:
    """Read back the congestion-control algorithm actually in effect."""
    raw = sock.getsockopt(socket.IPPROTO_TCP, TCP_CONGESTION, 16)
    return raw.split(b"\x00", 1)[0].decode(errors="replace")


def turn_prompt_size(turn_index: int, system_prompt_bytes: int,
                      turn_user_msg_bytes: int, history_bytes: int) -> int:
    """Bytes this turn's request body will carry.

    Turn 0: system prompt (once) + this turn's new user message.
    Turn N>0: accumulated history (already contains the system prompt from
    turn 0) + this turn's new user message.
    """
    if turn_index == 0:
        return system_prompt_bytes + turn_user_msg_bytes + history_bytes
    return history_bytes + turn_user_msg_bytes


def build_turns(
    *,
    num_turns: int,
    system_prompt_bytes: int = 0,
    turn_user_msg_bytes: int,
    mock_response_bytes: int,
    inference_delay_ms: int,
    idle_duration_ms: int,
) -> list[dict]:
    """Precompute each turn's prompt size under cumulative-context growth."""
    if num_turns <= 0:
        raise ValueError("num_turns must be positive")

    specs = []
    history = 0
    for i in range(num_turns):
        prompt_bytes = turn_prompt_size(i, system_prompt_bytes,
                                        turn_user_msg_bytes, history)
        specs.append({
            "turn": i,
            "prompt_bytes": prompt_bytes,
            "inference_delay_ms": inference_delay_ms,
            "idle_duration_ms": idle_duration_ms,
        })
        history = prompt_bytes + mock_response_bytes
    return specs


def _http_post(conn: socket.socket, host: str, path: str, body: bytes) -> dict:
    """POST *body*, return the parsed JSON response (empty dict on a
    malformed/empty reply -- callers treat that as "no answer field")."""
    req = (f"POST {path} HTTP/1.1\r\nHost: {host}\r\n"
           f"Connection: keep-alive\r\nContent-Length: {len(body)}\r\n\r\n"
           ).encode() + body
    conn.sendall(req)
    return _drain(conn)


def _drain(conn: socket.socket) -> dict:
    import json as _json
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            return {}
        buf += chunk
    head, _, body = buf.partition(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip() or 0)
    while len(body) < length:
        chunk = conn.recv(4096)
        if not chunk:
            break
        body += chunk
    try:
        return _json.loads(body.decode(errors="replace") or "{}")
    except ValueError:
        return {}


def _connect_with_algorithm(host: str, port: int, algorithm: str | None,
                            timeout: float) -> tuple[socket.socket, str]:
    """Like socket.create_connection(), but sets TCP_CONGESTION pre-connect.

    Returns (socket, algorithm_error) -- algorithm_error is "" on success,
    or a message when the kernel rejected an unknown/unloaded algorithm
    name, so the caller can surface that without aborting the whole run.
    """
    last_exc: OSError | None = None
    for family, socktype, proto, _, sockaddr in socket.getaddrinfo(
            host, port, socket.AF_UNSPEC, socket.SOCK_STREAM):
        sock = socket.socket(family, socktype, proto)
        algorithm_error = ""
        if algorithm:
            try:
                set_congestion_algorithm(sock, algorithm)
            except OSError as exc:
                algorithm_error = (f"could not set congestion algorithm "
                                   f"'{algorithm}': {exc}")
        sock.settimeout(timeout)
        try:
            sock.connect(sockaddr)
            return sock, algorithm_error
        except OSError as exc:
            last_exc = exc
            sock.close()
    raise last_exc or OSError(f"could not connect to {host}:{port}")


def run(
    *,
    host: str,
    port: int,
    num_turns: int = 5,
    system_prompt_bytes: int = 0,
    turn_user_msg_bytes: int = 500,
    mock_response_bytes: int = 400,
    inference_delay_ms: int = 2000,
    idle_duration_ms: int = 3000,
    ping_interval_ms: int = 50,
    label: str = "conversation",
    capture: bool = False,
    algorithm: str | None = None,
    enable_ping_probes: bool = True,
) -> dict:
    """Run one multi-turn conversation over a single keep-alive connection.

    Migrated unchanged in behaviour from
    ``tcp_congestion.conversation.run`` -- the raw byte-size-sweep script
    used directly against a bare ``aipt.backends.mock.server.Server``
    (no scenario record, no Backend-protocol lifecycle). See ``MockBackend`` for
    the Backend-protocol-driven equivalent.
    """
    turns = build_turns(
        num_turns=num_turns,
        system_prompt_bytes=system_prompt_bytes,
        turn_user_msg_bytes=turn_user_msg_bytes,
        mock_response_bytes=mock_response_bytes,
        inference_delay_ms=inference_delay_ms,
        idle_duration_ms=idle_duration_ms,
    )

    monitor = cwnd.Monitor(label, host, port=port)
    monitor.__enter__()

    cap = None
    if capture:
        cap = capture_mod.Capture(
            timestamp=str(time.time()), label=label, host=host, port=port)
        cap.__enter__()

    conn, algorithm_error = _connect_with_algorithm(host, port, algorithm, timeout=30)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        algorithm_actual = get_congestion_algorithm(conn)
    except OSError:
        algorithm_actual = ""
    monitor.announce(conn)

    turn_marks: list[dict] = []
    all_probes: list[dict] = []

    try:
        for spec in turns:
            path = (f"/inference-mock?delay={spec['inference_delay_ms']}"
                    f"&response_bytes={mock_response_bytes}")
            body = b"x" * spec["prompt_bytes"]

            t_send_start = time.monotonic()
            _http_post(conn, host, path, body)
            t_send_end = time.monotonic()

            # idle gap: probe RTT only, no delivery_rate update
            stop_event = threading.Event()
            probe_results: list[dict] = []
            probe_thread = None
            if enable_ping_probes:
                probe_thread = threading.Thread(
                    target=probe.run_probes,
                    args=(conn,),
                    kwargs={
                        "host": host,
                        "interval_ms": ping_interval_ms,
                        "stop": stop_event,
                        "out": probe_results,
                    },
                    daemon=True,
                )
                probe_thread.start()
            time.sleep(spec["idle_duration_ms"] / 1000)
            stop_event.set()
            if probe_thread is not None:
                probe_thread.join(timeout=2)

            all_probes.append({"turn": spec["turn"], "samples": probe_results})
            turn_marks.append({
                "turn": spec["turn"],
                "prompt_bytes": spec["prompt_bytes"],
                "request_ms": (t_send_end - t_send_start) * 1000,
                "idle_ms": spec["idle_duration_ms"],
            })
    finally:
        conn.close()
        monitor.stop()
        if cap is not None:
            cap.__exit__(None, None, None)

    result = monitor.result()
    result["turns"] = turn_marks
    result["probes"] = all_probes
    result["pcap"] = cap.result() if cap is not None else None
    result["algorithm_requested"] = algorithm or ""
    result["algorithm"] = algorithm_actual
    result["algorithm_error"] = algorithm_error
    result["ping_probes_enabled"] = enable_ping_probes
    return result


# --- Backend protocol wrapper ----------------------------------------------


class MockBackend:
    """``aipt.backends.base.Backend`` implementation for the Mock backend.

    Owns one ``aipt.backends.mock.server.Server`` (started on
    ``connect()``, stopped on ``close()``), one keep-alive socket to it,
    and one ``aipt.core.cwnd.Monitor`` running for the connection's whole
    lifetime -- mirroring what ``run()`` above does in one shot, but split
    across the ``connect``/``send_turn``*/``close`` lifecycle the client
    drives turn by turn.

    ``record`` (optional): an ``aipt.backends.mock.records.ScenarioRecord`` (Q&A
    loaded, byte-size-swept, or replay-built via
    ``aipt.backends.mock.replay``) whose ``turns[i].answer`` is served for
    ``send_turn(turn=i, ...)``. Without one, every turn gets the plain
    dummy-byte response padded to ``mock_response_bytes``.

    ``inference_delay_ms`` is the only latency knob (DESIGN.md 4.5: replay
    reproduces byte patterns, never real timing) -- it is sent to the
    server as the ``delay`` query param on every turn.
    """

    NAME = "mock"
    DEFAULT_MODEL = "mock-record"
    # Display-only labels the web UI's arm dropdown offers -- MockBackend
    # never validates ``arm`` against this list (unlike public_ai/
    # local_llm), since what actually drives its behaviour is
    # aipt.web.routes_run.RunRequest.input_mode ("dummy" vs "record") plus
    # whichever ScenarioRecord (if any) gets bound at construction time.
    ARMS = ("dummy", "record")
    HEADLINE_ARMS = ("dummy", "record")

    #: Env vars pointing MockBackend at the standalone `mock-server`
    #: Docker service (docker-compose.yml, DESIGN.md 4.7's L3 topology)
    #: instead of spawning an in-process server on loopback. Mirrors
    #: LOCAL_LLM_ENGINE_URL's "read from the environment, constructor arg
    #: wins if given" pattern (aipt.backends.local_llm.EngineAdapter):
    #: unset (the default in every non-Docker/dev run) means "spawn our
    #: own server exactly as before" -- fully backward compatible.
    #:
    #: Why this matters (found 2026-08-31, user-reported): before this,
    #: EVERY web-UI Mock run bound its server to 127.0.0.1 inside the
    #: `web` container and talked to itself over loopback -- never
    #: touching `net-backend`, and therefore never traversing `gateway`.
    #: The mock-server container and its gateway-routed L3 topology
    #: (docker-compose.yml, already built) was simply never reached by
    #: /api/run -- a Wireshark capture of a
    #: "Mock" run showed loopback traffic with zero netem effect, because
    #: there was no gateway hop in the path to apply netem to at all.
    _MOCK_SERVER_HOST_ENV = "MOCK_SERVER_HOST"
    _MOCK_SERVER_PORT_ENV = "MOCK_SERVER_PORT"

    def __init__(
        self,
        *,
        record: ScenarioRecord | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        mock_response_bytes: int = 400,
        inference_delay_ms: int = 0,
        algorithm: str | None = None,
        label: str = "mock-conversation",
        transport: Transport = "http1",
    ) -> None:
        self.record = record
        self._bind_host = host
        self._bind_port = port
        self.mock_response_bytes = mock_response_bytes
        self.inference_delay_ms = inference_delay_ms
        self.algorithm = algorithm
        self.label = label
        self.transport = transport

        # External target (mock-server container, reached via gateway) if
        # configured -- see _MOCK_SERVER_HOST_ENV's docstring above.
        # Constructor args are not exposed for this on purpose (every
        # other backend reads its external target from the environment
        # only, e.g. LOCAL_LLM_ENGINE_URL -- keeping this the same shape
        # means _build_backend() in routes_run.py needs no special case).
        self._external_host = os.environ.get(self._MOCK_SERVER_HOST_ENV, "").strip()
        _external_port_raw = os.environ.get(self._MOCK_SERVER_PORT_ENV, "").strip()
        self._external_port = int(_external_port_raw) if _external_port_raw else None

        self._server: Server | None = None
        self._server_thread: threading.Thread | None = None
        self._conn: socket.socket | None = None
        self._monitor: cwnd.Monitor | None = None
        self._cwnd_result: dict = {}
        self.algorithm_requested = algorithm or ""
        self.algorithm_actual = ""
        self.algorithm_error = ""
        #: The host/port actually connected to -- set in resolve_target(),
        #: whichever of external-server/in-process-server won.
        self._peer_host: str = ""
        self._peer_port: int = 0
        #: resolve_target() is idempotent -- connect() calls it too (for a
        #: caller that never calls resolve_target() itself, e.g. direct
        #: test/CLI use), so a second call from routes_run.py (which now
        #: calls it explicitly, before opening a capture window) must not
        #: spawn a second server or overwrite an already-resolved target.
        self._target_resolved = False
        #: System prompt bound at connect() (only meaningful in record
        #: mode -- see _request_body_text()'s docstring) and the growing
        #: (question, answer) history built up as send_turn() completes
        #: each turn, both consumed by _request_body_text() to reproduce
        #: real stateless-multi-turn-client cumulative upload growth.
        self._system: str = ""
        self._history: list[tuple[str, str]] = []

    def ready(self) -> tuple[bool, str]:
        return True, "mock backend has no external dependency"

    def api_host(self) -> str:
        if self._peer_host:
            return f"{self._peer_host}:{self._peer_port}"
        return f"{self._bind_host}:{self._bind_port}"

    def resolve_target(self) -> None:
        """Pick (and, for the in-process case, start) the server this
        connection will talk to -- *without* opening the client socket or
        putting a packet on the wire.

        Split out of ``connect()`` (idempotent, safe to call first) so a
        caller that wants a packet capture to see the TCP handshake can
        call this to learn ``api_host()`` and open the capture window
        *before* calling ``connect()`` -- which is exactly what
        ``aipt/web/routes_run.py`` does now. Found via a Wireshark capture
        showing zero handshake packets: capture opened
        after ``connect()`` had already finished the handshake, on every
        backend -- this fixes it for Mock/TCP.
        """
        if self._target_resolved:
            return
        if self._external_host and self._external_port:
            self._peer_host, self._peer_port = self._external_host, self._external_port
        else:
            self._server = Server(host=self._bind_host, port=self._bind_port,
                                   record=self.record)
            self._server_thread = threading.Thread(
                target=self._server.serve_forever, daemon=True,
                name=f"mock-server:{self.label}")
            self._server_thread.start()
            time.sleep(0.05)  # let the listener come up before connecting
            self._peer_host, self._peer_port = self._server.host, self._server.port
        self._target_resolved = True

    def connect(self, arm: str, model: str, system: str) -> None:
        """Open the client TCP connection (the actual handshake) to
        whichever target ``resolve_target()`` picked -- external
        `mock-server` (gateway-routed, netem applies) or an in-process
        server on loopback; see ``_MOCK_SERVER_HOST_ENV``'s docstring
        above. ``system`` is accepted for protocol compatibility but
        unused -- a record's ``system_prompt`` (set at construction) is
        the mock backend's equivalent, since the mock server never
        actually consults a system prompt to generate anything (it IS,
        however, sent on the wire as part of turn 0's request body -- see
        ``send_turn()``'s cumulative-history accounting).

        The external server is generic (no record loaded into it) --
        this backend always asks it for a byte-size-only dummy reply
        (``response_bytes`` computed from the record's own answer length
        when a record is bound) rather than relying on the server to
        echo real answer text, matching DESIGN.md 4.5's stated Mock
        replay philosophy ("바이트 패턴 재현이지 텍스트 동일성 아님" --
        wire byte-size fidelity, not content fidelity). ``send_turn()``
        below reports the record's actual answer text client-side
        either way, so an external-server run and an in-process run
        produce identical ``response_text`` regardless of which path
        was taken.
        """
        self._system = system or ""
        self.resolve_target()

        self._monitor = cwnd.Monitor(self.label, self._peer_host, port=self._peer_port)
        self._monitor.__enter__()

        self._conn, self.algorithm_error = _connect_with_algorithm(
            self._peer_host, self._peer_port, self.algorithm, timeout=30)
        self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self.algorithm_actual = get_congestion_algorithm(self._conn)
        except OSError:
            self.algorithm_actual = ""
        self._monitor.announce(self._conn)

    def _request_body_text(self, turn: int, question: str) -> str:
        """The actual text this turn puts on the wire.

        ``input_mode="dummy"`` (no record bound) already hands ``question``
        in as the *full* cumulative-context text -- ``routes_run._resolve_turns()``
        pre-computed it via ``build_turns()`` (system prompt folded into
        turn 0, running history folded into every turn after), so this
        just passes it through unchanged.

        ``input_mode="record"`` (a ``ScenarioRecord`` bound, e.g.
        ``records/perf.json``) used to do the opposite: ``question`` was
        only that turn's own short question text, sent alone, with no
        system prompt and no prior turns re-sent -- so a 20-turn record
        whose system prompt is 20KB and whose real per-turn text is a few
        hundred bytes came out on the wire as a few-hundred-byte request
        every turn, never growing, which is not what any real stateless
        multi-turn client (the thing Mock's dummy mode explicitly models,
        per ``build_turns``'s own docstring) actually sends -- a real
        client re-uploads the system prompt plus every prior turn's
        question+answer on every new turn. Found 2026-08-31 (user report:
        wire sizes captured under 1000 bytes on every turn of a 20-turn
        record run, implausible for a record with a >=20KB system prompt).

        Fixed the same way for both modes now: reconstruct the growing
        transcript from ``self._system`` (bound at ``connect()``) and
        ``self._history`` (this turn's own prior turns, appended after
        each ``send_turn()`` call below) -- system prompt on EVERY turn
        (not just turn 0), then every prior (question, answer) pair, then
        this turn's own question. System prompt is resent every turn, not
        just folded into turn 0, to match ``build_turns()``'s dummy-mode
        semantics: there, ``system_prompt_bytes`` is added into turn 0's
        ``prompt_bytes``, and that whole figure (system prompt included)
        becomes part of ``history``, which then keeps compounding into
        every later turn's total via ``history = prompt_bytes + mock_response_bytes``
        -- i.e. the system prompt's byte weight never drops out of the
        running total, so a faithful text reconstruction must resend it
        every turn too, not just once. ``self.record is None`` (dummy
        mode) skips this and keeps the old pass-through behaviour, since
        that text is already cumulative from the caller's side and
        re-growing it here would double-count.
        """
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
        # self._server stays None forever when connect() used the
        # external mock-server branch (see connect()'s docstring) -- it
        # is not "still uninitialized" in that case, self._peer_host/
        # self._peer_port are the actual signal of a completed connect().
        # The original check (self._server is None) predates the
        # external-server branch and would raise on every external-server
        # call; found via a real /api/run(transport="http1",
        # backend="mock") run against the gateway-routed mock-server that
        # reported ok=true with 0 turns and this exact RuntimeError
        # buried in run["error"] -- ok=true does not mean 0 turns
        # succeeded, always check turns/error together.
        if self._conn is None or not self._peer_host:
            raise RuntimeError("send_turn called before connect()")
        from aipt.backends.base import progress
        progress(on_progress, backend=self.NAME, arm="record" if self.record else "dummy",
                  phase="steady", turn=turn, turns=len(self.record) if self.record else 0)

        request_text = self._request_body_text(turn, question)
        body = request_text.encode()
        path = (f"/inference-mock?delay={self.inference_delay_ms}"
                f"&response_bytes={self.mock_response_bytes}&turn={turn}")

        t_req_start = time.monotonic()
        try:
            # self._peer_host, not self._server.host -- the latter is
            # None in external-server mode (same root cause as this
            # method's own connect()-guard fix above: self._server is
            # only ever set for the in-process-server branch).
            resp = _http_post(self._conn, self._peer_host, path, body)
            error = None
        except OSError as exc:
            resp = {}
            error = str(exc)
        t_req_end = time.monotonic()

        answer = resp.get("answer", "") if isinstance(resp, dict) else ""
        req_ms = int((t_req_end - t_req_start) * 1000)

        if self.record is not None:
            # Grow history AFTER this turn, from the real question/answer
            # pair -- never from the accumulated request_text itself
            # (which already contains every earlier turn; appending it
            # again would double every turn's contribution on the next
            # request).
            self._history.append((question, answer))

        return Exchange(
            wire_sent=len(body),
            wire_recv=len(answer.encode()) if answer else 0,
            req_payload_bytes=len(body),
            resp_payload_bytes=len(answer.encode()) if answer else 0,
            req_sent_ms=0,
            ttfb_ms=req_ms,
            ttft_ms=req_ms,
            ttlt_ms=req_ms,
            turn_end_ms=req_ms,
            text=answer,
            request_json={"question": question},
            response_json=resp,
            error=error,
        )

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
        if self._monitor is not None:
            self._monitor.stop()
            self._cwnd_result = self._monitor.result()
            self._monitor = None
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=5)
            self._server_thread = None

    def cwnd_result(self) -> dict:
        """The continuous cwnd trace for this connection's lifetime.

        Available both while connected (live snapshot from the still-running
        monitor) and after ``close()`` (the cached result taken at
        ``stop()`` time, since ``close()`` tears the monitor down). Empty
        dict if ``connect()`` was never called.
        """
        if self._monitor is not None:
            return self._monitor.result()
        return self._cwnd_result
