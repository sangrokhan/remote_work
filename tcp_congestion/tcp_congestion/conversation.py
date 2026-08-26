"""conversation: run/build a multi-turn scenario with cumulative context.

Turn N's prompt = the running history (all previous turns' user message +
mock response bytes) + this turn's new user message, exactly like a chat
client that resends the whole conversation every turn. This is what makes
turn N's request larger than turn N-1's -- the growth the user asked us to
measure alongside the idle-gap cwnd reset.

`run()` drives one keep-alive connection through all turns, with the
continuous cwnd.Monitor watching from connect to close, so the "cwnd reset
happens on the *next* transmission after idle" moment is captured as an
actual sample, not inferred from two snapshots.
"""

from __future__ import annotations

import socket
import threading
import time

from tcp_congestion import capture as capture_mod
from tcp_congestion import cwnd, probe

# Linux-only sockopt (setsockopt(IPPROTO_TCP, TCP_CONGESTION, b"bbr")).
# socket.TCP_CONGESTION exists on CPython/Linux (value 13); fall back to
# the numeric value so this still imports on a non-Linux interpreter.
TCP_CONGESTION = getattr(socket, "TCP_CONGESTION", 13)


def set_congestion_algorithm(sock: socket.socket, algo: str) -> None:
    """Pin *sock* to congestion-control algorithm *algo* (e.g. "bbr").

    Must be set before connect() -- Linux only applies TCP_CONGESTION to
    the handshake and subsequent behaviour when set pre-connect; setting it
    after data has flowed is undefined for some algorithms. Raises OSError
    if the kernel does not have *algo* loaded (see tcp_congestion.congestion
    for how the web UI surfaces that as actionable guidance up front).
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
    Turn N>0: accumulated history (which already contains the system prompt
    from turn 0) + this turn's new user message. The system prompt is never
    re-sent as "new" bytes after turn 0.
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
    """Precompute each turn's prompt size under cumulative-context growth.

    system_prompt_bytes is folded into turn 0 only; turn_user_msg_bytes is
    added fresh on every turn on top of the running history.
    """
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


def _http_post(conn: socket.socket, host: str, path: str, body: bytes) -> None:
    req = (f"POST {path} HTTP/1.1\r\nHost: {host}\r\n"
           f"Connection: keep-alive\r\nContent-Length: {len(body)}\r\n\r\n"
           ).encode() + body
    conn.sendall(req)
    _drain(conn)


def _drain(conn: socket.socket) -> None:
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            return
        buf += chunk
    head, _, body = buf.partition(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip() or 0)
    while len(body) < length:
        chunk = conn.recv(4096)
        if not chunk:
            return
        body += chunk


def _connect_with_algorithm(host: str, port: int, algorithm: str | None,
                            timeout: float) -> tuple[socket.socket, str]:
    """Like socket.create_connection(), but sets TCP_CONGESTION pre-connect.

    TCP_CONGESTION must be set before connect() to reliably take effect, so
    this cannot use socket.create_connection() (which connects internally).
    Mirrors its getaddrinfo/try-each-family loop. Returns (socket,
    algorithm_error) -- algorithm_error is "" on success, or a message when
    the kernel rejected an unknown/unloaded algorithm name, so the caller
    can surface that without raising and aborting the whole run. socket.
    socket instances don't support arbitrary attribute assignment, hence
    the tuple instead of tagging the socket directly.
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

    Returns a dict with the per-turn schedule, the continuous cwnd trace
    (from tcp_congestion.cwnd.Monitor), and the RTT probe samples collected
    during each idle gap. When `capture=True`, also runs tcpdump for the
    duration of the run and includes its result under "pcap" -- None when
    capture was not requested, a dict (possibly with an "error") otherwise,
    so a caller never has to guess whether capture ran.

    `algorithm`, when given (e.g. "cubic"/"reno"/"bbr"/"vegas"), is applied
    via TCP_CONGESTION before connect(). The result's "algorithm" field
    always reports what the kernel actually used (read back post-connect
    with getsockopt), so a caller can tell a requested-but-unavailable
    algorithm apart from a silent fallback to the socket default.

    `enable_ping_probes` controls whether the idle gap sends periodic HTTP
    PINGs (tcp_congestion.probe) to sample RTT during the wait -- these are
    extra small requests on the same connection, which is exactly the kind
    of "keepalive/health-check traffic during idle" a caller may want to
    exclude to see the *pure* effect of the idle gap on cwnd, with no
    probe-induced ACK/keepalive activity at all. When False, the idle
    duration still elapses (unchanged), only no PINGs are sent and
    "probes" for that turn is an empty sample list.
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
