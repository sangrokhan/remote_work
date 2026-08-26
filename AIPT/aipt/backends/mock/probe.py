"""aipt.backends.mock.probe: idle-period HTTP PING for RTT measurement only.

Migrated verbatim from ``tcp_congestion/tcp_congestion/probe.py``
(DESIGN.md 5, A3/mock migration) -- no merge target existed in
token_traffic, so this is a straight move.

Sends a GET /ping on an already-open keep-alive socket and measures the
wall-clock round trip. delivery_rate is intentionally NOT included in the
sample -- it would be corrupted by the tiny probe payload (~100 B),
producing a nonsense BW estimate. Callers inject the TCP_INFO snapshot
value taken right after the preceding data transfer.
"""

from __future__ import annotations

import socket
import threading
import time


def ping(conn: socket.socket, host: str = "localhost") -> dict:
    """Send one HTTP PING and return {ts, rtt_ms}."""
    req = (f"GET /ping HTTP/1.1\r\nHost: {host}\r\nConnection: keep-alive\r\n\r\n"
           ).encode()
    t0 = time.monotonic()
    conn.sendall(req)
    _drain_one(conn)
    rtt_ms = (time.monotonic() - t0) * 1000
    return {"ts": time.time(), "rtt_ms": rtt_ms}


def _drain_one(conn: socket.socket) -> None:
    """Read exactly one HTTP response from *conn*."""
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


def run_probes(
    conn: socket.socket,
    *,
    host: str = "localhost",
    interval_ms: int = 50,
    stop: threading.Event,
    out: list,
) -> None:
    """Send PINGs every *interval_ms* until *stop* is set; append to *out*."""
    while not stop.is_set():
        try:
            sample = ping(conn, host=host)
            out.append(sample)
        except OSError:
            break
        stop.wait(timeout=interval_ms / 1000)
