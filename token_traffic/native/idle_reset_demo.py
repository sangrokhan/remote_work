#!/usr/bin/env python3
"""Show the idle reset happening, on whatever path you point it at.

The automated suite cannot assert this. It only has loopback, where the round trip is
tens of microseconds and a restarted window climbs all the way back inside a single
10 ms sample -- so the drop is there, but whether a sample lands on it is luck. A real
path does not have that problem: at 34 ms RTT slow start takes hundreds of milliseconds
to climb, and every step of it lands in its own sample.

So this is the demonstration, run by hand, against a host you choose:

    python native/idle_reset_demo.py --host api.openai.com
    python native/idle_reset_demo.py --host 192.0.2.10 --port 8443 --idle 5

It opens one TLS connection, sends a small HTTP request, waits `--idle` seconds doing
nothing -- which is what a client does while a model is thinking -- sends the same
request again, and prints the congestion window across the whole thing.

What it costs: one TLS handshake and two HTTP requests to whatever `--host` is. Against
an LLM vendor those are unauthenticated and get a 401 back, which is a real answer over
a real connection and is all this needs. It is not a chat completion and it is not
billed. Nothing here should be pointed at a host you are not allowed to probe.
"""

from __future__ import annotations

import argparse
import socket
import ssl
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import cwnd  # noqa: E402


def probe(host: str, port: int, idle: float, path: str) -> dict:
    ok, reason = cwnd.available()
    if not ok:
        sys.exit(f"cwnd monitor unavailable: {reason}")

    request = (f"GET {path} HTTP/1.1\r\nHost: {host}\r\n"
               "User-Agent: cwnd-idle-reset-demo\r\nConnection: keep-alive\r\n\r\n"
               ).encode()

    monitor = cwnd.Monitor("demo", "idle", host, port=port)
    monitor.__enter__()
    marks = []
    try:
        time.sleep(0.3)                       # ticking before the SYN goes out
        ctx = ssl.create_default_context()
        raw = socket.create_connection((host, port), timeout=30)
        sock = ctx.wrap_socket(raw, server_hostname=host)
        try:
            marks.append(("connected", time.monotonic()))

            sock.sendall(request)
            _drain(sock)
            marks.append(("request 1 answered", time.monotonic()))

            time.sleep(idle)                  # the model, thinking
            marks.append(("idle over", time.monotonic()))

            sock.sendall(request)
            _drain(sock)
            marks.append(("request 2 answered", time.monotonic()))
            time.sleep(0.5)                   # let the tail land in the samples
        finally:
            sock.close()
    finally:
        monitor.stop()

    return {"result": monitor.result(), "marks": marks}


def _drain(sock) -> None:
    """Read one HTTP response. Enough of it that the exchange is over -- the point is
    to make the connection go quiet afterwards, not to parse anything."""
    sock.settimeout(30)
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            return
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip() or 0)
    while len(rest) < length:
        chunk = sock.recv(65536)
        if not chunk:
            return
        rest += chunk


def report(out: dict, host: str, idle: float) -> int:
    result = out["result"]
    samples = result["samples"]
    if not samples:
        print(f"no samples. {result['error'] or 'the socket never matched the filter'}")
        return 1

    t0 = out["marks"][0][1] if out["marks"] else 0

    print(f"host        {host}")
    print(f"sockets     {', '.join(result['sockets'])}")
    print(f"samples     {result['sample_count']} over {result['seconds']}s "
          f"at {result['interval_ms']}ms")
    print(f"cwnd        peak {result['peak_cwnd']}, final {result['final_cwnd']}")
    print(f"idle resets {result['idle_resets']}")
    print()

    for name, at in out["marks"]:
        print(f"  {(at - t0) * 1000:9.0f} ms  {name}")
    print()

    for event in result["reset_events"]:
        print(f"  RESET at {event['t_ms']:.0f} ms: cwnd {event['from']} -> "
              f"{event['to']} on {event['local']}")

    # The series itself, thinned to something a terminal can hold. cwnd is the column
    # that matters; ssthresh says where slow start would hand off, and rtt says what
    # each round trip the reset costs is worth.
    print("\n     t_ms   cwnd  ssthresh    rtt_ms  state")
    step = max(1, len(samples) // 60)
    for s in samples[::step]:
        ssth = s["snd_ssthresh"]
        ssth = "inf" if ssth >= (1 << 30) else str(ssth)
        print(f"  {s['t_ms']:7.0f}  {s['snd_cwnd']:5d}  {ssth:>8}  "
              f"{s['rtt_us'] / 1000:8.2f}  {s['ca_state']}")

    print()
    if result["idle_resets"]:
        print(f"VERDICT: the window was reset after the {idle}s idle gap. Every turn "
              f"after the first re-enters slow start on this path.")
    else:
        print(f"VERDICT: the window survived the {idle}s idle gap. Either "
              f"net.ipv4.tcp_slow_start_after_idle is 0 here, or the window never grew "
              f"past {cwnd.INIT_CWND} for there to be anything to lose.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True, help="the host to open one connection to")
    ap.add_argument("--port", type=int, default=443)
    ap.add_argument("--path", default="/", help="request path (a 401 is a fine answer)")
    ap.add_argument("--idle", type=float, default=5.0,
                    help="seconds of silence between the two requests")
    args = ap.parse_args()

    return report(probe(args.host, args.port, args.idle, args.path),
                  args.host, args.idle)


if __name__ == "__main__":
    raise SystemExit(main())
