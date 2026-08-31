#!/usr/bin/env python3
"""docker/entrypoint_quic_mock_server.py -- container entrypoint for the
QUIC idle-probe spike's echo server (aipt/backends/quic_mock/,
DESIGN.md "QUIC idle-probe spike" section).

Mirrors docker/entrypoint_mockserver.py's routing setup exactly (same
DESIGN.md 4.7 확정 설계 1 L3 topology: this service lives on net-backend
only and needs an explicit route back to net-client via `gateway`, or
its UDP replies have no way back). The only difference from the
HTTP/1.1 mock-server entrypoint is which server module it starts.

Env vars:
  QUIC_MOCK_HOST (default "0.0.0.0")
  QUIC_MOCK_PORT (default "4433")
  QUIC_MOCK_CERT (default "/app/quic_cert/cert.pem")
  QUIC_MOCK_KEY  (default "/app/quic_cert/key.pem")
  GATEWAY_PEER_SUBNET -- net-client CIDR to route via gateway
  GATEWAY_ROUTE_VIA   -- gateway's own IP address on net-backend
"""
import asyncio
import os
import subprocess
import sys

sys.path.insert(0, "/app")

PEER_SUBNET = os.environ.get("GATEWAY_PEER_SUBNET", "").strip()
ROUTE_VIA = os.environ.get("GATEWAY_ROUTE_VIA", "").strip()


def _add_route() -> None:
    if not PEER_SUBNET or not ROUTE_VIA:
        print(
            "[entrypoint_quic_mock_server] GATEWAY_PEER_SUBNET/GATEWAY_ROUTE_VIA "
            "not set -- skipping explicit route via gateway (fine for "
            "standalone/dev runs outside the Docker topology)."
        )
        return
    argv = ["ip", "route", "add", PEER_SUBNET, "via", ROUTE_VIA]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        print("[entrypoint_quic_mock_server] `ip` (iproute2) not installed -- continuing anyway.")
        return
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[entrypoint_quic_mock_server] route setup failed: {exc} -- continuing anyway.")
        return
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if "File exists" in err:
            print(f"[entrypoint_quic_mock_server] route to {PEER_SUBNET} via {ROUTE_VIA} already present, skipping.")
            return
        print(
            f"[entrypoint_quic_mock_server] `{' '.join(argv)}` exited {proc.returncode}: {err[:200]} "
            "-- likely missing NET_ADMIN. Continuing anyway; replies may not traverse gateway correctly."
        )
    else:
        print(f"[entrypoint_quic_mock_server] route added: {PEER_SUBNET} via {ROUTE_VIA}")


_add_route()

from aipt.backends.quic_mock.server import run_server  # noqa: E402
from aipt.backends.quic_mock.backend import _MockEchoProtocol  # noqa: E402

HOST = os.environ.get("QUIC_MOCK_HOST", "0.0.0.0")
PORT = int(os.environ.get("QUIC_MOCK_PORT", "4433"))
CERT = os.environ.get("QUIC_MOCK_CERT", "/app/quic_cert/cert.pem")
KEY = os.environ.get("QUIC_MOCK_KEY", "/app/quic_cert/key.pem")


async def _main() -> None:
    import signal

    server = await run_server(HOST, PORT, CERT, KEY, create_protocol=_MockEchoProtocol)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    server.close()


if __name__ == "__main__":
    asyncio.run(_main())
