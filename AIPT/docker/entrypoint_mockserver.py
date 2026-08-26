#!/usr/bin/env python3
"""docker/entrypoint_mockserver.py -- container entrypoint for the AIPT
mock-server image (docker/Dockerfile.mockserver, DESIGN.md 4.7 B10, and
4.7 미해결 세부사항 1: 확정 L3 라우팅 설계).

``aipt.backends.mock.server.Server`` (see that module) is a plain
``socketserver.ThreadingTCPServer`` subclass with no ``if __name__ ==
"__main__":`` block of its own -- it's designed to be started in-process by
``aipt.backends.mock.conversation.MockBackend`` for the default (non-Docker)
run path. This script is the standalone-process equivalent for the
Docker topology: it binds ``Server(host, port)`` and calls
``serve_forever()`` so the container can run this backend out-of-process,
reachable over the network from the ``gateway``/``web`` services (DESIGN.md
4.7's "MockBackend must only be reachable via Gateway" topology decision).

Deliberately NOT applying tc netem/offload here -- DESIGN.md 4.7 moved that
responsibility to the dedicated ``gateway`` container (``aipt/gateway/``,
``docker/Dockerfile.gateway``); this container only serves fixed/dummy
inference-mock responses.

**Routing (added for the L3 확정 설계, 2026-08-26)**: ``mock-server`` lives
on ``net-backend`` only and has no route to ``net-client`` (where ``web``
lives) by default -- a plain Docker bridge network only gives a container a
route to its own subnet. Without an explicit route back through
``gateway``, response packets to ``web`` would have nowhere to go (or, on
some Docker network drivers, could take a shortcut that bypasses
``gateway`` entirely, defeating the whole point of routing traffic through
it for netem). Before starting the server, this entrypoint adds
``ip route add <net-client subnet> via <gateway's net-backend IP>`` so the
response leg is also forced through ``gateway``, mirroring
``docker/entrypoint_web.py``'s route setup on the other side.

Needs NET_ADMIN in the ``mock-server`` container for the ``ip route add``
call (docker-compose.yml: ``cap_add: [NET_ADMIN]`` on `mock-server`) --
route failures are logged and swallowed, never crash the container (same
honesty-over-crash posture as ``aipt.gateway.netem_control``).

Env vars:
  MOCK_HOST (default "0.0.0.0")
  MOCK_PORT (default "8888")
  GATEWAY_PEER_SUBNET -- net-client CIDR to route via gateway
  GATEWAY_ROUTE_VIA   -- gateway's own IP address on net-backend
"""
import os
import subprocess
import sys

sys.path.insert(0, "/app")

PEER_SUBNET = os.environ.get("GATEWAY_PEER_SUBNET", "").strip()
ROUTE_VIA = os.environ.get("GATEWAY_ROUTE_VIA", "").strip()


def _add_route() -> None:
    if not PEER_SUBNET or not ROUTE_VIA:
        print(
            "[entrypoint_mockserver] GATEWAY_PEER_SUBNET/GATEWAY_ROUTE_VIA not "
            "set -- skipping explicit route via gateway (fine for "
            "standalone/dev runs outside the DESIGN.md 4.7 Docker topology)."
        )
        return
    argv = ["ip", "route", "add", PEER_SUBNET, "via", ROUTE_VIA]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        print("[entrypoint_mockserver] `ip` (iproute2) not installed -- cannot add route, continuing anyway.")
        return
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[entrypoint_mockserver] route setup failed: {exc} -- continuing anyway.")
        return
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if "File exists" in err:
            print(f"[entrypoint_mockserver] route to {PEER_SUBNET} via {ROUTE_VIA} already present, skipping.")
            return
        print(
            f"[entrypoint_mockserver] `{' '.join(argv)}` exited {proc.returncode}: {err[:200]} "
            "-- likely missing NET_ADMIN (docker-compose: cap_add: [NET_ADMIN] on `mock-server`). "
            "Continuing anyway; response traffic may not traverse gateway correctly."
        )
    else:
        print(f"[entrypoint_mockserver] route added: {PEER_SUBNET} via {ROUTE_VIA}")


_add_route()

from aipt.backends.mock.server import Server  # noqa: E402

host = os.environ.get("MOCK_HOST", "0.0.0.0")
port = int(os.environ.get("MOCK_PORT", "8888"))
print(f"[mock-server] listening on {host}:{port}")

srv = Server(host=host, port=port)
try:
    srv.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    srv.shutdown()
