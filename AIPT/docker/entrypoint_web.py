#!/usr/bin/env python3
"""docker/entrypoint_web.py -- container entrypoint wrapper for the AIPT
`web` image (docker/Dockerfile.web, DESIGN.md 4.7 확정 설계 1: L3 라우팅).

DESIGN.md 4.7's confirmed L3 design puts `web` on `net-client` only and
`mock-server`/`local-llm` on `net-backend` only, with `gateway` straddling
both and doing kernel IP forwarding between them. That alone is not
enough to force round-trip traffic through `gateway`: a plain Docker
bridge network gives each container a route to its own subnet only, so
without an explicit route, `web` has no way to reach `net-backend`'s
subnet at all (it isn't attached to that network). This wrapper adds that
route (via `gateway`'s address on `net-client`) *before* handing off to
the real web app, so `web -> net-backend` traffic goes out through
`gateway` (as required for the response leg to also traverse `gateway`,
which is why `mock-server`/`local-llm` need the mirror-image route --
see entrypoint_mockserver.py).

Needs NET_ADMIN in the `web` container (already granted for
aipt.core.cwnd/offload) -- this reuses that same capability grant, no new
one required.

Env vars (set by docker-compose.yml's `web` service):
  GATEWAY_PEER_SUBNET  -- the subnet to route via gateway, e.g. the
                          net-backend CIDR (172.28.2.0/24). If unset/empty,
                          route setup is skipped (e.g. running `web`
                          standalone/bare-metal for dev, not in the
                          gateway-mediated Docker topology).
  GATEWAY_ROUTE_VIA    -- gateway's own IP address on `net-client`, the
                          next hop for GATEWAY_PEER_SUBNET.
  WEB_HOST / WEB_PORT / TRAFFIC_PCAP_DIR -- forwarded to uvicorn/the app
                          unchanged; this wrapper only prepends routing
                          setup, it does not change how the app starts.

Never fails the container over a routing failure -- `ip route add` errors
are logged and swallowed (same honesty-over-crash posture as
aipt.gateway.netem_control/forwarding: report, don't 500/crash). A
missing route just means `web` can't reach net-backend at all until fixed,
which the app's own backend `ready()`/connect-failure paths already
surface.
"""
import os
import subprocess
import sys

PEER_SUBNET = os.environ.get("GATEWAY_PEER_SUBNET", "").strip()
ROUTE_VIA = os.environ.get("GATEWAY_ROUTE_VIA", "").strip()


def _add_route() -> None:
    if not PEER_SUBNET or not ROUTE_VIA:
        print(
            "[entrypoint_web] GATEWAY_PEER_SUBNET/GATEWAY_ROUTE_VIA not set -- "
            "skipping explicit route via gateway (fine for standalone/dev runs "
            "outside the DESIGN.md 4.7 Docker topology)."
        )
        return
    argv = ["ip", "route", "add", PEER_SUBNET, "via", ROUTE_VIA]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        print("[entrypoint_web] `ip` (iproute2) not installed -- cannot add route, continuing anyway.")
        return
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[entrypoint_web] route setup failed: {exc} -- continuing anyway.")
        return
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        # "File exists" -- route already present (container restart with
        # netns reused, or set up twice); not a real failure.
        if "File exists" in err:
            print(f"[entrypoint_web] route to {PEER_SUBNET} via {ROUTE_VIA} already present, skipping.")
            return
        print(
            f"[entrypoint_web] `{' '.join(argv)}` exited {proc.returncode}: {err[:200]} "
            "-- likely missing NET_ADMIN (docker-compose: cap_add: [NET_ADMIN] on `web`). "
            "Continuing anyway; net-backend traffic will not be reachable."
        )
    else:
        print(f"[entrypoint_web] route added: {PEER_SUBNET} via {ROUTE_VIA}")


def main() -> None:
    _add_route()
    host = os.environ.get("WEB_HOST", "0.0.0.0")
    port = os.environ.get("WEB_PORT", "10000")
    argv = [
        "uvicorn",
        "aipt.web.app:create_app",
        "--factory",
        "--host",
        host,
        "--port",
        port,
    ]
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
    sys.exit(0)  # unreachable after execvp on success, kept for clarity
