"""aipt/web/routes_gateway.py -- web-UI proxy for two kinds of runtime
toggle:

1. Network Gateway profile (`/gateway/profile`, DESIGN.md 4.7 B11) --
   tc netem delay/jitter/loss/reorder presets on the `gateway` container.
   Was previously entirely unimplemented: GATEWAY_HOST/GATEWAY_PORT were
   injected into `web`'s environment (docker-compose.yml) but nothing in
   this codebase ever read them or called the Gateway's API -- the 2026-09-01
   ooo audit (docs/seed-2026-09-01-ooo-audit.md, T1) found this as the
   single highest-priority documented-but-missing feature: the Gateway
   backend was complete and directly curl-able, but the web UI had no way
   to reach it.

2. idle-reset (`net.ipv4.tcp_slow_start_after_idle`) toggle -- ALWAYS on
   `web` itself (this process's own /proc/sys, via aipt.core.idle_reset,
   in-process, no network hop). REDESIGNED 2026-09-02 (operator
   correction) after the 2026-09-01 causal experiment
   (docs/experiments/2026-09-01-idle-reset-results.md) found the original
   design measured the wrong side: it's `web`'s own send-side cwnd that
   slow-start-after-idle resets for the metric that matters (next-turn
   request upload latency), not the responding backend's (mock-server's/
   local-llm's). The original design proxied this toggle to an admin route
   on the *responding* backend container (mock-server's `/admin/idle-reset`,
   local-llm's `docker/idle_reset_admin.py` sidecar) -- that whole proxy
   path, those containers' admin routes/sidecars, and the `privileged: true`
   grant they only needed for this write were removed once the redesign
   made them dead code (2026-09-02 operator direction: "제거해"). See
   git history for that code if it's ever needed again.

Never a raw 500 to the browser for either endpoint: `write_ok`/`ok` false
with a reason string instead of an unhandled exception.
"""

from __future__ import annotations

import os

import requests
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

router = APIRouter()

# -- Network Gateway profile proxy --------------------------------------

GATEWAY_HOST_ENV = "GATEWAY_HOST"
GATEWAY_PORT_ENV = "GATEWAY_PORT"
DEFAULT_GATEWAY_HOST = "gateway"
DEFAULT_GATEWAY_PORT = "8080"

# Timeout for admin/control-plane calls -- these never carry experiment
# payload, just a profile name or a sysctl flip, so a short timeout is
# appropriate (unlike backend.send_turn(), which can legitimately take
# seconds for a real inference response).
_ADMIN_TIMEOUT_S = 5


def _gateway_base_url() -> str:
    host = os.environ.get(GATEWAY_HOST_ENV, DEFAULT_GATEWAY_HOST)
    port = os.environ.get(GATEWAY_PORT_ENV, DEFAULT_GATEWAY_PORT)
    return f"http://{host}:{port}"


@router.get("/api/gateway/profile")
def get_gateway_profile():
    """Current Gateway netem profile -- proxies GET {gateway}/gateway/profile."""
    try:
        resp = requests.get(f"{_gateway_base_url()}/gateway/profile", timeout=_ADMIN_TIMEOUT_S)
        return JSONResponse(resp.json(), status_code=resp.status_code)
    except requests.RequestException as exc:
        return JSONResponse({"ok": False, "reason": f"gateway unreachable: {exc}"}, status_code=200)


@router.post("/api/gateway/profile")
def set_gateway_profile(profile: str = Query(..., description="clean|wired|wireless|custom")):
    """Switch the Gateway's active netem profile -- proxies POST
    {gateway}/gateway/profile. Applies to the live L3 hop immediately
    (aipt.gateway.netem_control applies per-interface, no restart needed --
    see DESIGN.md 4.7 '미해결 세부사항' 2)."""
    try:
        resp = requests.post(f"{_gateway_base_url()}/gateway/profile",
                              json={"profile": profile}, timeout=_ADMIN_TIMEOUT_S)
        return JSONResponse(resp.json(), status_code=resp.status_code)
    except requests.RequestException as exc:
        return JSONResponse({"ok": False, "reason": f"gateway unreachable: {exc}"}, status_code=200)


# -- idle-reset toggle (CLIENT side, `web` itself) -------------------------
#
# `web` is this same process's own container: no separate admin server to
# proxy to -- aipt.core.idle_reset is called in-process, directly against
# this container's own /proc/sys. See the module docstring above for why
# this always targets `web`, never a backend container.
from aipt.core import idle_reset as _idle_reset  # noqa: E402


def _web_client_idle_reset_status() -> dict:
    return _idle_reset.status()


def _web_client_idle_reset_write(enabled: bool) -> dict:
    ok, reason = _idle_reset.write(enabled)
    body = _idle_reset.status()
    body["write_ok"] = ok
    body["write_reason"] = reason
    return body


@router.get("/api/idle-reset")
def get_idle_reset():
    """Current net.ipv4.tcp_slow_start_after_idle state on `web` itself
    (the CLIENT side -- see module docstring)."""
    return JSONResponse(_web_client_idle_reset_status())


@router.post("/api/idle-reset")
def set_idle_reset(enabled: bool = Query(..., description="True=Linux default (reset), False=disabled")):
    """Toggle net.ipv4.tcp_slow_start_after_idle on `web` itself -- the
    causal idle-reset TTFT experiment's control knob (2026-09-01 ooo
    interview, redesigned 2026-09-02 to target the client)."""
    return JSONResponse(_web_client_idle_reset_write(enabled))
