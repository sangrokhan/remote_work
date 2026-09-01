"""aipt/web/routes_gateway.py -- web-UI proxy for two kinds of runtime
toggle that live on the *backend* containers, not on `web` itself:

1. Network Gateway profile (`/gateway/profile`, DESIGN.md 4.7 B11) --
   tc netem delay/jitter/loss/reorder presets on the `gateway` container.
   Was previously entirely unimplemented: GATEWAY_HOST/GATEWAY_PORT were
   injected into `web`'s environment (docker-compose.yml) but nothing in
   this codebase ever read them or called the Gateway's API -- the 2026-09-01
   ooo audit (docs/seed-2026-09-01-ooo-audit.md, T1) found this as the
   single highest-priority documented-but-missing feature: the Gateway
   backend was complete and directly curl-able, but the web UI had no way
   to reach it.

2. idle-reset (`net.ipv4.tcp_slow_start_after_idle`) toggle on the
   *responding* backend -- mock-server's own `/admin/idle-reset` route
   (aipt/backends/mock/server.py) or local-llm's sidecar admin server
   (docker/idle_reset_admin.py) -- for the causal idle-reset TTFT
   experiment (2026-09-01 ooo interview). This is deliberately proxied
   through `web` rather than exposed directly: mock-server/local-llm have
   no `ports:` mapping (DESIGN.md 4.7 topology -- reachable only via
   `gateway`'s L3 forwarding or, for admin traffic that must not be
   netem-impaired, via `web`'s direct net-backend route, the same one
   LOCAL_LLM_ENGINE_URL/MOCK_SERVER_HOST already use for inference calls).

Both proxies share the same posture as `aipt.gateway.forwarding`/
`netem_control`: never a raw 500 to the browser. A backend that is
unreachable, lacks CAP_NET_ADMIN, or isn't running this build yet reports
`{"ok": false, "reason": ...}` with a 200 (or the backend's own non-2xx,
passed through) rather than an unhandled exception.
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


# -- idle-reset toggle proxy (mock-server / local-llm) --------------------

# Same env-var names + defaults aipt.backends.mock.conversation and
# aipt.backends.local_llm.engine_adapter already use to *reach* these
# backends for inference calls -- reused here so the admin toggle targets
# exactly the container a run against that backend will actually talk to,
# with no separate config surface to keep in sync.
_MOCK_SERVER_HOST_ENV = "MOCK_SERVER_HOST"
_MOCK_SERVER_PORT_ENV = "MOCK_SERVER_PORT"
_DEFAULT_MOCK_HOST = "mock-server"
_DEFAULT_MOCK_PORT = "8888"

_LOCAL_LLM_ENGINE_URL_ENV = "LOCAL_LLM_ENGINE_URL"
_DEFAULT_LOCAL_LLM_ENGINE_URL = "http://127.0.0.1:40080"
# The idle-reset sidecar (docker/idle_reset_admin.py) listens one port
# above the engine port by convention (40081 next to 40080) -- see that
# module's own IDLE_RESET_ADMIN_PORT default.
_DEFAULT_IDLE_RESET_ADMIN_PORT_OFFSET = 1


def _mock_admin_url() -> str:
    host = os.environ.get(_MOCK_SERVER_HOST_ENV, _DEFAULT_MOCK_HOST)
    port = os.environ.get(_MOCK_SERVER_PORT_ENV, _DEFAULT_MOCK_PORT)
    return f"http://{host}:{port}/admin/idle-reset"


def _local_llm_admin_url() -> str:
    engine_url = os.environ.get(_LOCAL_LLM_ENGINE_URL_ENV, _DEFAULT_LOCAL_LLM_ENGINE_URL)
    # engine_url is like "http://172.28.2.4:40080" -- the admin sidecar is
    # the same host, port+1, same reasoning as
    # docker/idle_reset_admin.py's IDLE_RESET_ADMIN_PORT default (40081).
    scheme, _, rest = engine_url.partition("://")
    host, _, port_s = rest.partition(":")
    try:
        admin_port = int(port_s or "40080") + _DEFAULT_IDLE_RESET_ADMIN_PORT_OFFSET
    except ValueError:
        admin_port = 40081
    return f"{scheme}://{host}:{admin_port}/admin/idle-reset"


def _admin_url_for(backend: str) -> str | None:
    if backend == "mock":
        return _mock_admin_url()
    if backend == "local_llm":
        return _local_llm_admin_url()
    return None  # public_ai/quic_mock: no admin toggle (real internet / spike-only)


@router.get("/api/idle-reset")
def get_idle_reset(backend: str = Query(..., description="mock|local_llm")):
    """Current net.ipv4.tcp_slow_start_after_idle state on *backend*'s
    responding side. Proxies GET {backend admin}/admin/idle-reset."""
    url = _admin_url_for(backend)
    if url is None:
        return JSONResponse(
            {"ok": False, "reason": f"idle-reset toggle not available for backend={backend!r} "
                                     "(only mock and local_llm run inside this project's own "
                                     "containers; public_ai is the real internet, quic_mock is UDP)"},
            status_code=200,
        )
    try:
        resp = requests.get(url, timeout=_ADMIN_TIMEOUT_S)
        return JSONResponse(resp.json(), status_code=resp.status_code)
    except requests.RequestException as exc:
        return JSONResponse({"ok": False, "reason": f"{backend} admin unreachable: {exc}"}, status_code=200)


@router.post("/api/idle-reset")
def set_idle_reset(backend: str = Query(..., description="mock|local_llm"),
                    enabled: bool = Query(..., description="True=Linux default (reset), False=disabled")):
    """Toggle net.ipv4.tcp_slow_start_after_idle on *backend*'s responding
    side -- the causal idle-reset TTFT experiment's control knob
    (2026-09-01 ooo interview). Proxies POST {backend admin}/admin/idle-reset."""
    url = _admin_url_for(backend)
    if url is None:
        return JSONResponse(
            {"ok": False, "reason": f"idle-reset toggle not available for backend={backend!r}"},
            status_code=200,
        )
    try:
        resp = requests.post(url, params={"enabled": "1" if enabled else "0"}, timeout=_ADMIN_TIMEOUT_S)
        return JSONResponse(resp.json(), status_code=resp.status_code)
    except requests.RequestException as exc:
        return JSONResponse({"ok": False, "reason": f"{backend} admin unreachable: {exc}"}, status_code=200)
