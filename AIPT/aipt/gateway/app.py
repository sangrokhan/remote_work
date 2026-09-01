"""aipt.gateway.app -- the Network Gateway container's standalone FastAPI
mini-app (DESIGN.md 4.7 B9).

Deliberately a **separate app/process** from ``aipt.web`` -- this is meant
to run inside its own container (``docker/Dockerfile.gateway``) sitting on
the network path between the client and ``mock-server``/``local-llm``, not
mounted into the experiment-runner web UI. ``aipt/web`` talks to this over
HTTP (``POST /gateway/profile``) rather than importing it.

DESIGN.md 4.7 "미해결 세부사항" 1 (확정, 2026-08-26): the Gateway is a pure
L3 IP-forwarding container, not an application-level proxy. It straddles
two separate Docker bridge networks (``net-client``, ``net-backend``) and
relies on the kernel (``net.ipv4.ip_forward=1``) to route packets between
them -- this app never touches the TCP payload.

2026-09 client-link-only 재설계 (``aipt.gateway.netem_control`` 모듈
독스트링 참고): Gateway의 두 leg는 더 이상 동일하게 취급되지 않는다.
client-facing leg(``net-client``)만 사용자가 고른 프로파일을 양방향(egress
직접 + ingress는 IFB 리다이렉트)으로 겪고, backend-facing leg
(``net-backend``)는 사용자 선택과 무관하게 항상
``profiles.ETHERNET_BASELINE``(사실상 무손상)만 적용된다 — Gateway<->backend
가 실제로는 같은 데이터센터/호스트 내부의 Ethernet 홉이라는 토폴로지를
반영한 것.

Routes:
  * ``GET /health`` -- liveness probe. Reports whether ``tc`` netem
    control is usable (``netem_control.available()``) plus each of the
    two interfaces' individual netem readiness, and whether kernel IP
    forwarding is actually turned on (``forwarding.available()``).
  * ``GET /gateway/profile`` -- the profile currently applied to the
    client-facing leg (egress + ingress) and the fixed baseline on the
    backend-facing leg.
  * ``POST /gateway/profile`` -- switch the client-facing leg's profile
    (egress + ingress via IFB) via ``netem_control.apply_gateway_profile``;
    the backend-facing leg is re-asserted to ``ETHERNET_BASELINE``
    regardless of the request body. Body is ``{"profile": "wireless"}``
    for a preset, or ``{"profile": "custom", "delay_ms":.., "jitter_ms":..,
    "loss_pct":.., "reorder_pct":..}`` for arbitrary values. Never 500s on
    a netem failure (e.g. missing CAP_NET_ADMIN, missing IFB kernel
    module) -- it reports ``ok: false`` with a reason in the body instead
    (naming which leg/direction failed), same honesty contract as
    ``netem_control.apply_gateway_profile``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from aipt.gateway import forwarding, netem_control, profiles


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """DESIGN.md 4.7 설정 방식 (a): install the env-derived profile
    (``profiles.from_env()`` -- GATEWAY_PROFILE / GATEWAY_DELAY_MS etc.) on
    the client-facing leg (and the fixed baseline on the backend-facing
    leg) once at container boot, mirroring what :func:`set_profile` does
    for a runtime POST. Without this hook the env vars were read but never
    actually installed via ``tc qdisc`` -- the container booted with
    GATEWAY_DELAY_MS=20 set yet `tc qdisc show` stayed `noqueue` until a
    POST /gateway/profile call. Best-effort like every other netem_control
    call: failures (missing CAP_NET_ADMIN, no `tc`, no IFB module) are
    swallowed here since GET /health already surfaces the same underlying
    reason via netem_control.available().
    """
    netem_control.apply_gateway_profile(
        netem_control.DEFAULT_CLIENT_IFACE,
        netem_control.DEFAULT_BACKEND_IFACE,
        netem_control.DEFAULT_IFB_DEV,
        profiles.from_env(),
    )
    yield


app = FastAPI(
    title="AIPT Network Gateway",
    description=(
        "Pure L3 IP-forwarding + tc netem-based traffic shaping, applied "
        "only to the client-facing leg (backend-facing leg is a fixed "
        "Ethernet baseline), between net-client and net-backend for "
        "MockBackend/LocalLLMBackend (DESIGN.md 4.7, B9, and 4.7 미해결 "
        "세부사항 1; 2026-09 client-link-only redesign)"
    ),
    lifespan=_lifespan,
)


class ProfileRequest(BaseModel):
    profile: str
    delay_ms: int = Field(default=0, ge=0)
    jitter_ms: int = Field(default=0, ge=0)
    loss_pct: float = Field(default=0.0, ge=0.0)
    reorder_pct: float = Field(default=0.0, ge=0.0)


@app.get("/health")
def health() -> dict:
    netem_ok, netem_reason = netem_control.available()
    forward_ok, forward_reason = forwarding.available()
    return {
        "status": "ok",
        "netem_available": netem_ok,
        "netem_reason": netem_reason,
        # Deprecated single-iface field, kept for backward compatibility
        # with any existing caller/dashboard reading `iface`.
        "iface": netem_control.DEFAULT_IFACE,
        "client_iface": netem_control.DEFAULT_CLIENT_IFACE,
        "backend_iface": netem_control.DEFAULT_BACKEND_IFACE,
        "ifb_dev": netem_control.DEFAULT_IFB_DEV,
        "ip_forward_available": forward_ok,
        "ip_forward_reason": forward_reason,
    }


@app.get("/gateway/profile")
def get_profile() -> dict:
    return netem_control.current_gateway_profile(
        netem_control.DEFAULT_CLIENT_IFACE,
        netem_control.DEFAULT_BACKEND_IFACE,
        netem_control.DEFAULT_IFB_DEV,
    )


@app.post("/gateway/profile")
def set_profile(req: ProfileRequest) -> dict:
    name = req.profile.strip().lower()
    if name not in profiles.PRESET_NAMES:
        return {
            "ok": False,
            "reason": (
                f"unknown profile {req.profile!r}; known values: "
                f"{', '.join(profiles.PRESET_NAMES)}"
            ),
        }

    profile = profiles.resolve(
        name,
        delay_ms=req.delay_ms,
        jitter_ms=req.jitter_ms,
        loss_pct=req.loss_pct,
        reorder_pct=req.reorder_pct,
    )
    return netem_control.apply_gateway_profile(
        netem_control.DEFAULT_CLIENT_IFACE,
        netem_control.DEFAULT_BACKEND_IFACE,
        netem_control.DEFAULT_IFB_DEV,
        profile,
    )


# uvicorn aipt.gateway.app:app
__all__ = ["app"]
