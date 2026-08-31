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
them -- this app never touches the TCP payload. Its only two jobs are (1)
netem profile control on *both* of its interfaces so round-trip traffic is
impaired identically in each direction, and (2) reporting whether IP
forwarding is actually active (``aipt.gateway.forwarding``).

Routes:
  * ``GET /health`` -- liveness probe. Reports whether ``tc`` netem
    control is usable (``netem_control.available()``) plus each of the
    two interfaces' individual netem readiness, and whether kernel IP
    forwarding is actually turned on (``forwarding.available()``).
  * ``GET /gateway/profile`` -- the profile currently applied to each of
    the client-facing and backend-facing interfaces.
  * ``POST /gateway/profile`` -- switch the running profile on **both**
    interfaces (client-facing + backend-facing) via
    ``netem_control.apply_profile_both``. Body is ``{"profile": "3g"}``
    for a preset, or ``{"profile": "custom", "delay_ms":.., "jitter_ms":..,
    "loss_pct":.., "reorder_pct":..}`` for arbitrary values. Never 500s on
    a netem failure (e.g. missing CAP_NET_ADMIN) -- it reports ``ok: false``
    with a reason in the body instead (naming which interface(s) failed),
    same honesty contract as ``netem_control.apply_profile_both``.
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
    both interfaces once at container boot, mirroring what
    :func:`set_profile` does for a runtime POST. Without this hook the env
    vars were read but never actually installed via ``tc qdisc`` -- the
    container booted with GATEWAY_DELAY_MS=20 set yet `tc qdisc show`
    stayed `noqueue` until a POST /gateway/profile call. Best-effort like
    every other netem_control call: failures (missing CAP_NET_ADMIN, no
    `tc`) are swallowed here since GET /health already surfaces the same
    underlying reason via netem_control.available().
    """
    netem_control.apply_profile_both(
        netem_control.DEFAULT_CLIENT_IFACE,
        netem_control.DEFAULT_BACKEND_IFACE,
        profiles.from_env(),
    )
    yield


app = FastAPI(
    title="AIPT Network Gateway",
    description=(
        "Pure L3 IP-forwarding + tc netem-based traffic shaping between "
        "net-client and net-backend for MockBackend/LocalLLMBackend "
        "(DESIGN.md 4.7, B9, and 4.7 미해결 세부사항 1)"
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
        "ip_forward_available": forward_ok,
        "ip_forward_reason": forward_reason,
    }


@app.get("/gateway/profile")
def get_profile() -> dict:
    return netem_control.current_profile_both(
        netem_control.DEFAULT_CLIENT_IFACE, netem_control.DEFAULT_BACKEND_IFACE
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
    return netem_control.apply_profile_both(
        netem_control.DEFAULT_CLIENT_IFACE, netem_control.DEFAULT_BACKEND_IFACE, profile
    )


# uvicorn aipt.gateway.app:app
__all__ = ["app"]
