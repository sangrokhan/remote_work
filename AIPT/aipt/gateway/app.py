"""aipt.gateway.app -- the Network Gateway container's standalone FastAPI
mini-app (DESIGN.md 4.7 B9).

Deliberately a **separate app/process** from ``aipt.web`` -- this is meant
to run inside its own container (``docker/Dockerfile.gateway``) sitting on
the network path between the client and ``mock-server``/``local-llm``, not
mounted into the experiment-runner web UI. ``aipt/web`` talks to this over
HTTP (``POST /gateway/profile``) rather than importing it.

Routes:
  * ``GET /health`` -- liveness probe (also reports whether ``tc`` netem
    control is actually usable in this process, per
    ``netem_control.available()``).
  * ``GET /gateway/profile`` -- the profile currently applied to
    ``GATEWAY_IFACE``.
  * ``POST /gateway/profile`` -- switch the running profile. Body is
    ``{"profile": "3g"}`` for a preset, or
    ``{"profile": "custom", "delay_ms":.., "jitter_ms":.., "loss_pct":..,
    "reorder_pct":..}`` for arbitrary values. Never 500s on a netem
    failure (e.g. missing CAP_NET_ADMIN) -- it reports ``ok: false`` with
    a reason in the body instead, same honesty contract as
    ``netem_control.apply_profile``.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from aipt.gateway import netem_control, profiles

app = FastAPI(
    title="AIPT Network Gateway",
    description="tc netem-based L3/L4 traffic shaping for MockBackend/LocalLLMBackend (DESIGN.md 4.7, B9)",
)


class ProfileRequest(BaseModel):
    profile: str
    delay_ms: int = Field(default=0, ge=0)
    jitter_ms: int = Field(default=0, ge=0)
    loss_pct: float = Field(default=0.0, ge=0.0)
    reorder_pct: float = Field(default=0.0, ge=0.0)


@app.get("/health")
def health() -> dict:
    ok, reason = netem_control.available()
    return {
        "status": "ok",
        "netem_available": ok,
        "netem_reason": reason,
        "iface": netem_control.DEFAULT_IFACE,
    }


@app.get("/gateway/profile")
def get_profile() -> dict:
    profile = netem_control.current_profile(netem_control.DEFAULT_IFACE)
    return {"iface": netem_control.DEFAULT_IFACE, **profile.as_dict()}


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
    result = netem_control.apply_profile(netem_control.DEFAULT_IFACE, profile)
    return {"iface": netem_control.DEFAULT_IFACE, **result}


# uvicorn aipt.gateway.app:app
__all__ = ["app"]
