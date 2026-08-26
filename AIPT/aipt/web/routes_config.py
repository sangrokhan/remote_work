"""GET / (landing) and GET /api/config -- what the operator needs before
picking a backend and spending anything.

DESIGN.md 4.5 replaced the "external_api lab / synthetic_mock lab" landing
page with a **backend-selection** landing page: three cards (public_ai /
mock / local_llm), because the client now always talks to exactly one
backend chosen at run time, not one lab mounted at its own URL prefix.
`aipt.backends.names()` is the single source of truth for which backends
exist; this module never hardcodes the three names beyond display labels.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import aipt.backends as backends_registry
from aipt.core import capture as capture_mod
from aipt.core import cwnd as cwndmon

router = APIRouter()

# Congestion-control algorithms surfaced in the experiment form. Not probed
# from /proc/sys/net/ipv4/tcp_available_congestion_control here -- that file
# is host-specific and a CI box may only have cubic/reno loaded, which would
# make the dropdown lie about what a production host supports. The fixed
# list mirrors tcp_congestion's original UI; MockBackend.connect() still
# reports algorithm_error if the kernel does not actually have one loaded.
CONGESTION_ALGORITHMS = ("cubic", "reno", "bbr", "vegas", "bic", "htcp")

#: Backend display metadata for the landing page. Keyed by the same names
#: aipt.backends.names() returns, so a card is never shown for a backend
#: the registry does not know, and never silently missing one it does.
_BACKEND_DISPLAY = {
    "public_ai": {
        "label": "Public AI",
        "description": "Gemini / ChatGPT over the real network (billed).",
    },
    "mock": {
        "label": "Mock",
        "description": "Fixed or replayed JSON I/O, no network cost.",
    },
    "local_llm": {
        "label": "Local LLM",
        "description": "Standard serving engine (llama.cpp/vLLM) behind an in-repo gateway.",
    },
}

# The attribute names that mark a backend module as having a real
# Backend-protocol implementation, checked by presence rather than a
# hardcoded name list -- so "implemented" flips automatically the moment a
# parallel work stream (e.g. DESIGN.md 5 B4 for local_llm) lands its real
# class, with no change needed here.
_FACADE_ATTRS = ("PublicAIBackend", "MockBackend", "LocalLLMBackend")


def _facade_class(module):
    """The Backend-protocol class this backend module exposes, or None for
    an import-path-only stub (``NotImplementedBackend``)."""
    for attr in _FACADE_ATTRS:
        facade = getattr(module, attr, None)
        if facade is not None:
            return facade
    return None


def _backend_ready(name: str) -> tuple[bool, str]:
    """(ok, reason) for one backend, without ever letting an unimplemented
    or misconfigured backend's constructor crash the config/landing page."""
    try:
        module = backends_registry.get(name)
    except KeyError as exc:
        return False, str(exc)
    facade = _facade_class(module)
    if facade is None:
        return False, "not implemented yet"
    try:
        return facade().ready()
    except NotImplementedError as exc:
        return False, f"not implemented yet: {exc}"
    except Exception as exc:  # never let a landing page 500 on a bad backend
        return False, f"error checking readiness: {exc}"


def _backend_arms(name: str) -> list[str]:
    """The arm names a backend accepts, for the experiment form's arm
    dropdown. Empty for an unimplemented backend rather than raising -- the
    form still renders, the fieldset just says so."""
    try:
        module = backends_registry.get(name)
    except KeyError:
        return []
    facade = _facade_class(module)
    if facade is None:
        return []
    try:
        return list(getattr(facade, "ARMS", ()))
    except Exception:
        return []


def backends_view() -> list[dict]:
    """One entry per registered backend name, for both the landing page and
    /api/config -- the same list, so the UI's cards and the JSON contract
    never drift apart."""
    out = []
    for name in backends_registry.names():
        display = _BACKEND_DISPLAY.get(name, {"label": name, "description": ""})
        ok, reason = _backend_ready(name)
        try:
            implemented = _facade_class(backends_registry.get(name)) is not None
        except KeyError:
            implemented = False
        out.append({
            "name": name,
            "label": display["label"],
            "description": display["description"],
            "implemented": implemented,
            "ready": ok,
            "reason": reason,
            "arms": _backend_arms(name),
        })
    return out


def config_payload() -> dict:
    """The single dict both GET /api/config and the landing page template
    context are built from."""
    cwnd_ok, cwnd_reason = cwndmon.available()
    cap_ok, cap_reason = capture_mod.available()
    return {
        "backends": backends_view(),
        "congestion_algorithms": list(CONGESTION_ALGORITHMS),
        "cwnd": {
            "available": cwnd_ok,
            "reason": cwnd_reason,
            "interval_ms": cwndmon.interval_ms(),
        },
        "capture": {
            "available": cap_ok,
            "reason": cap_reason,
            "dir": str(capture_mod.pcap_dir()),
        },
    }


def register(app, templates: Jinja2Templates) -> None:
    """Registers the routes on *app* rather than decorating a module-level
    APIRouter for the template context -- the landing page needs a
    Jinja2Templates instance to render, and create_app() is the one place
    that constructs it (DESIGN.md 5: web UI FastAPI 통합 방침)."""

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(
            request, "index.html", {"config": config_payload()}
        )

    @app.get("/api/config")
    def api_config():
        return JSONResponse(config_payload())
