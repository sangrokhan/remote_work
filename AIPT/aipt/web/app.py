"""aipt.web -- the single FastAPI app (DESIGN.md 5, "웹 UI FastAPI 통합
방침"; 4.5's backend-selection architecture rather than the earlier
external-api/synthetic-mock URL-namespace split).

``create_app()`` builds one FastAPI instance, mounts ``aipt/web/static`` at
``/static``, wires ``aipt/web/templates`` through Jinja2Templates, and
includes the three route modules:

  * ``routes_config`` -- ``GET /`` (landing page, backend-selection cards)
    and ``GET /api/config``.
  * ``routes_run``    -- ``POST /api/run`` (blocking, whole-run response)
    and ``POST /api/run/stream`` (SSE, one event per turn as it finishes).
  * ``routes_runs``   -- ``GET/DELETE /api/runs*``, the CSV/bundle/pcap
    download endpoints.

Run store persists to disk now (``aipt/web/store.py``, ``RUN_STORE_DIR``,
default ``data/runs/``) -- a restart rehydrates recent runs instead of
losing them. ``/api/run/stream`` (SSE, per-turn progress) now exists
alongside the original blocking ``/api/run`` -- see ``routes_run.py``'s
module docstring for how the two share one generator.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from aipt.web import routes_config, routes_run, routes_runs

_HERE = Path(__file__).resolve().parent
TEMPLATES_DIR = _HERE / "templates"
STATIC_DIR = _HERE / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="AIPT", description="AI Protocol Traffic lab -- backend-selection web UI")

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    routes_config.register(app, templates)
    app.include_router(routes_run.router)
    app.include_router(routes_runs.router)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


# uvicorn aipt.web.app:app  (or --factory aipt.web.app:create_app)
app = create_app()
