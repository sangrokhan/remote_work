"""FastAPI application factory for Python backend."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .schema_api import create_workflow_schema_router
from .run_state_api import create_run_state_router
from .history_api import create_run_history_router
from .run_state_store import RunStateStore


def create_app(*, run_store: RunStateStore | None = None):
    store = run_store or RunStateStore()
    app = FastAPI(title="langgraph_vis Python API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    app.include_router(create_workflow_schema_router())
    app.include_router(create_run_state_router(store=store))
    app.include_router(create_run_history_router(store=store))

    frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="ui")
    return app


app = create_app()
