from __future__ import annotations

import json
from typing import Generator

from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from python_backend.stategraph_workflow import build_workflow_graph, DemoState
from python_backend.graph_schema import serialize_stategraph_to_json

app = FastAPI(title="LangGraph Vis")

_workflow_graph = build_workflow_graph()
_base_dir = Path(__file__).resolve().parents[2]
_frontend_dir = _base_dir / "frontend"


@app.get("/run")
def run_workflow_stream() -> Response:
    initial_state: DemoState = {
        "llm_input": "demo",
        "planner_output": "",
        "executor_output": "",
        "refiner_output": "",
        "final_output": "",
        "hop_count": 0,
    }

    def stream_updates() -> Generator[str, None, None]:
        for event in _workflow_graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_state in event.items():
                payload = {
                    "node": node_name,
                    "state": node_state,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {\"status\":\"complete\"}\n\n"

    return StreamingResponse(
        stream_updates(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


app.mount("/static", NoCacheStaticFiles(directory=_frontend_dir), name="static")


@app.get("/")
def serve_index() -> Response:
    response = FileResponse(_frontend_dir / "index.html")
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/ui")
@app.get("/ui/")
def serve_index_alias() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=307)


@app.get("/api/graph")
def serve_graph_schema_alias() -> dict:
    return serialize_stategraph_to_json(_workflow_graph)


@app.get("/api/run")
def serve_run_alias() -> Response:
    return run_workflow_stream()


@app.get("/graph")
def serve_graph_schema() -> dict:
    return serialize_stategraph_to_json(_workflow_graph)
