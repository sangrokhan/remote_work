from __future__ import annotations

import json
from typing import Generator

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from python_backend.stategraph_workflow import build_workflow_graph, DemoState

app = FastAPI(title="LangGraph Vis")


@app.get("/run")
def run_workflow_stream() -> Response:
    initial_state: DemoState = {
        "llm_input": "demo",
        "planner_output": "",
        "executor_output": "",
        "refiner_output": "",
        "final_output": "",
    }

    graph = build_workflow_graph()

    def stream_updates() -> Generator[str, None, None]:
        for event in graph.stream(initial_state, stream_mode="updates"):
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


frontend_dir = "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
def serve_index() -> Response:
    return FileResponse(f"{frontend_dir}/index.html")
