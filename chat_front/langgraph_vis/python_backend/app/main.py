from __future__ import annotations

import json
from typing import AsyncGenerator, Mapping

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
    workflow_nodes = {"planner", "executor", "refiner", "synthesizer", "__start__", "__end__"}

    def _extract_node_name(event: Mapping[str, object]) -> str | None:
        candidates = ("name", "node", "node_name")
        for key in candidates:
            value = event.get(key)
            if isinstance(value, str):
                node_name = value.strip()
                if node_name:
                    return node_name

        metadata = event.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("langgraph_node", "langgraph_node_name", "name"):
                value = metadata.get(key)
                if isinstance(value, str):
                    node_name = value.strip()
                    if node_name:
                        return node_name

        return None

    def _extract_node_state(event_data: object) -> object | None:
        if not isinstance(event_data, Mapping):
            return None
        output = event_data.get("output")
        if isinstance(output, Mapping):
            return output
        for key in ("state", "input", "result"):
            value = event_data.get(key)
            if value is not None:
                return value
        return None

    async def stream_updates() -> AsyncGenerator[str, None]:
        has_emitted_events = False

        def _yield_payload(payload: dict) -> str:
            nonlocal has_emitted_events
            has_emitted_events = True
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        try:
            if hasattr(_workflow_graph, "astream_events"):
                async for event in _workflow_graph.astream_events(initial_state, version="v2"):
                    if not isinstance(event, Mapping):
                        continue
                    event_name = str(event.get("event", "")).strip()
                    raw_node = _extract_node_name(event)
                    if not raw_node:
                        continue
                    node_name = raw_node
                    if node_name not in workflow_nodes:
                        continue
                    if event_name in {"on_node_start", "node_start", "on_chain_start"}:
                        payload = {
                            "node": node_name,
                            "state": None,
                            "stage": "start",
                        }
                        yield _yield_payload(payload)
                    elif event_name in {"on_node_end", "node_end", "on_chain_end"}:
                        node_state = _extract_node_state(event.get("data"))
                        payload = {
                            "node": node_name,
                            "state": node_state,
                            "stage": "end",
                        }
                        yield _yield_payload(payload)
            if not has_emitted_events:
                raise RuntimeError("No node events from astream_events")
        except Exception:
            for event in _workflow_graph.stream(initial_state, stream_mode="updates"):
                for node_name, node_state in event.items():
                    payload = {
                        "node": node_name,
                        "state": None,
                        "stage": "start",
                    }
                    yield _yield_payload(payload)
                    payload = {
                        "node": node_name,
                        "state": node_state,
                        "stage": "end",
                    }
                    yield _yield_payload(payload)
            else:
                raise AttributeError("astream_events not available")
        except Exception:
            for event in _workflow_graph.stream(initial_state, stream_mode="updates"):
                for node_name, node_state in event.items():
                    payload = {
                        "node": node_name,
                        "state": None,
                        "stage": "start",
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    payload = {
                        "node": node_name,
                        "state": node_state,
                        "stage": "end",
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
