from __future__ import annotations

import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.models import RunWorkflowRequest
from stategraph_workflow import build_workflow_graph, run_demo_workflow_events_async
from graph_schema import serialize_stategraph_to_json

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("workflow_api")

app = FastAPI(title="LangGraph Vis")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_workflow_graph = build_workflow_graph()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/graph")
def serve_graph_schema_alias() -> dict:
    return serialize_stategraph_to_json(_workflow_graph)


@app.get("/graph")
def serve_graph_schema() -> dict:
    return serialize_stategraph_to_json(_workflow_graph)


@app.post("/api/run")
async def run_workflow_sse(req: RunWorkflowRequest) -> StreamingResponse:
    logger.debug(
        "POST /api/run: model=%s agentic_rag=%s response_mode=%s max_tokens=%s input_len=%d",
        req.model, req.agentic_rag, req.response_mode, req.max_tokens, len(req.input),
    )

    async def event_gen():
        init = {
            "event": "run_started",
            "run_id": req.run_id,
            "model": req.model,
            "agentic_rag": req.agentic_rag,
            "response_mode": req.response_mode,
            "max_tokens": req.max_tokens,
        }
        yield f"event: run_started\ndata: {json.dumps(init, ensure_ascii=False)}\n\n"
        try:
            async for event in run_demo_workflow_events_async(req):
                event_type = event.get("event", "workflow_event")
                yield f"event: {event_type}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            err = {"event": "workflow_error", "message": str(exc)}
            yield f"event: workflow_error\ndata: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _safe_text(value: object, fallback: str = '') -> str:
    if value is None:
        return fallback
    if isinstance(value, str):
        return value
    return str(value)


def _safe_json_message(data: object) -> dict:
    if isinstance(data, dict):
        return data
    return {}


def _parse_json_or_string(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


@app.websocket("/ws/connect")
async def websocket_graph(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "connected", "status": "ready"})

    try:
        while True:
            incoming = await websocket.receive_text()
            if incoming == "ping":
                await websocket.send_text("pong")
                continue

            payload = _safe_json_message(_safe_text(incoming).strip() and _parse_json_or_string(incoming))
            incoming_type = payload.get("type") if isinstance(payload, dict) else None

            if incoming in {"graph", "refresh", "get_graph"} or incoming_type == "get_graph":
                await websocket.send_json(
                    {"type": "graph", "payload": serialize_stategraph_to_json(_workflow_graph)}
                )
                continue

            if incoming == "close":
                await websocket.close(code=1000)
                return

            await websocket.send_text(f"echo: {incoming}")
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        return
