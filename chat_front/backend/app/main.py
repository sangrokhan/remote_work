from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from stategraph_workflow import build_workflow_graph, run_demo_workflow_events
from graph_schema import serialize_stategraph_to_json

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


@app.websocket("/ws/connect")
async def websocket_wait(websocket: WebSocket) -> None:
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
            run_id = None
            if isinstance(payload, dict):
                run_id = _safe_text(payload.get("run_id"))
                if not run_id:
                    run_id = str(uuid.uuid4())

            if incoming in {"graph", "refresh", "get_graph"} or incoming_type == "get_graph":
                await websocket.send_json(
                    {"type": "graph", "payload": serialize_stategraph_to_json(_workflow_graph)}
                )
                continue

            if incoming_type == "run_workflow":
                user_input = _safe_text(payload.get("input") if isinstance(payload, dict) else None, '')
                if not user_input:
                    await websocket.send_json(
                        {
                            "type": "workflow_error",
                            "run_id": run_id,
                            "message": "워크플로우 입력값이 없습니다.",
                        }
                    )
                    continue

                await websocket.send_json(
                    {"type": "workflow_started", "run_id": run_id, "message": "워크플로우 실행됨"}
                )

                try:
                    for event in run_demo_workflow_events(user_input):
                        payload = {
                            "type": "workflow_event",
                            "run_id": run_id,
                            **event,
                        }
                        await websocket.send_json(payload)
                        await asyncio.sleep(0)
                    await websocket.send_json(
                        {"type": "workflow_complete", "run_id": run_id, "message": "워크플로우 완료"}
                    )
                except Exception as error:
                    await websocket.send_json(
                        {
                            "type": "workflow_error",
                            "run_id": run_id,
                            "message": f"워크플로우 실행 중 오류: {error}",
                        }
                    )
                continue

            if incoming == "close":
                await websocket.close(code=1000)
                return
            await websocket.send_text(f"echo: {incoming}")
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        return


def _parse_json_or_string(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
