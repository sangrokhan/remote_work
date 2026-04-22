"""
FastAPI entry point.

Endpoints
---------
GET  /health       liveness probe
GET  /graph        LangGraph schema {nodes, edges} for Cytoscape visualization
POST /api/run      SSE stream — routes to simple_flow (agentic_rag=false)
                   or agentic_rag_flow (agentic_rag=true)
"""
from __future__ import annotations

import json
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from app.models import RunWorkflowRequest
from graph_schema import serialize_stategraph_to_json
from langgraph_flow.agents.graph import create_agentic_rag_graph
from services.simple_flow import run_simple_flow
from services.agentic_rag_flow import run_agentic_rag_flow

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

_workflow_graph = create_agentic_rag_graph(False)._graph


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


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
            flow = run_agentic_rag_flow(req) if req.agentic_rag else run_simple_flow(req)
            async for event in flow:
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
