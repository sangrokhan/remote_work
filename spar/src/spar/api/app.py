"""SPAR API — query entry point.

Start:
    uvicorn spar.api.app:app --reload --port 9000
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, field_validator

from spar.encoder.registry import get_encoder
from spar.pipeline.graph import build_graph
from spar.pipeline.state import SparState
from spar.reranker.registry import get_reranker
from spar.retrieval.milvus_client import MilvusConfig, SparMilvusClient
from spar.router.hybrid_router import HybridRouter

_graph: CompiledStateGraph | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    encoder = await get_encoder()
    reranker = await get_reranker()
    router = HybridRouter(encoder=encoder, use_llm=False)
    milvus_client = SparMilvusClient(MilvusConfig())
    try:
        _graph = build_graph(router=router, reranker=reranker, encoder=encoder, milvus=milvus_client)
        yield
    finally:
        milvus_client.close()
        _graph = None


app = FastAPI(title="SPAR", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ConversationMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query")
    product: str | None = Field(None, description="LTE | NR | both")
    release: str | None = Field(None, description="e.g. v6.0")
    top_k: int = Field(10, ge=1, le=50)
    history: list[ConversationMessage] = Field(default_factory=list, description="Recent conversation turns (max 5)")

    @field_validator("history")
    @classmethod
    def _trim_history(cls, v: list[ConversationMessage]) -> list[ConversationMessage]:
        return v[-10:]  # max 5 turns × 2 messages


class QueryResponse(BaseModel):
    request_id: str
    query: str
    route: str
    answer: str
    sources: list[dict[str, Any]]
    latency_ms: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    if _graph is None:
        raise HTTPException(status_code=503, detail="Service not ready.")

    t0 = time.monotonic()
    initial_state: SparState = {
        "query": req.query,
        "product": req.product,
        "release": req.release,
        "top_k": req.top_k,
        "request_id": str(uuid.uuid4()),
        "history": [m.model_dump() for m in req.history],
    }
    final_state: SparState = await _graph.ainvoke(initial_state)

    chunks = final_state.get("reranked_chunks") or final_state.get("raw_chunks", [])
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    return QueryResponse(
        request_id=final_state["request_id"],
        query=final_state["query"],
        route=final_state["route_result"].route.value,
        answer=final_state["answer"],
        sources=chunks,
        latency_ms=round((time.monotonic() - t0) * 1000, 1),
    )
