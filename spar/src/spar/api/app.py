"""SPAR API — query entry point.

Start:
    uvicorn spar.api.app:app --reload --port 9000
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from spar.router.hybrid_router import HybridRouter

app = FastAPI(title="SPAR", version="0.1.0")
_router = HybridRouter(use_llm=False)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query")
    product: str | None = Field(None, description="LTE | NR | both")
    release: str | None = Field(None, description="e.g. v6.0")
    top_k: int = Field(10, ge=1, le=50)


class QueryResponse(BaseModel):
    request_id: str
    query: str
    route: str
    answer: str
    sources: list[dict[str, Any]]
    latency_ms: float


# ---------------------------------------------------------------------------
# Pipeline stubs (비동기 — 실제 구현으로 교체)
# ---------------------------------------------------------------------------

async def retrieve(query: str, product: str | None, release: str | None, top_k: int) -> list[dict]:
    """Stub: vector + BM25 hybrid retrieval."""
    await asyncio.sleep(0)  # replace with actual retrieval
    return [
        {"chunk_id": "stub-001", "score": 0.95, "text": f"[stub] relevant chunk for: {query}"}
    ]


async def generate(query: str, chunks: list[dict]) -> str:
    """Stub: LLM generation over retrieved chunks."""
    await asyncio.sleep(0)  # replace with vLLM call
    return f"[stub] Answer for '{query}' based on {len(chunks)} chunks."


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    t0 = time.monotonic()
    request_id = str(uuid.uuid4())

    route_result = await _router.route(req.query)

    chunks = await retrieve(req.query, req.product, req.release, req.top_k)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    answer = await generate(req.query, chunks)

    return QueryResponse(
        request_id=request_id,
        query=req.query,
        route=route_result.route.value,
        answer=answer,
        sources=chunks,
        latency_ms=round((time.monotonic() - t0) * 1000, 1),
    )
