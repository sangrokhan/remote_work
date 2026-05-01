from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from spar.pipeline.graph import build_graph
from spar.pipeline.state import SparState
from spar.router.schemas import Route, RouteResult


import numpy as np


def _make_router(route: Route = Route.DEFAULT_RAG) -> AsyncMock:
    router = AsyncMock()
    router.route.return_value = RouteResult(route=route, confidence=0.8, layer="test")
    return router


def _make_reranker(scores: list[float] | None = None) -> AsyncMock:
    reranker = AsyncMock()
    reranker.rerank.return_value = scores or [0.9]
    return reranker


def _make_encoder() -> MagicMock:
    encoder = MagicMock()
    encoder.encode.return_value = np.array([[0.1] * 1024])
    return encoder


def _make_milvus() -> MagicMock:
    milvus = MagicMock()
    milvus.hybrid_search.return_value = [
        {"chunk_id": "c1", "score": 0.9, "text": "chunk text"}
    ]
    return milvus


@pytest.fixture
def base_state() -> SparState:
    return {
        "query": "What is TTT parameter?",
        "product": "LTE",
        "release": "v6.0",
        "top_k": 5,
        "request_id": "test-id",
    }


@pytest.mark.unit
async def test_default_rag_path(base_state: SparState) -> None:
    graph = build_graph(router=_make_router(Route.DEFAULT_RAG), reranker=_make_reranker(), encoder=_make_encoder(), milvus=_make_milvus())
    result = await graph.ainvoke(base_state)
    assert "answer" in result
    assert "rag_retrieve" in result["node_trace"]
    assert "rerank" in result["node_trace"]
    assert "generate" in result["node_trace"]


@pytest.mark.unit
async def test_structured_lookup_path(base_state: SparState) -> None:
    graph = build_graph(router=_make_router(Route.STRUCTURED_LOOKUP), reranker=_make_reranker(), encoder=_make_encoder(), milvus=_make_milvus())
    result = await graph.ainvoke(base_state)
    # stub falls back to rag_retrieve until Phase 3 KG/DB is implemented
    assert "structured_retrieve" in result["node_trace"]


@pytest.mark.unit
async def test_diagnostic_path(base_state: SparState) -> None:
    graph = build_graph(router=_make_router(Route.DIAGNOSTIC), reranker=_make_reranker(), encoder=_make_encoder(), milvus=_make_milvus())
    result = await graph.ainvoke(base_state)
    # stub falls back to rag_retrieve until Phase 5 iterative retrieval is implemented
    assert "multi_hop_retrieve" in result["node_trace"]


@pytest.mark.parametrize("route", [Route.DEFINITION_EXPLAIN, Route.PROCEDURAL, Route.COMPARATIVE])
@pytest.mark.unit
async def test_remaining_routes_use_rag(route: Route, base_state: SparState) -> None:
    graph = build_graph(router=_make_router(route), reranker=_make_reranker(), encoder=_make_encoder(), milvus=_make_milvus())
    result = await graph.ainvoke(base_state)
    assert "rag_retrieve" in result["node_trace"]


@pytest.mark.unit
async def test_full_trace_default_rag(base_state: SparState) -> None:
    graph = build_graph(router=_make_router(Route.DEFAULT_RAG), reranker=_make_reranker(), encoder=_make_encoder(), milvus=_make_milvus())
    result = await graph.ainvoke(base_state)
    assert result["node_trace"][0] == "preprocess"
    assert result["node_trace"][1].startswith("rewrite_query:")
    assert result["node_trace"][2:] == [
        "prepare_context",
        "route",
        "rag_retrieve",
        "rerank",
        "generate",
    ]


@pytest.mark.unit
async def test_state_has_required_fields(base_state: SparState) -> None:
    graph = build_graph(router=_make_router(), reranker=_make_reranker(), encoder=_make_encoder(), milvus=_make_milvus())
    result = await graph.ainvoke(base_state)
    assert "route_result" in result
    assert "answer" in result
    assert "node_trace" in result
