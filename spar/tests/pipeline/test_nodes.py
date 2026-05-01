from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from spar.pipeline.nodes import Nodes
from spar.pipeline.state import SparState
from spar.router.schemas import Route, RouteResult


def _make_nodes(route: Route = Route.DEFAULT_RAG) -> Nodes:
    router = AsyncMock()
    router.route.return_value = RouteResult(route=route, confidence=0.8, layer="test")
    reranker = AsyncMock()
    reranker.rerank.return_value = [0.9, 0.7]
    encoder = MagicMock()
    import numpy as np
    encoder.encode.return_value = np.array([[0.1] * 1024])
    milvus = MagicMock()
    milvus.hybrid_search.return_value = [
        {"chunk_id": "stub-001", "score": 0.95, "text": "chunk for query"}
    ]
    return Nodes(
        router=router,
        reranker=reranker,
        encoder=encoder,
        milvus=milvus,
        _acronyms={},
        _reverse_index={},
        _keywords=set(),
    )


@pytest.fixture
def base_state() -> SparState:
    return {
        "query": "What is HO?",
        "product": "LTE",
        "release": "v6.0",
        "top_k": 5,
        "request_id": "test-id",
    }


@pytest.mark.unit
async def test_preprocess_expands_query(base_state: SparState) -> None:
    nodes = _make_nodes()
    result = await nodes.preprocess(base_state)
    assert "expanded_query" in result
    assert result["node_trace"] == ["preprocess"]


@pytest.mark.unit
async def test_preprocess_empty_acronyms_passthrough(base_state: SparState) -> None:
    nodes = _make_nodes()
    result = await nodes.preprocess(base_state)
    assert result["expanded_query"] == base_state["query"]


@pytest.mark.unit
async def test_route_uses_expanded_query(base_state: SparState) -> None:
    nodes = _make_nodes(Route.DEFAULT_RAG)
    state = {**base_state, "expanded_query": "What is Handover?"}
    result = await nodes.route(state)
    nodes.router.route.assert_called_once_with("What is Handover?")
    assert result["route_result"].route == Route.DEFAULT_RAG
    assert "route" in result["node_trace"]


@pytest.mark.unit
async def test_route_falls_back_to_query(base_state: SparState) -> None:
    nodes = _make_nodes()
    result = await nodes.route(base_state)
    nodes.router.route.assert_called_once_with(base_state["query"])


@pytest.mark.unit
async def test_rag_retrieve_returns_chunks(base_state: SparState) -> None:
    from spar.router.schemas import RouteResult
    nodes = _make_nodes()
    state = {
        **base_state,
        "expanded_query": "What is HO?",
        "route_result": RouteResult(route=Route.DEFAULT_RAG, confidence=0.8, layer="test"),
    }
    result = await nodes.rag_retrieve(state)
    assert isinstance(result["raw_chunks"], list)
    assert len(result["raw_chunks"]) > 0
    assert "rag_retrieve" in result["node_trace"]


@pytest.mark.unit
async def test_rerank_orders_by_score(base_state: SparState) -> None:
    nodes = _make_nodes()
    nodes.reranker.rerank.return_value = [0.3, 0.9]
    state = {
        **base_state,
        "expanded_query": "What is HO?",
        "raw_chunks": [
            {"chunk_id": "c1", "score": 0.8, "text": "low relevance"},
            {"chunk_id": "c2", "score": 0.6, "text": "high relevance"},
        ],
    }
    result = await nodes.rerank(state)
    assert result["reranked_chunks"][0]["chunk_id"] == "c2"
    assert "rerank" in result["node_trace"]


@pytest.mark.unit
async def test_rerank_empty_chunks(base_state: SparState) -> None:
    nodes = _make_nodes()
    state = {**base_state, "raw_chunks": []}
    result = await nodes.rerank(state)
    assert result["reranked_chunks"] == []
    nodes.reranker.rerank.assert_not_called()


@pytest.mark.unit
async def test_generate_uses_reranked_chunks(base_state: SparState) -> None:
    nodes = _make_nodes()
    state = {
        **base_state,
        "reranked_chunks": [{"chunk_id": "c1", "text": "text"}],
        "raw_chunks": [{"chunk_id": "c2", "text": "fallback"}],
    }
    result = await nodes.generate(state)
    assert "context=1 chunks" in result["answer"]
    assert "generate" in result["node_trace"]


@pytest.mark.unit
async def test_generate_falls_back_to_raw_chunks(base_state: SparState) -> None:
    nodes = _make_nodes()
    state = {**base_state, "raw_chunks": [{"chunk_id": "c1", "text": "t"}, {"chunk_id": "c2", "text": "t"}]}
    result = await nodes.generate(state)
    assert "context=2 chunks" in result["answer"]


@pytest.mark.unit
async def test_node_trace_accumulates(base_state: SparState) -> None:
    nodes = _make_nodes()
    state = {**base_state, "node_trace": ["preprocess"]}
    state = {**state, "expanded_query": "x"}
    result = await nodes.route(state)
    assert result["node_trace"] == ["preprocess", "route"]


@pytest.mark.unit
async def test_preprocess_populates_matched_terms(base_state: SparState) -> None:
    router = AsyncMock()
    router.route.return_value = RouteResult(route=Route.DEFAULT_RAG, confidence=0.8, layer="test")
    nodes = Nodes(
        router=router,
        reranker=MagicMock(),
        encoder=MagicMock(),
        milvus=MagicMock(),
        _acronyms={},
        _reverse_index={},
        _keywords={"NRCellDU", "maxRetransmissions"},
    )
    state: SparState = {**base_state, "query": "What is NRCellDU config?"}
    result = await nodes.preprocess(state)
    assert "NRCellDU" in result["matched_terms"]
    assert "maxRetransmissions" not in result["matched_terms"]


@pytest.mark.unit
async def test_preprocess_empty_matched_terms_no_keywords(base_state: SparState) -> None:
    nodes = _make_nodes()
    state: SparState = {**base_state, "query": "NRCellDU question"}
    result = await nodes.preprocess(state)
    assert result["matched_terms"] == []
