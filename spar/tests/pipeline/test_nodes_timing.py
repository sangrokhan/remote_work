# tests/pipeline/test_nodes_timing.py
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from spar.pipeline.nodes import Nodes
from spar.pipeline.state import SparState


def _make_nodes() -> Nodes:
    router = MagicMock()
    router.route = AsyncMock(return_value=MagicMock(route=MagicMock()))
    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=[0.9, 0.8])
    encoder = MagicMock()
    encoder.encode = MagicMock(return_value=np.zeros((1, 3)))
    milvus = MagicMock()
    return Nodes.create(
        router=router, reranker=reranker, encoder=encoder,
        milvus=milvus, acronyms_path=None,
    )


@pytest.mark.asyncio
async def test_preprocess_records_timing():
    nodes = _make_nodes()
    state: SparState = {"query": "test query", "node_timings": {}}
    result = await nodes.preprocess(state)
    assert "preprocess" in result["node_timings"]
    assert result["node_timings"]["preprocess"] >= 0.0


@pytest.mark.asyncio
async def test_timing_accumulates_across_nodes():
    nodes = _make_nodes()
    state: SparState = {"query": "test", "node_timings": {}}
    s1 = await nodes.preprocess(state)
    s2 = await nodes.prepare_context(s1)
    s3 = await nodes.generate(s2)
    assert "preprocess" in s3["node_timings"]
    assert "prepare_context" in s3["node_timings"]
    assert "generate" in s3["node_timings"]
    assert len(s3["node_timings"]) == 3


@pytest.mark.asyncio
async def test_structured_retrieve_no_rag_retrieve_timing_leak():
    router = MagicMock()
    router.route = AsyncMock(return_value=MagicMock(route=MagicMock()))
    reranker = MagicMock()
    encoder = MagicMock()
    encoder.encode = MagicMock(return_value=np.zeros((1, 3)))
    milvus = MagicMock()
    milvus.hybrid_search = MagicMock(return_value=[])
    nodes = Nodes.create(
        router=router, reranker=reranker, encoder=encoder,
        milvus=milvus, acronyms_path=None,
    )
    from spar.router.schemas import RouteResult, Route
    route_result = MagicMock(spec=RouteResult)
    route_result.route = Route.DEFAULT_RAG
    state: SparState = {
        "query": "test",
        "route_result": route_result,
        "node_timings": {},
        "top_k": 5,
    }
    result = await nodes.structured_retrieve(state)
    assert "structured_retrieve" in result["node_timings"]
    assert "rag_retrieve" not in result["node_timings"]
