from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from spar.pipeline.nodes import Nodes
from spar.pipeline.state import SparState
from spar.router.schemas import Route, RouteResult


def _make_nodes():
    router = MagicMock()
    reranker = MagicMock()
    encoder = MagicMock()
    encoder.encode.return_value = np.array([[0.1] * 1024])
    milvus = MagicMock()
    milvus.hybrid_search.return_value = [
        {"chunk_id": "c1", "score": 0.9, "text": "real chunk"}
    ]
    return Nodes.create(
        router=router,
        reranker=reranker,
        encoder=encoder,
        milvus=milvus,
    ), encoder, milvus


def _state(route=Route.DEFAULT_RAG, entities=None, product=None, release=None):
    return SparState(
        query="test query",
        expanded_query="test query expanded",
        route_result=RouteResult(
            route=route, confidence=0.9, layer="test",
            entities=entities or {}, product=product, release=release,
        ),
    )


@pytest.mark.asyncio
async def test_rag_retrieve_calls_hybrid_search():
    nodes, encoder, milvus = _make_nodes()
    state = _state()
    result = await nodes.rag_retrieve(state)

    encoder.encode.assert_called_once()
    assert milvus.hybrid_search.called
    assert len(result["raw_chunks"]) > 0
    assert result["raw_chunks"][0]["text"] == "real chunk"


@pytest.mark.asyncio
async def test_rag_retrieve_not_stub():
    nodes, _, _ = _make_nodes()
    state = _state()
    result = await nodes.rag_retrieve(state)
    assert "[stub]" not in result["raw_chunks"][0]["text"]


@pytest.mark.asyncio
async def test_structured_retrieve_alarm_targets_alarm_ref():
    nodes, encoder, milvus = _make_nodes()
    state = _state(
        route=Route.STRUCTURED_LOOKUP,
        entities={"alarm_code": "ALM-1234"},
    )
    await nodes.structured_retrieve(state)
    # hybrid_search should be called with alarm_ref doc_type
    call_args_list = milvus.hybrid_search.call_args_list
    assert any(
        call.kwargs.get("doc_type") == "alarm_ref" or
        (call.args and call.args[0] == "alarm_ref")
        for call in call_args_list
    )


@pytest.mark.asyncio
async def test_rag_retrieve_passes_product_filter():
    nodes, encoder, milvus = _make_nodes()
    state = _state(product="NR")
    await nodes.rag_retrieve(state)
    for call in milvus.hybrid_search.call_args_list:
        assert call.kwargs.get("expr") == 'product == "NR"'
