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
    assert "rag_retrieve" not in result.get("node_trace", [])


@pytest.mark.asyncio
async def test_timing_accumulates_through_retrieval():
    from spar.router.schemas import RouteResult, Route
    router = MagicMock()
    router.route = AsyncMock(return_value=MagicMock(route=MagicMock()))
    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=[0.9])
    encoder = MagicMock()
    encoder.encode = MagicMock(return_value=np.zeros((1, 3)))
    milvus = MagicMock()
    milvus.hybrid_search = MagicMock(
        return_value=[{"text": "ctx", "score": 0.9, "source_doc": "doc.md", "section_num": "1.1"}]
    )

    nodes = Nodes.create(
        router=router, reranker=reranker, encoder=encoder, milvus=milvus, acronyms_path=None
    )

    route_result = MagicMock(spec=RouteResult)
    route_result.route = Route.DEFAULT_RAG
    state: SparState = {
        "query": "test", "route_result": route_result, "top_k": 5, "node_timings": {},
    }
    s1 = await nodes.rag_retrieve(state)
    s2 = await nodes.rerank(s1)
    s3 = await nodes.generate(s2)
    assert "rag_retrieve" in s3["node_timings"]
    assert "rerank" in s3["node_timings"]
    assert "generate" in s3["node_timings"]


from spar.llm.client import LLMClient


@pytest.mark.asyncio
async def test_generate_stub_when_no_llm():
    nodes = _make_nodes()
    state: SparState = {
        "query": "What is maxHARQTx?",
        "raw_chunks": [{"text": "maxHARQTx default is 5.", "score": 0.9, "source_doc": "ref.md", "section_num": "4.1"}],
        "node_timings": {},
    }
    result = await nodes.generate(state)
    assert result["answer"].startswith("[stub]")
    assert "generate" in result["node_timings"]


@pytest.mark.asyncio
async def test_generate_calls_llm_when_provided():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat = AsyncMock(return_value="maxHARQTx default value is 5.")

    nodes = Nodes.create(
        router=MagicMock(), reranker=MagicMock(),
        encoder=MagicMock(encode=MagicMock(return_value=np.zeros((1, 3)))),
        milvus=MagicMock(), acronyms_path=None, llm=mock_llm,
    )
    state: SparState = {
        "query": "What is maxHARQTx?",
        "raw_chunks": [{"text": "maxHARQTx default is 5.", "score": 0.9, "source_doc": "ref.md", "section_num": "4.1"}],
        "node_timings": {},
    }
    result = await nodes.generate(state)
    assert result["answer"] == "maxHARQTx default value is 5."
    mock_llm.chat.assert_called_once()
