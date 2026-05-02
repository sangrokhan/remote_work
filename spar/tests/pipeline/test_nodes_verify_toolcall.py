from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from spar.pipeline.nodes import Nodes
from spar.pipeline.state import SparState


def _make_nodes(llm_response: str) -> Nodes:
    router = MagicMock()
    router.route = AsyncMock(return_value=MagicMock(route=MagicMock(), needs_decomposition=False))
    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=[0.9])
    encoder = MagicMock()
    encoder.encode = MagicMock(return_value=np.zeros((2, 3)))
    milvus = MagicMock()
    milvus.hybrid_search = MagicMock(return_value=[])
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=llm_response)
    return Nodes.create(
        router=router,
        reranker=reranker,
        encoder=encoder,
        milvus=milvus,
        llm=llm,
    )


@pytest.mark.asyncio
async def test_verify_sufficient_sets_score_gte_3():
    nodes = _make_nodes(json.dumps({"score": 4, "reason": "answer is complete"}))
    state: SparState = {
        "query": "what is maxUE?",
        "answer": "maxUE controls the maximum number of UEs.",
        "reranked_chunks": [{"text": "maxUE is a parameter", "score": 0.9}],
        "retry_count": 0,
        "tried_strategies": ["rag"],
    }
    result = await nodes.verify(state)
    assert result["verify_score"] == 4
    assert result["verify_reason"] == "answer is complete"
    assert "verify" in result["node_trace"]


@pytest.mark.asyncio
async def test_verify_insufficient_sets_score_lt_3():
    nodes = _make_nodes(json.dumps({"score": 1, "reason": "missing parameter range"}))
    state: SparState = {
        "query": "what is the range of maxUE?",
        "answer": "I don't have enough information.",
        "reranked_chunks": [],
        "retry_count": 0,
        "tried_strategies": ["rag"],
    }
    result = await nodes.verify(state)
    assert result["verify_score"] == 1
    assert result["verify_reason"] == "missing parameter range"


@pytest.mark.asyncio
async def test_verify_malformed_json_defaults_sufficient():
    nodes = _make_nodes("not valid json at all")
    state: SparState = {
        "query": "test",
        "answer": "answer",
        "reranked_chunks": [],
        "retry_count": 0,
        "tried_strategies": [],
    }
    result = await nodes.verify(state)
    assert result["verify_score"] == 5.0


@pytest.mark.asyncio
async def test_tool_call_picks_next_untried_strategy():
    nodes = _make_nodes("what is maxUE default value")
    state: SparState = {
        "query": "what is maxUE?",
        "tried_strategies": ["rag"],
        "retry_count": 0,
        "raw_chunks": [],
        "verify_reason": "missing default value info",
        "route_result": MagicMock(needs_decomposition=False, route=MagicMock()),
        "top_k": 5,
        "matched_terms": [],
    }
    result = await nodes.tool_call(state)
    assert "decomposed" in result["tried_strategies"]
    assert result["retry_count"] == 1
    assert result["improved_query"] == "what is maxUE default value"
    assert "tool_call" in result["node_trace"]


@pytest.mark.asyncio
async def test_tool_call_skips_already_tried():
    nodes = _make_nodes("retry query")
    state: SparState = {
        "query": "test",
        "tried_strategies": ["rag", "decomposed", "multi_hop"],
        "retry_count": 2,
        "raw_chunks": [],
        "verify_reason": "still missing info",
        "route_result": MagicMock(needs_decomposition=False, route=MagicMock()),
        "top_k": 5,
        "matched_terms": [],
    }
    result = await nodes.tool_call(state)
    assert "structured" in result["tried_strategies"]
    assert result["retry_count"] == 3


@pytest.mark.asyncio
async def test_tool_call_no_untried_strategy_returns_state_unchanged():
    nodes = _make_nodes("query")
    state: SparState = {
        "query": "test",
        "tried_strategies": ["rag", "decomposed", "multi_hop", "structured"],
        "retry_count": 3,
        "raw_chunks": [],
        "verify_reason": "exhausted",
        "route_result": MagicMock(needs_decomposition=False, route=MagicMock()),
        "top_k": 5,
        "matched_terms": [],
    }
    result = await nodes.tool_call(state)
    assert result["retry_count"] == 3
    assert "tool_call" in result["node_trace"]
