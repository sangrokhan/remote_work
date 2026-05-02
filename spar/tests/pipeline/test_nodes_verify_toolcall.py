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
