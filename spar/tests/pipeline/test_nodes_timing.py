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
    s2 = await nodes.generate(s1)
    assert "preprocess" in s2["node_timings"]
    assert "generate" in s2["node_timings"]
