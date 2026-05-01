# tests/eval/test_eval_suite.py
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from spar.eval.eval_suite import run_suite, print_comparison_table
from spar.pipeline.config import GraphConfig

GOLDSET = [
    {
        "query_id": "q001",
        "query": "What is maxHARQTx default?",
        "type": "parameter_lookup",
        "source_doc": "param_ref.md",
        "section": "4.1",
        "gold_answer": "The default is 5.",
    }
]

FAKE_STATE = {
    "query": "What is maxHARQTx default?",
    "raw_chunks": [
        {"text": "maxHARQTx default is 5.", "score": 0.9, "source_doc": "param_ref.md", "section_num": "4.1"}
    ],
    "reranked_chunks": [],
    "answer": "[stub]",
    "node_timings": {"route": 10.0, "rag_retrieve": 80.0, "generate": 5.0},
    "node_trace": ["route", "rag_retrieve", "generate"],
}


@pytest.mark.asyncio
async def test_run_suite_returns_one_result_per_config():
    configs = [GraphConfig(name="baseline"), GraphConfig(name="+reranker", use_reranker=True)]

    with patch("spar.eval.eval_suite.build_graph") as mock_build:
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=FAKE_STATE)
        mock_build.return_value = mock_graph

        results = await run_suite(
            configs=configs, goldset=GOLDSET,
            router=MagicMock(), reranker=MagicMock(),
            encoder=MagicMock(encode=MagicMock(return_value=np.zeros((1,3)))),
            milvus=MagicMock(), top_k=10,
        )

    assert len(results) == 2
    assert results[0]["config_name"] == "baseline"
    assert len(results[0]["per_query"]) == 1


@pytest.mark.asyncio
async def test_run_suite_per_query_has_recall_and_latency():
    configs = [GraphConfig(name="baseline")]

    with patch("spar.eval.eval_suite.build_graph") as mock_build:
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=FAKE_STATE)
        mock_build.return_value = mock_graph

        results = await run_suite(
            configs=configs, goldset=GOLDSET,
            router=MagicMock(), reranker=MagicMock(),
            encoder=MagicMock(encode=MagicMock(return_value=np.zeros((1,3)))),
            milvus=MagicMock(), top_k=10,
        )

    pq = results[0]["per_query"][0]
    assert "recall_at_5" in pq
    assert "recall_at_10" in pq
    assert "mrr" in pq
    assert "latency_ms" in pq
    assert pq["latency_ms"] == pytest.approx(95.0)  # 10+80+5
