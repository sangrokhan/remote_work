# tests/eval/test_metrics_faithfulness.py
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from spar.eval.metrics import compute_faithfulness, compute_suite_metrics


@pytest.mark.asyncio
async def test_compute_faithfulness_returns_float():
    llm = MagicMock()
    llm.chat = AsyncMock(return_value="0.85")
    chunks = [{"text": "maxHARQTx default is 5.", "score": 0.9}]
    score = await compute_faithfulness(
        answer="The default value is 5.",
        context_chunks=chunks,
        gold_answer="maxHARQTx default value is 5.",
        llm_client=llm,
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_compute_faithfulness_handles_malformed_response():
    llm = MagicMock()
    llm.chat = AsyncMock(return_value="I cannot determine the score.")
    score = await compute_faithfulness(
        answer="some answer",
        context_chunks=[{"text": "some context", "score": 0.5}],
        gold_answer="reference",
        llm_client=llm,
    )
    assert score == 0.0


def test_compute_suite_metrics_aggregates():
    results = [
        {
            "config_name": "baseline",
            "per_query": [
                {"recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0, "latency_ms": 100.0, "faithfulness": None},
                {"recall_at_5": 0.0, "recall_at_10": 1.0, "mrr": 0.5, "latency_ms": 120.0, "faithfulness": None},
            ],
        }
    ]
    table = compute_suite_metrics(results)
    assert len(table) == 1
    row = table[0]
    assert row["config"] == "baseline"
    assert row["recall_at_5"] == pytest.approx(0.5)
    assert row["recall_at_10"] == pytest.approx(1.0)
    assert row["mrr"] == pytest.approx(0.75)
    assert row["faithfulness"] is None


def test_compute_suite_metrics_faithfulness_average():
    results = [
        {
            "config_name": "+reranker",
            "per_query": [
                {"recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0, "latency_ms": 200.0, "faithfulness": 0.9},
                {"recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0, "latency_ms": 220.0, "faithfulness": 0.7},
            ],
        }
    ]
    table = compute_suite_metrics(results)
    assert table[0]["faithfulness"] == pytest.approx(0.8)
