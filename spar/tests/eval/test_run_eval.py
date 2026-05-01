# tests/eval/test_run_eval.py
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from spar.eval.run_eval import _collect_results_via_graph, _save_output

GOLDSET = [
    {
        "query_id": "q001",
        "query": "What is maxHARQTx default?",
        "type": "parameter_lookup",
        "source_doc": "param_ref.md",
        "section": "4.1",
    }
]

FAKE_STATE = {
    "query": "What is maxHARQTx default?",
    "raw_chunks": [
        {"text": "maxHARQTx default is 5.", "score": 0.9, "source_doc": "param_ref.md", "section_num": "4.1"}
    ],
    "reranked_chunks": [
        {"text": "maxHARQTx default is 5.", "score": 0.9, "source_doc": "param_ref.md", "section_num": "4.1"}
    ],
    "node_timings": {"route": 10.0, "rag_retrieve": 80.0},
    "node_trace": ["route", "rag_retrieve"],
}


@pytest.mark.asyncio
async def test_collect_results_via_graph_returns_gold_and_retrieved():
    with patch("spar.eval.run_eval.build_graph") as mock_build:
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=FAKE_STATE)
        mock_build.return_value = mock_graph

        results = await _collect_results_via_graph(
            goldset=GOLDSET, doc_type="spec", top_k=10,
            router=MagicMock(), reranker=MagicMock(),
            encoder=MagicMock(encode=MagicMock(return_value=np.zeros((1, 3)))),
            milvus=MagicMock(),
        )

    assert len(results) == 1
    assert results[0]["gold"] == GOLDSET[0]
    assert results[0]["retrieved"] == FAKE_STATE["reranked_chunks"]


@pytest.mark.asyncio
async def test_collect_results_skips_on_error():
    with patch("spar.eval.run_eval.build_graph") as mock_build:
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("graph error"))
        mock_build.return_value = mock_graph

        results = await _collect_results_via_graph(
            goldset=GOLDSET, doc_type="spec", top_k=10,
            router=MagicMock(), reranker=MagicMock(),
            encoder=MagicMock(encode=MagicMock(return_value=np.zeros((1, 3)))),
            milvus=MagicMock(),
        )

    assert results == []


def test_save_output_handles_missing_keys(tmp_path: Path):
    out = tmp_path / "result.json"
    results = [{"gold": {"query_id": "q1", "query": "q?"}, "retrieved": []}]
    metrics = {"n_queries": 1, "mrr": 0.0}
    _save_output(out, metrics, results)
    data = json.loads(out.read_text())
    assert data["details"][0]["expected_doc"] is None
    assert data["details"][0]["expected_section"] is None
