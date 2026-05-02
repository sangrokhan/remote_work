from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from spar.eval.run_e2e_eval import _save_ragas_dataset, _to_ragas_samples

# ── fixtures ───────────────────────────────────────────────────────────────────

GOLD = {
    "query_id": "q001",
    "query": "What is maxHARQTx default?",
    "gold_answer": "The default is 5.",
    "source_doc": "param_ref.md",
    "section": "4.1",
}

CHUNK = {"text": "maxHARQTx default is 5.", "score": 0.9, "source_doc": "param_ref.md", "section_num": "4.1"}

FAKE_STATE_WITH_RERANK = {
    "query": GOLD["query"],
    "raw_chunks": [CHUNK],
    "reranked_chunks": [CHUNK],
    "answer": "The default is 5.",
    "node_timings": {"route": 10.0, "rag_retrieve": 80.0},
    "node_trace": ["route", "rag_retrieve"],
}

FAKE_STATE_NO_RERANK = {
    "query": GOLD["query"],
    "raw_chunks": [CHUNK],
    "reranked_chunks": None,
    "answer": "The default is 5.",
    "node_timings": {},
    "node_trace": [],
}


# ── _to_ragas_samples ──────────────────────────────────────────────────────────

def test_to_ragas_samples_uses_reranked_chunks():
    pairs = [(GOLD, FAKE_STATE_WITH_RERANK)]
    samples = _to_ragas_samples(pairs)
    assert len(samples) == 1
    s = samples[0]
    assert s["query_id"] == "q001"
    assert s["question"] == GOLD["query"]
    assert s["answer"] == "The default is 5."
    assert s["contexts"] == [CHUNK["text"]]
    assert s["ground_truth"] == GOLD["gold_answer"]


def test_to_ragas_samples_falls_back_to_raw_chunks():
    pairs = [(GOLD, FAKE_STATE_NO_RERANK)]
    samples = _to_ragas_samples(pairs)
    assert samples[0]["contexts"] == [CHUNK["text"]]


def test_to_ragas_samples_no_gold_answer_skips_ground_truth():
    gold_no_answer = {**GOLD, "gold_answer": None}
    pairs = [(gold_no_answer, FAKE_STATE_WITH_RERANK)]
    samples = _to_ragas_samples(pairs)
    assert "ground_truth" not in samples[0]


def test_to_ragas_samples_empty_contexts_when_no_text():
    state = {**FAKE_STATE_WITH_RERANK, "reranked_chunks": [{"score": 0.9}]}
    samples = _to_ragas_samples([(GOLD, state)])
    assert samples[0]["contexts"] == []


# ── _save_ragas_dataset ────────────────────────────────────────────────────────

def test_save_ragas_dataset_creates_valid_jsonl(tmp_path: Path):
    samples = [
        {"query_id": "q1", "question": "q?", "answer": "a", "contexts": ["ctx"]},
        {"query_id": "q2", "question": "q2?", "answer": "a2", "contexts": ["ctx2"], "ground_truth": "gt"},
    ]
    out = tmp_path / "sub" / "dataset.jsonl"
    _save_ragas_dataset(samples, out)
    assert out.exists()
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[1]["ground_truth"] == "gt"


# ── _run_pipeline ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_pipeline_returns_pairs():
    from spar.eval.run_e2e_eval import _run_pipeline

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=FAKE_STATE_WITH_RERANK)

    with patch("spar.eval.run_e2e_eval.build_graph", return_value=mock_graph):
        pairs = await _run_pipeline(
            goldset=[GOLD],
            config_name="e2e",
            top_k=10,
            llm_client=MagicMock(),
            router=MagicMock(),
            reranker=MagicMock(),
            encoder=MagicMock(),
            milvus=MagicMock(),
        )

    assert len(pairs) == 1
    gold_out, state_out = pairs[0]
    assert gold_out == GOLD
    assert state_out["answer"] == "The default is 5."


@pytest.mark.asyncio
async def test_run_pipeline_skips_failed_queries():
    from spar.eval.run_e2e_eval import _run_pipeline

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("graph error"))

    with patch("spar.eval.run_e2e_eval.build_graph", return_value=mock_graph):
        pairs = await _run_pipeline(
            goldset=[GOLD],
            config_name="e2e",
            top_k=10,
            llm_client=MagicMock(),
            router=MagicMock(),
            reranker=MagicMock(),
            encoder=MagicMock(),
            milvus=MagicMock(),
        )

    assert pairs == []


# ── _run (integration) ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_saves_output_and_dataset(tmp_path: Path):
    goldset_path = tmp_path / "goldset.jsonl"
    goldset_path.write_text(json.dumps(GOLD) + "\n")
    out = tmp_path / "result.json"
    ds_out = tmp_path / "dataset.jsonl"

    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_encoder = MagicMock()
    mock_encoder.model_name = "test-encoder"

    ragas_result = {
        "n_samples": 1,
        "faithfulness": 0.9,
        "per_sample": [{"query_id": "q001", "faithfulness": 0.9}],
    }

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=FAKE_STATE_WITH_RERANK)

    with (
        patch("spar.eval.run_e2e_eval.get_client", new=AsyncMock(return_value=mock_llm)),
        patch("spar.eval.run_e2e_eval.get_encoder", new=AsyncMock(return_value=mock_encoder)),
        patch("spar.eval.run_e2e_eval.get_reranker", new=AsyncMock(return_value=MagicMock())),
        patch("spar.eval.run_e2e_eval.HybridRouter", return_value=MagicMock()),
        patch("spar.eval.run_e2e_eval.SparMilvusClient", return_value=MagicMock()),
        patch("spar.eval.run_e2e_eval.build_graph", return_value=mock_graph),
        patch("spar.eval.run_e2e_eval.compute_ragas_metrics", new=AsyncMock(return_value=ragas_result)),
    ):
        from spar.eval.run_e2e_eval import _run

        args = argparse.Namespace(
            goldset=goldset_path,
            config="e2e",
            metrics="faithfulness",
            top_k=10,
            save_dataset=ds_out,
            output=out,
        )
        await _run(args)

    assert out.exists()
    saved = json.loads(out.read_text())
    assert saved["faithfulness"] == pytest.approx(0.9)

    assert ds_out.exists()
    ds_lines = [json.loads(l) for l in ds_out.read_text().splitlines() if l.strip()]
    assert ds_lines[0]["question"] == GOLD["query"]
