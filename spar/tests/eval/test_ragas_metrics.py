from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from spar.eval.ragas_metrics import (
    RagasSample,
    answer_relevancy_score,
    compute_ragas_metrics,
    faithfulness_score,
    load_ragas_dataset,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_llm(*responses: str) -> AsyncMock:
    llm = AsyncMock()
    llm.chat = AsyncMock(side_effect=list(responses))
    return llm


def _make_encoder(vecs: np.ndarray) -> MagicMock:
    enc = MagicMock()
    enc.encode = MagicMock(return_value=vecs)
    return enc


SAMPLE: RagasSample = {
    "query_id": "Q001",
    "question": "What is handover?",
    "answer": "Handover transfers an active call between cells.",
    "contexts": ["Handover transfers an active call between cells."],
}


# ── load_ragas_dataset ─────────────────────────────────────────────────────────

def test_load_ragas_dataset_basic(tmp_path: Path) -> None:
    lines = [
        json.dumps({"query_id": "Q1", "question": "q", "answer": "a", "contexts": ["c"]}),
        "",
        json.dumps({"query_id": "Q2", "question": "q2", "answer": "a2", "contexts": ["c2"], "ground_truth": "gt"}),
    ]
    p = tmp_path / "ds.jsonl"
    p.write_text("\n".join(lines))
    samples = load_ragas_dataset(p)
    assert len(samples) == 2
    assert samples[0]["query_id"] == "Q1"
    assert "ground_truth" not in samples[0]
    assert samples[1]["ground_truth"] == "gt"


def test_load_ragas_dataset_missing_field(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"query_id": "Q1", "question": "q"}))
    with pytest.raises(ValueError, match="missing fields"):
        load_ragas_dataset(p)


# ── faithfulness_score ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_faithfulness_all_supported() -> None:
    llm = _make_llm(
        "Handover transfers a call between cells.\nX2 uses direct eNB interface.",
        "yes",
        "yes",
    )
    score = await faithfulness_score("q", "ans", ["ctx"], llm)
    assert score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_faithfulness_partial() -> None:
    llm = _make_llm(
        "Claim A.\nClaim B.\nClaim C.",
        "yes",
        "no",
        "yes",
    )
    score = await faithfulness_score("q", "ans", ["ctx"], llm)
    assert score == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_faithfulness_no_claims() -> None:
    llm = _make_llm("")
    score = await faithfulness_score("q", "", ["ctx"], llm)
    assert score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_faithfulness_none_supported() -> None:
    llm = _make_llm("Claim A.\nClaim B.", "no", "no")
    score = await faithfulness_score("q", "ans", ["ctx"], llm)
    assert score == pytest.approx(0.0)


# ── answer_relevancy_score ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_answer_relevancy_high() -> None:
    llm = _make_llm("What is handover?\nHow does handover work?")
    # normalized vecs: orig + 2 generated all pointing same direction → similarity ≈ 1
    vecs = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    enc = _make_encoder(vecs)
    score = await answer_relevancy_score("What is handover?", "ans", enc, llm, n_questions=2)
    assert score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_answer_relevancy_low() -> None:
    llm = _make_llm("Q1\nQ2")
    # orig orthogonal to generated
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    enc = _make_encoder(vecs)
    score = await answer_relevancy_score("q", "ans", enc, llm, n_questions=2)
    assert score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_answer_relevancy_no_generated() -> None:
    llm = _make_llm("")
    enc = _make_encoder(np.array([[1.0, 0.0]]))
    score = await answer_relevancy_score("q", "ans", enc, llm)
    assert score == pytest.approx(0.0)


# ── compute_ragas_metrics ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_compute_faithfulness_only() -> None:
    llm = _make_llm("Claim A.", "yes")
    result = await compute_ragas_metrics([SAMPLE], metrics=["faithfulness"], llm=llm)
    assert result["n_samples"] == 1
    assert result["faithfulness"] == pytest.approx(1.0)
    assert result["per_sample"][0]["query_id"] == "Q001"


@pytest.mark.asyncio
async def test_compute_both_metrics() -> None:
    llm = _make_llm(
        "Claim A.",   # faithfulness: claims
        "yes",        # faithfulness: verdict
        "Gen Q1?\nGen Q2?",  # answer_relevancy: generated questions
    )
    vecs = np.array([[1.0, 0.0], [0.8, 0.6], [0.9, 0.436]])
    enc = _make_encoder(vecs)
    result = await compute_ragas_metrics(
        [SAMPLE], metrics=["faithfulness", "answer_relevancy"], llm=llm, encoder=enc
    )
    assert "faithfulness" in result
    assert "answer_relevancy" in result
    assert 0.0 <= result["answer_relevancy"] <= 1.0


@pytest.mark.asyncio
async def test_compute_empty_raises() -> None:
    llm = AsyncMock()
    with pytest.raises(ValueError, match="empty"):
        await compute_ragas_metrics([], metrics=["faithfulness"], llm=llm)


@pytest.mark.asyncio
async def test_compute_unknown_metric_raises() -> None:
    llm = AsyncMock()
    with pytest.raises(ValueError, match="Unknown metrics"):
        await compute_ragas_metrics([SAMPLE], metrics=["bleu"], llm=llm)


@pytest.mark.asyncio
async def test_compute_answer_relevancy_requires_encoder() -> None:
    llm = AsyncMock()
    with pytest.raises(ValueError, match="encoder required"):
        await compute_ragas_metrics([SAMPLE], metrics=["answer_relevancy"], llm=llm)
