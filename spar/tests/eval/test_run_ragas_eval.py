from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spar.eval.run_ragas_eval import _print_summary, _save_output


def _make_dataset(tmp_path: Path, n: int = 2) -> Path:
    p = tmp_path / "ds.jsonl"
    lines = [
        json.dumps({"query_id": f"Q{i}", "question": f"q{i}", "answer": f"a{i}", "contexts": ["ctx"]})
        for i in range(n)
    ]
    p.write_text("\n".join(lines))
    return p


# ── _save_output ──────────────────────────────────────────────────────────────

def test_save_output_creates_file(tmp_path: Path) -> None:
    metrics = {"n_samples": 2, "faithfulness": 0.9, "per_sample": []}
    out = tmp_path / "sub" / "result.json"
    _save_output(out, metrics)
    assert out.exists()
    assert json.loads(out.read_text())["faithfulness"] == pytest.approx(0.9)


def test_save_output_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "a" / "b" / "c" / "result.json"
    _save_output(out, {"n_samples": 1, "per_sample": []})
    assert out.exists()


# ── _print_summary ────────────────────────────────────────────────────────────

def test_print_summary_no_crash(capsys) -> None:
    _print_summary({"n_samples": 3, "faithfulness": 0.8, "answer_relevancy": 0.75, "per_sample": []})
    captured = capsys.readouterr()
    assert "faithfulness" in captured.out
    assert "answer_relevancy" in captured.out
    assert "n_samples" in captured.out


# ── main (CLI) ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_dispatches_compute(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    out = tmp_path / "result.json"

    mock_result = {
        "n_samples": 2,
        "faithfulness": 0.85,
        "per_sample": [{"query_id": "Q0", "faithfulness": 0.85}, {"query_id": "Q1", "faithfulness": 0.85}],
    }

    mock_llm = MagicMock()
    mock_llm.model = "test-model"

    with (
        patch("spar.llm.registry.get_client", new=AsyncMock(return_value=mock_llm)),
        patch("spar.eval.ragas_metrics.compute_ragas_metrics", new=AsyncMock(return_value=mock_result)),
    ):
        import argparse
        from spar.eval.run_ragas_eval import _run

        args = argparse.Namespace(dataset=ds, metrics="faithfulness", output=out)
        await _run(args)

    assert out.exists()
    saved = json.loads(out.read_text())
    assert saved["faithfulness"] == pytest.approx(0.85)
