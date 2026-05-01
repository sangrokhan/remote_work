from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

ROUTES = [
    "structured_lookup",
    "definition_explain",
    "procedural",
    "diagnostic",
    "comparative",
    "default_rag",
]


@pytest.fixture()
def small_goldset():
    return [
        {"query_id": "RQ0001", "query": "ALM-1234 alarm", "expected_route": "structured_lookup",
         "source_doc": "d.md", "spec_number": "", "release": "Rel-18", "qa_query_id": "Q0001"},
        {"query_id": "RQ0002", "query": "What is Carrier Aggregation?", "expected_route": "definition_explain",
         "source_doc": "d.md", "spec_number": "", "release": "Rel-18", "qa_query_id": "Q0002"},
        {"query_id": "RQ0003", "query": "How to configure RACH parameters?", "expected_route": "procedural",
         "source_doc": "d.md", "spec_number": "", "release": "Rel-18", "qa_query_id": "Q0003"},
    ]


def test_compute_metrics_perfect():
    from run_router_eval import compute_metrics
    expected = ["structured_lookup", "definition_explain", "procedural"]
    predicted = ["structured_lookup", "definition_explain", "procedural"]
    metrics = compute_metrics(expected, predicted, ROUTES)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["per_route"]["structured_lookup"]["precision"] == pytest.approx(1.0)
    assert metrics["per_route"]["structured_lookup"]["recall"] == pytest.approx(1.0)


def test_compute_metrics_one_wrong():
    from run_router_eval import compute_metrics
    expected = ["structured_lookup", "definition_explain", "procedural"]
    predicted = ["structured_lookup", "structured_lookup", "procedural"]
    metrics = compute_metrics(expected, predicted, ROUTES)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["per_route"]["definition_explain"]["recall"] == pytest.approx(0.0)
    assert metrics["per_route"]["definition_explain"]["precision"] == pytest.approx(0.0)
    assert metrics["per_route"]["structured_lookup"]["precision"] == pytest.approx(0.5)


def test_compute_metrics_empty():
    from run_router_eval import compute_metrics
    metrics = compute_metrics([], [], ROUTES)
    assert metrics["accuracy"] == pytest.approx(0.0)


def test_build_confusion_matrix():
    from run_router_eval import build_confusion_matrix
    expected = ["structured_lookup", "definition_explain"]
    predicted = ["structured_lookup", "structured_lookup"]
    matrix = build_confusion_matrix(expected, predicted, ROUTES)
    assert matrix["structured_lookup"]["structured_lookup"] == 1
    assert matrix["definition_explain"]["structured_lookup"] == 1


def test_eval_regex_layer(small_goldset):
    from run_router_eval import eval_regex
    expected, predicted = eval_regex(small_goldset)
    # ALM-1234 → structured_lookup (regex matches alarm code)
    assert expected[0] == "structured_lookup"
    assert predicted[0] == "structured_lookup"
    assert len(expected) == len(predicted) == 3


def test_format_report_contains_accuracy():
    from run_router_eval import compute_metrics, format_report
    expected = ["structured_lookup"]
    predicted = ["structured_lookup"]
    metrics = compute_metrics(expected, predicted, ROUTES)
    matrix = {"structured_lookup": {"structured_lookup": 1}}
    report = format_report(metrics, matrix, layer="regex", date_str="2026-05-01")
    assert "accuracy" in report.lower()
    assert "regex" in report.lower()
    assert "structured_lookup" in report


def test_load_goldset(tmp_path):
    from run_router_eval import load_goldset
    items = [
        {"query_id": "RQ0001", "query": "q", "expected_route": "procedural",
         "source_doc": "d.md", "spec_number": "", "release": "Rel-18", "qa_query_id": "Q0001"},
    ]
    p = tmp_path / "goldset.jsonl"
    p.write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")
    loaded = load_goldset(p)
    assert len(loaded) == 1
    assert loaded[0]["expected_route"] == "procedural"


import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch


def _make_mock_encoder(dim: int = 8):
    """test_embedding_router.py 와 동일한 stub encoder."""
    from spar.encoder.base import EncoderClient
    encoder = MagicMock(spec=EncoderClient)
    rng = np.random.default_rng(42)
    cache: dict[str, np.ndarray] = {}

    def _encode(texts: list[str], *, normalize: bool = True) -> np.ndarray:
        vecs = []
        for t in texts:
            if t not in cache:
                v = rng.random(dim).astype(np.float32)
                if normalize:
                    v /= np.linalg.norm(v)
                cache[t] = v
            vecs.append(cache[t])
        return np.array(vecs)

    encoder.encode.side_effect = _encode
    return encoder


def test_eval_embedding_layer_returns_routes(small_goldset):
    from run_router_eval import eval_embedding
    mock_enc = _make_mock_encoder()
    with patch("spar.encoder.registry.get_encoder", new=AsyncMock(return_value=mock_enc)):
        expected, predicted = eval_embedding(small_goldset, threshold=0.0)
    assert len(expected) == len(predicted) == 3
    assert all(r in ROUTES for r in predicted)


def test_eval_embedding_strict_threshold_all_fallback(small_goldset):
    from run_router_eval import eval_embedding
    mock_enc = _make_mock_encoder()
    with patch("spar.encoder.registry.get_encoder", new=AsyncMock(return_value=mock_enc)):
        _, predicted = eval_embedding(small_goldset, threshold=2.0)
    # threshold > max possible cosine similarity → 전부 default_rag
    assert all(p == "default_rag" for p in predicted)
