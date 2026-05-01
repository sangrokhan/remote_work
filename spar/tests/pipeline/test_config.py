from __future__ import annotations

from spar.pipeline.config import GraphConfig, PRESET_CONFIGS
from spar.pipeline.state import SparState


def test_graphconfig_defaults():
    cfg = GraphConfig(name="x")
    assert cfg.use_query_expansion is False
    assert cfg.use_prepare_context is False
    assert cfg.use_reranker is False
    assert cfg.use_real_generate is False


def test_preset_names_unique():
    names = [c.name for c in PRESET_CONFIGS]
    assert len(names) == len(set(names))


def test_preset_baseline_all_false():
    baseline = next(c for c in PRESET_CONFIGS if c.name == "baseline")
    assert not baseline.use_query_expansion
    assert not baseline.use_prepare_context
    assert not baseline.use_reranker
    assert not baseline.use_real_generate


def test_preset_full_retrieval():
    cfg = next(c for c in PRESET_CONFIGS if c.name == "full_retrieval")
    assert cfg.use_query_expansion
    assert cfg.use_prepare_context
    assert cfg.use_reranker
    assert not cfg.use_real_generate


def test_preset_e2e_all_true():
    cfg = next(c for c in PRESET_CONFIGS if c.name == "e2e")
    assert cfg.use_query_expansion
    assert cfg.use_prepare_context
    assert cfg.use_reranker
    assert cfg.use_real_generate


def test_sparstate_has_timing_field():
    s: SparState = {"query": "test", "node_timings": {"preprocess": 12.5}}
    assert s["node_timings"]["preprocess"] == 12.5


def test_sparstate_has_performance_fields():
    # performance eval inputs (populated by eval_suite; ignored in production)
    metrics = {"recall_at_5": 1.0}
    s: SparState = {
        "query": "test",
        "gold_chunks": ["section_4.1"],
        "gold_answer": "The answer is 5.",
        "eval_metrics": metrics,
    }
    assert s["gold_chunks"] == ["section_4.1"]
    assert s["gold_answer"] == "The answer is 5."
    assert s["eval_metrics"]["recall_at_5"] == 1.0
