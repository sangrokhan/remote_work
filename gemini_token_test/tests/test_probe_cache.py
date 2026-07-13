"""The probe runs on page load, so it must not bill on every refresh.

It is also the only source of truth for the interaction arm: the model catalog
never advertises Interactions support, so a model's arm coverage is only complete
once the probe has answered for it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe


def _counting(monkeypatch, verdict="supported"):
    calls = {"n": 0}

    def fake():
        calls["n"] += 1
        return {"mock": False, "targets": [], "conclusion": {},
                "models": {"gemini-3.1-flash-lite": verdict}}

    monkeypatch.setattr(probe, "probe_interactions", fake)
    probe.clear_cache()
    return calls


def test_first_call_probes(monkeypatch):
    calls = _counting(monkeypatch)
    probe.probe_cached()
    assert calls["n"] == 1


def test_second_call_is_served_from_cache(monkeypatch):
    calls = _counting(monkeypatch)
    probe.probe_cached()
    probe.probe_cached()
    assert calls["n"] == 1


def test_cached_result_is_marked_as_cached(monkeypatch):
    _counting(monkeypatch)
    assert probe.probe_cached()["cached"] is False
    assert probe.probe_cached()["cached"] is True


def test_force_bypasses_the_cache(monkeypatch):
    calls = _counting(monkeypatch)
    probe.probe_cached()
    probe.probe_cached(force=True)
    assert calls["n"] == 2


def test_expired_cache_reprobes(monkeypatch):
    calls = _counting(monkeypatch)
    monkeypatch.setenv("PROBE_CACHE_TTL", "0")
    probe.probe_cached()
    probe.probe_cached()
    assert calls["n"] == 2


def test_interaction_verdicts_are_exposed_for_the_model_list(monkeypatch):
    _counting(monkeypatch, verdict="supported")
    probe.probe_cached()
    assert probe.interaction_verdicts() == {"gemini-3.1-flash-lite": "supported"}


def test_verdicts_are_empty_before_any_probe(monkeypatch):
    _counting(monkeypatch)
    assert probe.interaction_verdicts() == {}


def test_probe_targets_the_fixed_model_by_default(monkeypatch):
    # A probe pointed at models the experiment never runs answers nothing useful.
    import gemini_client
    assert gemini_client.DEFAULT_MODEL in probe.PROBE_MODELS
