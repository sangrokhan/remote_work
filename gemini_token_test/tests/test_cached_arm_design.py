"""The cached arm caches the real conversation, and its cache build is not billed
to the comparison.

Two things were wrong. The caches were built from a placeholder string standing in
for the model's answers, so the arm was measuring a cache of a conversation that
never happened. And the build traffic was counted as setup, which dominated
everything -- each build re-uploads the whole system prompt, so n turns cost O(n^2)
and the arm looked far worse than it is.

The build is a preparation step, not part of the measurement: it runs off the
stateless transcript, its traffic is recorded but excluded from the totals, and the
measured turns are the ones that reference the cache -- turn k against the cache
built for turn k-1.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiment
import metrics


def _run(monkeypatch, arms=("stateless", "cached"), turns=3):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    return experiment.run_comparison("gemini-3.1-flash-lite", turns=turns,
                                     arms=list(arms))


def _phase(out, arm, phase):
    return [r for r in out["records"] if r["arm"] == arm and r["phase"] == phase]


def test_cache_generation_is_its_own_phase(monkeypatch):
    out = _run(monkeypatch)
    assert len(_phase(out, "cached", "cachegen")) == 3
    assert not _phase(out, "cached", "setup"), "setup was the billed bucket; gone"


def test_caches_are_built_from_the_real_stateless_answers(monkeypatch):
    # The cache must hold what the model actually said, not a placeholder -- caching
    # a conversation that never happened measures nothing.
    out = _run(monkeypatch)
    answers = [r["response_text"] for r in _phase(out, "stateless", "steady")]
    built = _phase(out, "cached", "cachegen")
    blob = " ".join(r["request_raw"] for r in built)
    assert "placeholder" not in blob.lower()
    assert any(a and a in blob for a in answers)


def test_turn_k_references_the_cache_built_for_turn_k_minus_1(monkeypatch):
    out = _run(monkeypatch)
    gen = _phase(out, "cached", "cachegen")
    steady = _phase(out, "cached", "steady")
    ids = {r["turn"]: r["cache_id"] for r in gen}
    assert steady[0]["cache_id"] is None            # turn 1 has no prior cache
    assert steady[1]["cache_id"] == ids[1]          # turn 2 -> cache from turn 1
    assert steady[2]["cache_id"] == ids[2]


def test_cachegen_traffic_is_excluded_from_the_totals(monkeypatch):
    out = _run(monkeypatch)
    s = metrics.summarize_comparison(out)
    t = s["totals"]["cached"]
    steady = _phase(out, "cached", "steady")
    assert t["total_wire"] == sum(r["wire_sent"] + r["wire_recv"] for r in steady)
    assert t["total_input_tokens"] == sum(r["input_tokens"] for r in steady)


def test_cachegen_cost_is_still_reported_separately(monkeypatch):
    # Excluded from the comparison, but not hidden: the build is real money and the
    # operator should see what it cost.
    out = _run(monkeypatch)
    t = metrics.summarize_comparison(out)["totals"]["cached"]
    gen = _phase(out, "cached", "cachegen")
    assert t["cachegen_wire"] == sum(r["wire_sent"] + r["wire_recv"] for r in gen)
    assert t["cachegen_wire"] > 0


def test_cumulative_series_no_longer_carries_a_setup_offset(monkeypatch):
    out = _run(monkeypatch)
    s = metrics.summarize_comparison(out)
    ser = s["series"]["cached"]
    steady = _phase(out, "cached", "steady")
    first = steady[0]["wire_sent"] + steady[0]["wire_recv"]
    assert ser["cum_wire"][0] == first


def test_cached_without_stateless_still_gets_a_transcript(monkeypatch):
    # The cached arm needs the stateless conversation to cache. Asking for cached
    # alone must not silently cache a placeholder.
    out = _run(monkeypatch, arms=["cached"], turns=2)
    assert len(_phase(out, "cached", "cachegen")) == 2
    blob = " ".join(r["request_raw"] for r in _phase(out, "cached", "cachegen"))
    assert "placeholder" not in blob.lower()
