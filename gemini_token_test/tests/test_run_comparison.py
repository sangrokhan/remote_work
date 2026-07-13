"""run_comparison orchestrates the four stages into one comparable record set.

Runs in mock mode (no network). Every arm emits the shared per-turn record; the
cachebuild calls land in a `setup` phase attributed to `cached`, and everything
else is `steady`. The headline compares stateless (1) vs cached (3) vs interaction
(4); nocontext is the lower bound.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiment

_COMMON = {"arm", "phase", "turn", "wire_sent", "wire_recv", "elapsed_ms",
           "input_tokens", "cached_tokens", "output_tokens", "thought_tokens",
           "total_tokens", "request_raw", "response_raw", "error"}


def _run(monkeypatch, arms):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    return experiment.run_comparison("gemini-3.1-flash-lite", turns=3, arms=arms)


def _by(records, arm, phase=None):
    return [r for r in records
            if r["arm"] == arm and (phase is None or r["phase"] == phase)]


def test_every_record_has_the_common_schema(monkeypatch):
    out = _run(monkeypatch, ["stateless", "cached", "interaction", "nocontext"])
    for r in out["records"]:
        assert _COMMON <= set(r), f"missing keys in {r['arm']}: {_COMMON - set(r)}"


def test_stateless_is_three_steady_turns(monkeypatch):
    out = _run(monkeypatch, ["stateless"])
    steady = _by(out["records"], "stateless", "steady")
    assert [r["turn"] for r in steady] == [1, 2, 3]


def test_cached_has_a_setup_bucket_and_steady_turns(monkeypatch):
    out = _run(monkeypatch, ["cached"])
    setup = _by(out["records"], "cached", "setup")
    steady = _by(out["records"], "cached", "steady")
    assert setup, "cachebuild calls must be recorded as setup"
    assert all(r.get("cache_id") is not None or r.get("skipped") for r in setup)
    assert [r["turn"] for r in steady] == [1, 2, 3]


def test_interaction_arm_present(monkeypatch):
    out = _run(monkeypatch, ["interaction"])
    steady = _by(out["records"], "interaction", "steady")
    assert len(steady) == 3
    assert all(r["arm"] == "interaction" for r in steady)


def test_nocontext_arm_present(monkeypatch):
    out = _run(monkeypatch, ["nocontext"])
    assert len(_by(out["records"], "nocontext", "steady")) == 3


def test_headline_arms_default_without_nocontext(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=2)
    arms = {r["arm"] for r in out["records"]}
    assert {"stateless", "cached", "interaction"} <= arms


def test_thought_tokens_are_captured(monkeypatch):
    # generateContent arms must report thought tokens too, so the token axis is
    # comparable with the interaction arm which always reports them.
    out = _run(monkeypatch, ["stateless"])
    assert all("thought_tokens" in r for r in out["records"])
