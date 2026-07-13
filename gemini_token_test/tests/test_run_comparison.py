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


def _record_sleeps(monkeypatch):
    """The pause ticks run_comparison would sleep, without actually sleeping.

    Only the 1-second pause ticks are of interest; the sub-second settle after each
    arm (which lets the connection teardown land in that arm's pcap) is not a pause.
    """
    slept: list[float] = []
    monkeypatch.setattr(experiment.time, "sleep",
                        lambda s: slept.append(s) if s >= 1 else None)
    return slept


def test_pause_spaces_the_arms_apart(monkeypatch):
    # Arms hit the same rate-limited project back to back, so the operator can ask
    # for a gap between them. It goes *between* arms, never after the last one.
    monkeypatch.setenv("GEMINI_MOCK", "1")
    slept = _record_sleeps(monkeypatch)
    experiment.run_comparison("gemini-3.1-flash-lite", turns=1,
                              arms=["stateless", "cached", "interaction"],
                              pause_seconds=5)
    assert sum(slept) == 10          # 5s in each of the two gaps


def test_no_pause_by_default(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    slept = _record_sleeps(monkeypatch)
    experiment.run_comparison("gemini-3.1-flash-lite", turns=1,
                              arms=["stateless", "cached"])
    assert slept == []


def _pauses(events):
    return [e for e in events if e["stage"] == "pause"]


def test_pause_counts_down_second_by_second(monkeypatch):
    # A single "pausing" event and then a minute of silence is indistinguishable
    # from a hang. Tick, so the operator can see the gap draining.
    monkeypatch.setenv("GEMINI_MOCK", "1")
    slept = _record_sleeps(monkeypatch)
    events = []
    experiment.run_comparison("gemini-3.1-flash-lite", turns=1,
                              arms=["stateless", "cached"], pause_seconds=3,
                              on_progress=events.append)
    assert [e["remaining"] for e in _pauses(events)] == [3, 2, 1]
    assert slept == [1, 1, 1]


def test_pause_event_carries_the_total_so_a_bar_can_be_drawn(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    _record_sleeps(monkeypatch)
    events = []
    experiment.run_comparison("gemini-3.1-flash-lite", turns=1,
                              arms=["stateless", "cached"], pause_seconds=2,
                              on_progress=events.append)
    p = _pauses(events)
    assert all(e["pause_total"] == 2 for e in p)
    # And it names the arm the run is about to start, not the one just finished.
    assert all(e["next_arm"] == "cached" for e in p)


def test_thought_tokens_are_captured(monkeypatch):
    # generateContent arms must report thought tokens too, so the token axis is
    # comparable with the interaction arm which always reports them.
    out = _run(monkeypatch, ["stateless"])
    assert all("thought_tokens" in r for r in out["records"])


# --- per-turn progress -----------------------------------------------------

def _steps(events, arm):
    return [(e["phase"], e["turn"]) for e in events
            if e["stage"] == arm and e.get("phase")]


def test_stateless_arm_reports_every_turn(monkeypatch):
    # Only the interaction arm ever emitted turn events, so the other arms sat at
    # "turn 0/N" for the whole run and a stall was indistinguishable from progress.
    monkeypatch.setenv("GEMINI_MOCK", "1")
    events = []
    experiment.run_comparison("gemini-3.1-flash-lite", turns=3, arms=["stateless"],
                              on_progress=events.append)
    assert _steps(events, "stateless") == [("steady", 1), ("steady", 2), ("steady", 3)]


def test_nocontext_arm_reports_every_turn(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    events = []
    experiment.run_comparison("gemini-3.1-flash-lite", turns=2, arms=["nocontext"],
                              on_progress=events.append)
    assert _steps(events, "nocontext") == [("steady", 1), ("steady", 2)]


def test_cached_arm_reports_the_cache_builds_and_the_steady_turns(monkeypatch):
    # The cached arm builds one cache per turn before it answers anything. That is
    # the slowest part of the run; it must not look like a hang.
    monkeypatch.setenv("GEMINI_MOCK", "1")
    events = []
    experiment.run_comparison("gemini-3.1-flash-lite", turns=2, arms=["cached"],
                              on_progress=events.append)
    assert _steps(events, "cached") == [
        ("setup", 1), ("setup", 2), ("steady", 1), ("steady", 2),
    ]


def test_turn_events_carry_the_total(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    events = []
    experiment.run_comparison("gemini-3.1-flash-lite", turns=3, arms=["stateless"],
                              on_progress=events.append)
    turns = [e for e in events if e["stage"] == "stateless" and e.get("phase")]
    assert all(e["turns"] == 3 for e in turns)
