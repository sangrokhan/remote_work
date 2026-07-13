"""summarize_comparison turns the flat per-arm records into per-arm series and
totals, keeping the setup cost visible and the two axes (wire bytes, input tokens)
separate. No dollar cost.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiment
import metrics


def _out(monkeypatch, arms):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    return experiment.run_comparison("gemini-3.1-flash-lite", turns=3, arms=arms)


def test_per_arm_series_and_totals_present(monkeypatch):
    s = metrics.summarize_comparison(_out(monkeypatch, ["stateless", "cached", "interaction"]))
    assert set(s["series"]) == {"stateless", "cached", "interaction"}
    for arm in s["series"]:
        ser = s["series"][arm]
        assert ser["turns"] == [1, 2, 3]
        assert len(ser["cum_wire"]) == 3
        assert len(ser["cum_input_tokens"]) == 3


def test_cumulative_wire_is_monotonic(monkeypatch):
    s = metrics.summarize_comparison(_out(monkeypatch, ["stateless"]))
    cw = s["series"]["stateless"]["cum_wire"]
    assert cw == sorted(cw)


def test_cached_total_is_the_measured_turns_only(monkeypatch):
    # The cache build runs off the stateless transcript before any measured turn.
    # It is preparation, not traffic under test -- and folding it in would drown
    # every other number, since each build re-uploads the whole system prompt.
    s = metrics.summarize_comparison(_out(monkeypatch, ["cached"]))
    t = s["totals"]["cached"]
    assert t["total_wire"] == t["steady_wire"]
    assert t["cachegen_wire"] > 0         # still reported, just not billed here


def test_latency_stats_present(monkeypatch):
    s = metrics.summarize_comparison(_out(monkeypatch, ["stateless"]))
    lat = s["totals"]["stateless"]["latency"]
    assert set(lat) >= {"mean", "median", "min", "max"}


def test_no_dollar_cost_anywhere(monkeypatch):
    s = metrics.summarize_comparison(_out(monkeypatch, ["stateless", "cached"]))
    assert "cost_usd" not in s.get("totals", {})
    for arm in s["totals"]:
        assert "cost_usd" not in s["totals"][arm]
