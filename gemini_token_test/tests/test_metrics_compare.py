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


def test_cached_total_is_setup_plus_steady(monkeypatch):
    s = metrics.summarize_comparison(_out(monkeypatch, ["cached"]))
    t = s["totals"]["cached"]
    assert t["total_wire"] == t["setup_wire"] + t["steady_wire"]
    assert t["setup_wire"] > 0            # cache-build upload is counted


def test_cached_cumulative_starts_above_setup_cost(monkeypatch):
    # The cache-build cost is front-loaded, so cached's cumulative wire must begin
    # at least at the setup cost, not at zero -- otherwise the chart flatters it.
    s = metrics.summarize_comparison(_out(monkeypatch, ["cached"]))
    setup = s["totals"]["cached"]["setup_wire"]
    assert s["series"]["cached"]["cum_wire"][0] >= setup


def test_latency_stats_present(monkeypatch):
    s = metrics.summarize_comparison(_out(monkeypatch, ["stateless"]))
    lat = s["totals"]["stateless"]["latency"]
    assert set(lat) >= {"mean", "median", "min", "max"}


def test_no_dollar_cost_anywhere(monkeypatch):
    s = metrics.summarize_comparison(_out(monkeypatch, ["stateless", "cached"]))
    assert "cost_usd" not in s.get("totals", {})
    for arm in s["totals"]:
        assert "cost_usd" not in s["totals"][arm]
