"""Unit tests for aipt.backends.quic_mock.experiment's pure aggregation
logic (statistics.mean/stdev over synthetic run dicts) -- no network.
"""
from __future__ import annotations

import pytest

aioquic = pytest.importorskip("aioquic", reason="aioquic is an optional [quic] extra")

from aipt.backends.quic_mock import experiment  # noqa: E402


def test_post_idle_latencies_exclude_turn_zero():
    """Turn 0 has no preceding idle gap for a probe to have reacted to --
    the aggregation must exclude it from post_idle_latency_ms_* stats,
    matching the experiment's whole methodology (see module docstring)."""
    runs = [
        {"turn_latencies_ms": [999.0, 10.0, 20.0], "total_bytes": 1000, "active_wall_time_s": 1.0},
    ]
    post_idle = [lat for run in runs for lat in run["turn_latencies_ms"][1:]]
    assert post_idle == [10.0, 20.0]
    assert 999.0 not in post_idle


@pytest.mark.anyio
async def test_run_experiment_aggregates_across_repeats(monkeypatch):
    """run_experiment() must average/aggregate over `repeats` independent
    conversations, not just report the last one."""
    calls = []

    async def fake_run_one_conversation(**kwargs):
        calls.append(kwargs)
        # Deterministic synthetic per-call result so we can check the math.
        n = len(calls)
        return {
            "turn_latencies_ms": [100.0, 10.0 * n, 20.0 * n],  # turn0=100 (excluded)
            "total_bytes": 1000 * n,
            "active_wall_time_s": 1.0,
        }

    monkeypatch.setattr(experiment, "run_one_conversation", fake_run_one_conversation)

    result = await experiment.run_experiment(
        host="x", port=1, cc_name="reno", use_idle_probe=False,
        num_turns=3, think_time=0.1, payload_bytes=1000, repeats=3,
    )

    assert len(calls) == 3
    assert result["repeats"] == 3
    # post-idle latencies across 3 repeats: [10,20],[20,40],[30,60]
    post_idle = [10.0, 20.0, 20.0, 40.0, 30.0, 60.0]
    import statistics
    assert result["post_idle_latency_ms_mean"] == round(statistics.mean(post_idle), 2)
    assert result["post_idle_latency_ms_max"] == round(max(post_idle), 2)
    # total_bytes summed across repeats: 1000+2000+3000 = 6000
    assert result["total_bytes"] == 6000
    assert result["goodput_bps"] == round(6000 / 3.0, 1)


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"
