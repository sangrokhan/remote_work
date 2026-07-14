"""The runner's job is comparability. These tests are about the ways it can be lost.

An arm that rides the previous arm's socket, a prep call folded into the totals, a
`both` pass that double-appends to a server-side history -- each produces numbers that
look fine and mean nothing. So the assertions here are less about the happy path than
about the specific ways a run stops being evidence.
"""

from __future__ import annotations

import pytest

from core import metrics, runner
from providers import base

FIXTURE = {
    "system": "You are a terse assistant.",
    "steps": ["What is 2+2?", "And times three?"],
}


def _run(**kw):
    return runner.run(system=FIXTURE["system"], steps=FIXTURE["steps"], **kw)


class _NullCapture:
    """A capture that starts and stops and records nothing: the runner's window logic
    is what is under test, not tcpdump."""

    def __init__(self, arm: str):
        self.arm = arm

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return None

    def result(self) -> dict:
        return {"ok": True, "arm": self.arm}


class TestPlan:
    def test_defaults_to_every_provider_headline_arms(self):
        pairs = runner.plan()
        assert ("gemini", "stateless") in pairs
        assert ("openai", "chat_stateless") in pairs

    def test_diagnostic_arms_are_not_in_a_default_run(self):
        # `nocontext` answers with no history at all. It is a lower bound to read the
        # others against, not a strategy anyone would ship, and a default run that
        # includes it bills for an arm nobody asked about.
        assert ("gemini", "nocontext") not in runner.plan()
        assert "nocontext" in base.get("gemini").ARMS

    def test_arms_run_grouped_by_provider(self):
        pairs = runner.plan({"gemini": ["stateless", "cached"],
                             "openai": ["chat_stateless"]})
        providers = [p for p, _ in pairs]
        assert providers == sorted(providers, key=providers.index)  # no interleaving
        assert providers.count("gemini") == 2

    def test_unknown_arm_is_refused_before_any_call_goes_out(self):
        with pytest.raises(ValueError, match="no arm"):
            runner.plan({"gemini": ["telepathy"]})


class TestMeasure:
    def test_unknown_measure_is_refused(self):
        with pytest.raises(ValueError, match="measure"):
            _run(providers={"gemini": ["stateless"]}, measure="vibes")

    def test_both_on_a_stateful_openai_arm_warns(self):
        # Both passes carry the conversation id; OpenAI appends each of them; turn k+1
        # is then billed for turn k twice. The run still happens -- the operator may
        # have a reason -- but it must not happen quietly.
        run = _run(providers={"openai": ["responses_stateful"]}, measure="both")
        assert any("responses_stateful" in w and "twice" in w
                   for w in run["params"]["warnings"])

    def test_both_on_a_stateless_arm_does_not_warn(self):
        run = _run(providers={"openai": ["responses_stateless"]}, measure="both")
        assert run["params"]["warnings"] == []

    def test_measure_is_recorded_on_every_record(self):
        run = _run(providers={"gemini": ["stateless"]}, measure="latency")
        assert run["params"]["measure"] == "latency"
        assert {r["measure"] for r in run["records"]} == {"latency"}


class TestRecords:
    def test_one_run_holds_both_providers(self):
        run = _run(providers={"gemini": ["stateless"], "openai": ["chat_stateless"]})
        assert {r["provider"] for r in run["records"]} == {"gemini", "openai"}

    def test_every_steady_turn_is_recorded_once(self):
        run = _run(providers={"gemini": ["stateless"]})
        steady = [r for r in run["records"] if r["phase"] == "steady"]
        assert [r["turn"] for r in steady] == [1, 2]

    def test_prep_is_recorded_but_kept_out_of_the_totals(self):
        # The cached arm re-uploads the whole system prompt to build its cache. Folding
        # that into the arm's traffic would drown the thing the arm exists to show.
        run = _run(providers={"gemini": ["cached"]})
        phases = {r["phase"] for r in run["records"]}
        assert "cachegen" in phases

        summary = metrics.summarize(run)
        steady_sent = sum(r["wire_sent"] for r in run["records"]
                          if r["phase"] == "steady")
        assert summary["totals"]["gemini:cached"]["wire_sent"] == steady_sent
        assert summary["prep"]["gemini:cached"]["wire_sent"] > 0

    def test_wall_ms_is_keyed_by_provider_and_arm(self):
        # `stateless` exists on Gemini and could exist on any other provider. Keying on
        # the arm alone lets the second one overwrite the first's timing.
        run = _run(providers={"gemini": ["stateless"], "openai": ["chat_stateless"]})
        assert set(run["wall_ms"]) == {"gemini:stateless", "openai:chat_stateless"}

    def test_a_provider_that_is_not_ready_fails_named_and_does_not_stop_the_run(
            self, monkeypatch):
        monkeypatch.setattr(base.get("openai"), "ready",
                            lambda: (False, "OPENAI_API_KEY is not set"))
        run = _run(providers={"gemini": ["stateless"], "openai": ["chat_stateless"]})

        broken = [r for r in run["records"] if r["provider"] == "openai"]
        assert len(broken) == 1
        assert "OPENAI_API_KEY" in broken[0]["error"]
        # The other provider's arm still ran and still produced numbers.
        assert any(r["provider"] == "gemini" and not r.get("error")
                   for r in run["records"])


class TestIsolation:
    def test_each_arm_starts_on_a_fresh_connection(self, monkeypatch):
        # Without the reset, arm 2 rides arm 1's pooled TLS socket: its pcap opens onto
        # an established connection with no handshake in it, and arm 1's teardown lands
        # inside arm 2's window.
        from core import wire

        resets = []
        real = wire.reset_session
        monkeypatch.setattr(wire, "reset_session",
                            lambda: (resets.append(1), real())[1])
        monkeypatch.setattr(runner.wire, "reset_session", wire.reset_session)

        _run(providers={"gemini": ["stateless", "nocontext"]})
        assert len(resets) >= 3   # once before the run, once before each arm

    def test_capture_that_cannot_start_downgrades_the_run_instead_of_killing_it(self):
        # TRAFFIC_PCAP_DISABLE=1 in the suite's environment. The byte counts come off
        # the socket and stand without a pcap -- but the operator asked for one, so the
        # run has to say why there is none.
        run = _run(providers={"gemini": ["stateless"]}, want_capture=True)
        assert run["params"]["capture"] is False
        assert any("capture unavailable" in w for w in run["params"]["warnings"])
        assert run["records"]

    def test_the_capture_window_covers_the_steady_turns_and_nothing_else(
            self, monkeypatch):
        # A cache build re-uploads the whole prefix -- 185 KB of it on a four-turn run.
        # A pcap holding that cannot be read as evidence of what a 23 KB turn cost, and
        # the cache DELETEs afterwards are not traffic the turns produced either. The
        # window is bounded by the arm's own phases: it opens on the first `steady`
        # event and closes on `teardown`.
        opened, closed, phases_inside = [], [], []
        seen: list[str] = []

        class FakeCapture:
            def __init__(self, timestamp, provider, arm, host):
                self.arm = arm

            def __enter__(self):
                opened.append(self.arm)
                return self

            def __exit__(self, *a):
                closed.append(self.arm)
                phases_inside.append(list(seen))

            def result(self):
                return {"ok": True, "arm": self.arm}

        monkeypatch.setattr(runner.pcap, "available", lambda: (True, "ready"))
        monkeypatch.setattr(runner.pcap, "Capture", FakeCapture)
        monkeypatch.setattr(runner.time, "sleep", lambda s: None)

        def watch(event):
            if opened and not closed:
                seen.append(event["phase"])

        run = _run(providers={"gemini": ["cached"]}, want_capture=True,
                   on_progress=watch)

        assert opened == ["cached"] and closed == ["cached"]
        # Nothing but steady turns happened between the tcpdump start and its stop.
        assert set(phases_inside[0]) == {"steady"}
        assert run["pcaps"]["gemini:cached"]["ok"] is True

    def test_an_arm_with_no_prep_is_captured_whole(self, monkeypatch):
        monkeypatch.setattr(runner.pcap, "available", lambda: (True, "ready"))
        monkeypatch.setattr(runner.time, "sleep", lambda s: None)
        started = []
        monkeypatch.setattr(runner.pcap, "Capture",
                            lambda ts, p, a, h: started.append(a) or _NullCapture(a))
        _run(providers={"gemini": ["stateless"]}, want_capture=True)
        assert started == ["stateless"]

    def test_no_pause_after_the_last_arm(self, monkeypatch):
        slept = []
        monkeypatch.setattr(runner.time, "sleep", lambda s: slept.append(s))
        _run(providers={"gemini": ["stateless", "nocontext"]}, pause_seconds=2)
        # Two arms, one gap: the pause exists to keep the next arm from being rate
        # limited, and after the last one it delays nobody but the operator.
        assert sum(slept) == pytest.approx(2, abs=0.001)
