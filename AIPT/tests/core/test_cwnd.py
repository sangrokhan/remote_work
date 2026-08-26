"""The congestion-window monitor: capability reporting, sample parsing, reset counting.

What is tested here is everything that decides what a reader is told, because that is
where a wrong answer does damage. `idle_resets` in particular: it is the number the
feature exists to produce, and the difference between "the window was reset after idle"
and "the window collapsed because of loss" is one field. Getting that wrong would let a
lossy link masquerade as the finding -- or hide it.

What is not tested here is netlink itself. Reading real congestion state needs real
sockets, which needs a `@pytest.mark.live` test against a loopback server (checked
against `ss`), not this file. This file fakes the helper out entirely so it runs on a
box with no compiler and no netlink -- a test that silently skips is worse than no test.

Merged from `token_traffic/tests/test_cwnd.py` (detailed docstrings, provider/arm/kind
labelling, dumps/exact_queries instrumentation) and `tcp_congestion/tests/test_cwnd.py`
(`announce(sock)` API, single-string label). Per DESIGN.md section 6 decision #1,
`aipt.core.cwnd.Monitor` takes a single opaque `label` string -- callers that want
`provider:arm:kind` assemble it before construction. CSV-export tests from the
token_traffic side (`export.cwnd_csv`, `export.cwnd_summary_csv`) and the
sampling-period-not-hardcoded test (which reads `templates/index.html` / `static/app.js`
/ `cli.py`, none of which exist yet in this phase) are intentionally not carried over
here -- they belong with `aipt/export/` and `aipt/web/` once those land.
"""

import json
import socket
import subprocess

import pytest

from aipt.core import cwnd


@pytest.fixture(autouse=True)
def _fresh_probe(monkeypatch):
    """available() caches. Every test here changes what the answer should be."""
    cwnd.reset_capability_cache()
    monkeypatch.delenv("TRAFFIC_CWND_DISABLE", raising=False)
    monkeypatch.delenv("TRAFFIC_CWND_BIN", raising=False)
    yield
    cwnd.reset_capability_cache()


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# --- capability reporting ---------------------------------------------------

def test_disable_knob_is_honoured_and_named(monkeypatch):
    """An operator who turned it off has to be told which knob did it -- otherwise the
    empty column looks like a broken box."""
    monkeypatch.setenv("TRAFFIC_CWND_DISABLE", "1")
    ok, reason = cwnd.available()
    assert ok is False
    assert "TRAFFIC_CWND_DISABLE" in reason


def test_non_linux_says_so(monkeypatch):
    monkeypatch.setattr(cwnd.sys, "platform", "darwin")
    ok, reason = cwnd.available()
    assert ok is False
    assert "Linux-only" in reason


def test_helper_that_runs_but_says_nothing_is_unavailable(monkeypatch, tmp_path):
    """The failure this probe exists for: the binary is present and exits 0, but
    netlink is blocked, so it produces no trailer. Stat-ing the file would have called
    that ready and discovered the truth mid-run."""
    binary = tmp_path / "cwnd_monitor"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setenv("TRAFFIC_CWND_BIN", str(binary))
    monkeypatch.setattr(cwnd.sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc(0, "", ""))

    ok, reason = cwnd.available()
    assert ok is False
    assert "netlink" in reason


def test_helper_with_a_trailer_is_available(monkeypatch, tmp_path):
    binary = tmp_path / "cwnd_monitor"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setenv("TRAFFIC_CWND_BIN", str(binary))
    monkeypatch.setattr(cwnd.sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: _Proc(0, '{"type":"end","ticks":2}\n', ""))

    ok, reason = cwnd.available()
    assert ok is True
    assert str(binary) in reason


def test_probe_is_cached_until_reset(monkeypatch, tmp_path):
    """A config endpoint may ask on every page load; the answer only changes on a
    rebuild."""
    binary = tmp_path / "cwnd_monitor"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setenv("TRAFFIC_CWND_BIN", str(binary))
    monkeypatch.setattr(cwnd.sys, "platform", "linux")

    calls = []

    def counted(*a, **k):
        calls.append(a)
        return _Proc(0, '{"type":"end"}\n', "")

    monkeypatch.setattr(subprocess, "run", counted)
    cwnd.available()
    cwnd.available()
    assert len(calls) == 1

    cwnd.reset_capability_cache()
    cwnd.available()
    assert len(calls) == 2


def test_interval_default_and_override(monkeypatch):
    """2 ms, chosen against the RTT of the paths this measures rather than against the
    idle gap. A 10 ms default stepped over the reset entirely on a 3 ms path."""
    assert cwnd.interval_ms() == 2
    monkeypatch.setenv("TRAFFIC_CWND_INTERVAL_MS", "25")
    assert cwnd.interval_ms() == 25
    # Nonsense falls back rather than raising: a bad knob must not kill a run.
    monkeypatch.setenv("TRAFFIC_CWND_INTERVAL_MS", "banana")
    assert cwnd.interval_ms() == 2
    monkeypatch.setenv("TRAFFIC_CWND_INTERVAL_MS", "0")
    assert cwnd.interval_ms() == 2


# --- reset counting -----------------------------------------------------------

def _s(local, cwnd_val, ca="open", t=0.0, idle=0):
    return {"local": local, "snd_cwnd": cwnd_val, "ca_state": ca,
            "t_ms": t, "last_data_sent_ms": idle}


def test_a_grown_window_falling_back_to_ten_counts_as_a_reset():
    samples = [_s("a:1", 10, t=0), _s("a:1", 64, t=10), _s("a:1", 10, t=20, idle=900)]
    out = cwnd.idle_resets(samples)
    assert out["idle_resets"] == 1
    assert out["peak_cwnd"] == 64
    assert out["final_cwnd"] == 10
    assert out["reset_events"][0]["from"] == 64
    assert out["reset_events"][0]["idle_ms"] == 900


def test_a_window_that_never_grew_is_not_a_reset():
    """Starting at 10 and staying at 10 is a connection that never earned a window,
    not one that lost it."""
    samples = [_s("a:1", 10), _s("a:1", 10), _s("a:1", 10)]
    assert cwnd.idle_resets(samples)["idle_resets"] == 0


def test_a_collapse_during_loss_recovery_is_not_counted():
    """The distinction the whole measurement turns on. A window that fell while the
    kernel was in recovery fell because of loss, and folding that in would let a bad
    link look like the idle-reset problem."""
    samples = [_s("a:1", 80), _s("a:1", 10, ca="recovery")]
    assert cwnd.idle_resets(samples)["idle_resets"] == 0


def test_resets_are_counted_per_socket():
    """Two sockets interleaved in one stream. Comparing each sample against whatever
    came last -- rather than against the same socket's previous value -- would invent
    a reset every time the stream alternated."""
    samples = [_s("a:1", 60), _s("b:2", 10), _s("a:1", 60), _s("b:2", 10)]
    assert cwnd.idle_resets(samples)["idle_resets"] == 0


def test_repeated_idle_gaps_count_once_each():
    """Multi-turn: cwnd grows, resets, grows, resets -- each turn boundary is one
    event, which is exactly what a multi-turn conversation produces."""
    samples = [_s("a:1", 40), _s("a:1", 10), _s("a:1", 40), _s("a:1", 10)]
    assert cwnd.idle_resets(samples)["idle_resets"] == 2


def test_no_samples_is_zero_not_an_error():
    out = cwnd.idle_resets([])
    assert out == {"idle_resets": 0, "reset_events": [], "peak_cwnd": 0,
                   "final_cwnd": 0}


# --- the reader thread ---------------------------------------------------------

class _FakeProc:
    """Stands in for the helper: hands its NDJSON over as a file object would."""

    def __init__(self, lines):
        import io
        self.stdout = io.StringIO("".join(lines))
        self.stderr = io.StringIO("")


def test_reader_keeps_samples_and_ignores_junk():
    mon = cwnd.Monitor("openai:stateless:bytes", "api.openai.com")
    mon.proc = _FakeProc([
        '{"type":"meta","interval_ms":10}\n',
        '{"type":"sample","local":"1.1.1.1:1","snd_cwnd":10,"ca_state":"open"}\n',
        'not json at all\n',
        '\n',
        '{"type":"sample","local":"1.1.1.1:1","snd_cwnd":40,"ca_state":"open"}\n',
        '{"type":"end","ticks":2,"seconds":0.02,"dumps":1,"exact_queries":1,'
        '"tracked":1}\n',
    ])
    mon._drain()

    assert len(mon.samples) == 2
    assert mon.meta["interval_ms"] == 10
    assert mon.end["ticks"] == 2

    mon.proc = None
    out = mon.result()
    assert out["sample_count"] == 2
    assert out["peak_cwnd"] == 40
    assert out["sockets"] == ["1.1.1.1:1"]
    assert out["label"] == "openai:stateless:bytes"
    assert out["error"] == ""
    # dumps/exact_queries/tracked instrumentation is preserved from the helper's
    # trailer -- how the ticks were paid for, and whether the run stayed on the
    # cheap (exact-query) path instead of paying for a full table walk each tick.
    assert out["dumps"] == 1
    assert out["exact_queries"] == 1
    assert out["tracked"] == 1


def test_reader_stops_keeping_samples_at_the_cap_and_says_so(monkeypatch):
    """Truncation is reported, not silent. A capped series that claimed to be complete
    would put a reset outside the file and call it absent."""
    monkeypatch.setenv("TRAFFIC_CWND_MAX_SAMPLES", "2")
    mon = cwnd.Monitor("openai:stateless", "api.openai.com")
    mon.proc = _FakeProc(
        ['{"type":"sample","local":"a:1","snd_cwnd":%d,"ca_state":"open"}\n' % n
         for n in (10, 20, 30, 40)])
    mon._drain()
    mon.proc = None

    assert len(mon.samples) == 2
    assert mon.result()["truncated"] is True


def test_monitor_that_could_not_start_reports_it_rather_than_raising():
    """Best-effort, like capture: a monitor that raised would turn a missing column
    into a failed run."""
    mon = cwnd.Monitor("openai:stateless", "api.openai.com")
    mon.error = "monitor would not start: boom"
    out = mon.result()
    assert out["error"].startswith("monitor would not start")
    assert out["sample_count"] == 0


def test_stop_is_idempotent():
    mon = cwnd.Monitor("conv-1", "127.0.0.1")
    mon.stop()
    mon.stop()


def test_announce_with_no_proc_does_not_raise():
    mon = cwnd.Monitor("conv-1", "127.0.0.1")
    s1, s2 = socket.socketpair()
    try:
        mon.announce(s1)  # proc is None; must be a no-op, not an exception
    finally:
        s1.close()
        s2.close()


def test_announce_writes_a_track_line_for_the_matching_port(monkeypatch):
    """announce(sock) is the merged interface's replacement for the old shared
    connect-watcher registry: the caller pushes a socket at the monitor directly,
    right after connecting, instead of subscribing through core.wire."""
    mon = cwnd.Monitor("conv-1", "127.0.0.1", port=9999)

    class _FakeStdin:
        def __init__(self):
            self.lines = []

        def write(self, s):
            self.lines.append(s)

        def flush(self):
            pass

    class _FakeProcWithStdin:
        def __init__(self):
            self.stdin = _FakeStdin()

    mon.proc = _FakeProcWithStdin()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    mon.port = port  # match the ephemeral listening port for this test

    client = socket.create_connection(("127.0.0.1", port))
    try:
        mon.announce(client)
        assert mon.announced == 1
        assert len(mon.proc.stdin.lines) == 1
        assert mon.proc.stdin.lines[0].startswith("track ")
    finally:
        client.close()
        srv.close()


# --- B12: adaptive sampling period (DESIGN.md 4.9) ---------------------------

def test_interval_from_rtt_normal_path():
    """A 10 ms path samples about 5 times across a burst on the order of one
    RTT -- the same ratio the fixed 2 ms default was picked against for a 10 ms
    recovery window."""
    interval, reason = cwnd.interval_from_rtt(10.0)
    assert interval == 2
    assert reason == "adaptive:rtt=10.0ms"


def test_interval_from_rtt_scales_with_rtt():
    interval, reason = cwnd.interval_from_rtt(100.0)
    assert interval == 20
    assert reason == "adaptive:rtt=100.0ms"


def test_interval_from_rtt_very_short_path_hits_the_floor():
    """Loopback / same-container RTT (well under 1 ms) divided by K is below the
    netlink-tick floor, so the result is clamped and flagged rather than reported
    as a real measurement."""
    interval, reason = cwnd.interval_from_rtt(0.05)
    assert interval == 1
    assert reason == "floor_clamped"


def test_interval_from_rtt_custom_floor():
    interval, reason = cwnd.interval_from_rtt(0.05, min_interval_ms=5)
    assert interval == 5
    assert reason == "floor_clamped"


def test_interval_from_rtt_custom_k():
    """A caller wanting more samples per burst (smaller K denominator... larger K
    means MORE samples since interval = rtt/k) can override K directly."""
    interval, reason = cwnd.interval_from_rtt(10.0, k=10.0)
    assert interval == 1
    assert reason == "adaptive:rtt=10.0ms"


def test_interval_from_rtt_no_hint_falls_back_to_fixed():
    interval, reason = cwnd.interval_from_rtt(None)
    assert interval == cwnd.DEFAULT_INTERVAL_MS
    assert reason == "fixed"

    interval, reason = cwnd.interval_from_rtt(0)
    assert interval == cwnd.DEFAULT_INTERVAL_MS
    assert reason == "fixed"

    interval, reason = cwnd.interval_from_rtt(-5)
    assert interval == cwnd.DEFAULT_INTERVAL_MS
    assert reason == "fixed"


def test_monitor_without_rtt_hint_is_unchanged_from_before_b12():
    """Full backward-compat guarantee: no rtt_hint_ms -> identical behaviour to
    before this feature existed."""
    mon = cwnd.Monitor("openai:stateless:bytes", "api.openai.com")
    assert mon.interval == cwnd.interval_ms()
    assert mon.interval_reason == "fixed"
    assert mon.measurement_confidence == "high"

    out = mon.result()
    assert out["interval_ms"] == cwnd.interval_ms()
    assert out["interval_reason"] == "fixed"
    assert out["measurement_confidence"] == "high"


def test_monitor_with_explicit_interval_still_wins_over_rtt_hint():
    """An explicit interval is a caller that knows exactly what it wants; it must
    not be second-guessed by a simultaneously-passed rtt_hint_ms."""
    mon = cwnd.Monitor("conv-1", "127.0.0.1", interval=7, rtt_hint_ms=100.0)
    assert mon.interval == 7
    assert mon.interval_reason == "fixed"
    assert mon.measurement_confidence == "high"


def test_monitor_with_rtt_hint_computes_adaptive_interval():
    mon = cwnd.Monitor("mock:arm:bytes", "127.0.0.1", rtt_hint_ms=10.0)
    assert mon.interval == 2
    assert mon.interval_reason == "adaptive:rtt=10.0ms"
    assert mon.measurement_confidence == "high"

    out = mon.result()
    assert out["interval_ms"] == 2
    assert out["interval_reason"] == "adaptive:rtt=10.0ms"
    assert out["measurement_confidence"] == "high"


def test_monitor_with_short_rtt_hint_clamps_and_degrades_confidence():
    mon = cwnd.Monitor("mock:arm:bytes", "127.0.0.1", rtt_hint_ms=0.05)
    assert mon.interval == 1
    assert mon.interval_reason == "floor_clamped"
    assert mon.measurement_confidence == "degraded"

    out = mon.result()
    assert out["interval_reason"] == "floor_clamped"
    assert out["measurement_confidence"] == "degraded"
