"""cwnd: continuous congestion-window monitor -- capability, parsing, reset counting.

Adapted from token_traffic/tests/test_cwnd.py, with core.export references
removed (this project's CSV export is separate) and provider/arm/kind fields
dropped from Monitor (this project has one connection per experiment, not
per-arm).
"""

import json
import re
import subprocess

import pytest

from tcp_congestion import cwnd


@pytest.fixture(autouse=True)
def _fresh_probe(monkeypatch):
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


def test_disable_knob_is_honoured(monkeypatch):
    monkeypatch.setenv("TRAFFIC_CWND_DISABLE", "1")
    ok, reason = cwnd.available()
    assert ok is False
    assert "TRAFFIC_CWND_DISABLE" in reason


def test_non_linux_says_so(monkeypatch):
    monkeypatch.setattr(cwnd.sys, "platform", "darwin")
    ok, reason = cwnd.available()
    assert ok is False
    assert "Linux-only" in reason


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


def test_interval_default_and_override(monkeypatch):
    assert cwnd.interval_ms() == 2
    monkeypatch.setenv("TRAFFIC_CWND_INTERVAL_MS", "25")
    assert cwnd.interval_ms() == 25
    monkeypatch.setenv("TRAFFIC_CWND_INTERVAL_MS", "banana")
    assert cwnd.interval_ms() == 2


# --- reset counting ---------------------------------------------------------

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


def test_a_collapse_during_loss_recovery_is_not_counted():
    samples = [_s("a:1", 80), _s("a:1", 10, ca="recovery")]
    assert cwnd.idle_resets(samples)["idle_resets"] == 0


def test_repeated_idle_gaps_count_once_each():
    """Multi-turn: cwnd grows, resets, grows, resets — each turn boundary is
    one event, which is exactly what a multi-turn conversation produces."""
    samples = [_s("a:1", 40), _s("a:1", 10), _s("a:1", 40), _s("a:1", 10)]
    assert cwnd.idle_resets(samples)["idle_resets"] == 2


def test_no_samples_is_zero_not_an_error():
    out = cwnd.idle_resets([])
    assert out == {"idle_resets": 0, "reset_events": [], "peak_cwnd": 0,
                   "final_cwnd": 0}


# --- the reader thread -------------------------------------------------------

class _FakeProc:
    def __init__(self, lines):
        import io
        self.stdout = io.StringIO("".join(lines))
        self.stderr = io.StringIO("")


def test_reader_keeps_samples_and_ignores_junk():
    mon = cwnd.Monitor("conv-1", "127.0.0.1")
    mon.proc = _FakeProc([
        '{"type":"meta","interval_ms":10}\n',
        '{"type":"sample","local":"1.1.1.1:1","snd_cwnd":10,"ca_state":"open"}\n',
        'not json at all\n',
        '\n',
        '{"type":"sample","local":"1.1.1.1:1","snd_cwnd":40,"ca_state":"open"}\n',
        '{"type":"end","ticks":2,"seconds":0.02}\n',
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
    assert out["label"] == "conv-1"
    assert out["error"] == ""


def test_monitor_that_could_not_start_reports_it():
    mon = cwnd.Monitor("conv-1", "127.0.0.1")
    mon.error = "monitor would not start: boom"
    out = mon.result()
    assert out["error"].startswith("monitor would not start")
    assert out["sample_count"] == 0


def test_stop_is_idempotent():
    mon = cwnd.Monitor("conv-1", "127.0.0.1")
    mon.stop()
    mon.stop()


def test_announce_with_no_proc_does_not_raise():
    import socket
    mon = cwnd.Monitor("conv-1", "127.0.0.1")
    s1, s2 = socket.socketpair()
    try:
        mon.announce(s1)  # proc is None; must be a no-op, not an exception
    finally:
        s1.close()
        s2.close()
