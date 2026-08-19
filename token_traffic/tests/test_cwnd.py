"""The congestion-window monitor: capability reporting, sample parsing, reset counting.

What is tested here is everything that decides what a reader is told, because that is
where a wrong answer does damage. `idle_resets` in particular: it is the number the
feature exists to produce, and the difference between "the window was reset after idle"
and "the window collapsed because of loss" is one field. Getting that wrong would let a
lossy link masquerade as the finding -- or hide it.

What is not tested here is netlink itself. Reading real congestion state needs real
sockets, and `tests/test_cwnd_live.py` does exactly that, against a loopback server,
where it can be checked against `ss`. This file fakes the helper out entirely so it
runs on a box with no compiler and no netlink -- a test that silently skips is worse
than no test.
"""

import json
import subprocess

import pytest

from core import cwnd
from core import export


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


# --- capability reporting -------------------------------------------------

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
    """/api/config asks on every page load; the answer only changes on a rebuild."""
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
    assert cwnd.interval_ms() == 10
    monkeypatch.setenv("TRAFFIC_CWND_INTERVAL_MS", "25")
    assert cwnd.interval_ms() == 25
    # Nonsense falls back rather than raising: a bad knob must not kill a run.
    monkeypatch.setenv("TRAFFIC_CWND_INTERVAL_MS", "banana")
    assert cwnd.interval_ms() == 10
    monkeypatch.setenv("TRAFFIC_CWND_INTERVAL_MS", "0")
    assert cwnd.interval_ms() == 10


# --- reset counting -------------------------------------------------------

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
    samples = [_s("a:1", 40), _s("a:1", 10), _s("a:1", 40), _s("a:1", 10)]
    assert cwnd.idle_resets(samples)["idle_resets"] == 2


def test_no_samples_is_zero_not_an_error():
    out = cwnd.idle_resets([])
    assert out == {"idle_resets": 0, "reset_events": [], "peak_cwnd": 0,
                   "final_cwnd": 0}


# --- the reader thread ----------------------------------------------------

class _FakeProc:
    """Stands in for the helper: hands its NDJSON over as a file object would."""

    def __init__(self, lines):
        import io
        self.stdout = io.StringIO("".join(lines))
        self.stderr = io.StringIO("")


def test_reader_keeps_samples_and_ignores_junk():
    mon = cwnd.Monitor("openai", "stateless", "api.openai.com", kind="bytes")
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
    assert out["label"] == "openai:stateless:bytes"
    assert out["error"] == ""


def test_reader_stops_keeping_samples_at_the_cap_and_says_so(monkeypatch):
    """Truncation is reported, not silent. A capped series that claimed to be complete
    would put a reset outside the file and call it absent."""
    monkeypatch.setenv("TRAFFIC_CWND_MAX_SAMPLES", "2")
    mon = cwnd.Monitor("openai", "stateless", "api.openai.com")
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
    mon = cwnd.Monitor("openai", "stateless", "api.openai.com")
    mon.error = "monitor would not start: boom"
    out = mon.result()
    assert out["error"].startswith("monitor would not start")
    assert out["sample_count"] == 0


def test_stop_is_idempotent():
    mon = cwnd.Monitor("openai", "stateless", "api.openai.com")
    mon.stop()
    mon.stop()


# --- CSV ------------------------------------------------------------------

def _run_doc():
    return {
        "cwnd": {
            "openai:stateless": {
                "bytes": {
                    "provider": "openai", "arm": "stateless", "kind": "bytes",
                    "host": "api.openai.com", "ips": ["1.2.3.4"], "interval_ms": 10,
                    "sample_count": 2, "ticks": 2, "seconds": 0.02,
                    "sockets": ["10.0.0.1:5000"], "peak_cwnd": 40, "final_cwnd": 10,
                    "idle_resets": 1, "truncated": False, "error": "",
                    "samples": [
                        {"t_ms": 0.0, "local": "10.0.0.1:5000",
                         "remote": "1.2.3.4:443", "snd_cwnd": 40,
                         "snd_ssthresh": 32, "rtt_us": 34065, "ca_state": "open"},
                        {"t_ms": 10.0, "local": "10.0.0.1:5000",
                         "remote": "1.2.3.4:443", "snd_cwnd": 10,
                         "snd_ssthresh": 32, "rtt_us": 34120, "ca_state": "open"},
                    ],
                }
            }
        }
    }


def test_cwnd_csv_has_a_row_per_sample_and_names_the_arm():
    text = export.cwnd_csv(_run_doc())
    lines = text.strip().splitlines()
    header = lines[0].split(",")

    assert lines[0].startswith("provider,arm,kind,host,local,remote,")
    # The three fields the request was actually about, all present and only once each.
    for field in ("snd_cwnd", "snd_ssthresh", "rtt_us"):
        assert header.count(field) == 1
    assert header.count("local") == 1

    assert len(lines) == 3
    assert lines[1].startswith("openai,stateless,bytes,api.openai.com,")
    assert ",40," in lines[1]
    assert ",10," in lines[2]


def test_cwnd_summary_csv_carries_the_reset_count():
    text = export.cwnd_summary_csv(_run_doc())
    lines = text.strip().splitlines()
    assert len(lines) == 2
    assert "idle_resets" in lines[0]
    row = dict(zip(lines[0].split(","), lines[1].split(",")))
    assert row["idle_resets"] == "1"
    assert row["peak_cwnd"] == "40"
    assert row["final_cwnd"] == "10"
    assert row["sockets"] == "10.0.0.1:5000"


def test_csvs_of_an_unmonitored_run_are_headers_only():
    """Not an empty string: a file with a header says "monitored nothing", which is
    what a run without the flag did."""
    for fn in (export.cwnd_csv, export.cwnd_summary_csv):
        text = fn({"records": []})
        assert len(text.strip().splitlines()) == 1


def test_sample_fields_and_csv_columns_do_not_drift():
    """The C helper, SAMPLE_FIELDS and the CSV header are three lists that have to
    agree. This catches the two that live in Python; the live test catches the third."""
    for field in cwnd.SAMPLE_FIELDS:
        assert field in export.CWND_COLUMNS
