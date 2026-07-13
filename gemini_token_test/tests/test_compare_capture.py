"""Per-arm packet capture.

The socket-level wire counter and tcpdump measure the same thing two ways; one
pcap per arm is what lets them be cross-checked, and it is the only artifact that
shows retransmits and TLS overhead the counter can't see.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import capture
import experiment


class _FakeCapture:
    """Stands in for tcpdump: records what would have been captured."""

    started: list = []

    def __init__(self, timestamp, mode="stateless", interface=None):
        self.mode = mode
        self.timestamp = timestamp

    def __enter__(self):
        _FakeCapture.started.append(self.mode)
        return self

    def __exit__(self, *a):
        return False

    def result(self):
        return {"ok": True, "file": f"capture_{self.mode}.pcap", "bytes": 4096}


def _install(monkeypatch):
    _FakeCapture.started = []
    monkeypatch.setenv("GEMINI_MOCK", "1")
    monkeypatch.setattr(experiment.pcap, "Capture", _FakeCapture)
    monkeypatch.setattr(experiment.pcap, "available", lambda: (True, "ready"))


def test_capture_is_off_unless_asked(monkeypatch):
    _install(monkeypatch)
    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=1,
                                    arms=["stateless", "cached"])
    assert _FakeCapture.started == []
    assert out.get("pcaps") == {}


def test_one_capture_per_arm(monkeypatch):
    _install(monkeypatch)
    arms = ["stateless", "cached", "interaction"]
    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=1, arms=arms,
                                    want_capture=True, timestamp="2026-07-13T00:00:00")
    assert _FakeCapture.started == arms
    assert set(out["pcaps"]) == set(arms)
    assert out["pcaps"]["cached"]["bytes"] == 4096


def test_capture_targets_the_developer_api_host():
    # A capture filtered on the Vertex host would record zero packets: every arm
    # now talks to generativelanguage.
    cap = capture.Capture("2026-07-13T00:00:00", mode="stateless")
    assert cap.host == "generativelanguage.googleapis.com"


def test_capture_accepts_every_arm_name():
    for arm in ("stateless", "cached", "interaction", "nocontext"):
        assert capture.Capture("2026-07-13T00:00:00", mode=arm).mode == arm
