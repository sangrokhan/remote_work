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


# --- each pcap must be a self-contained conversation -----------------------

class _OrderedCapture(_FakeCapture):
    """Records capture open/close against the session resets, so the ordering can
    be asserted."""

    log: list = []

    def __enter__(self):
        _OrderedCapture.log.append(f"open:{self.mode}")
        return self

    def __exit__(self, *a):
        _OrderedCapture.log.append(f"close:{self.mode}")
        return False


def _install_ordered(monkeypatch):
    _OrderedCapture.log = []
    monkeypatch.setenv("GEMINI_MOCK", "1")
    monkeypatch.setattr(experiment.pcap, "Capture", _OrderedCapture)
    monkeypatch.setattr(experiment, "reset_session",
                        lambda: _OrderedCapture.log.append("reset"))
    monkeypatch.setattr(experiment.time, "sleep", lambda s: None)
    return _OrderedCapture.log


def test_each_arms_fin_lands_in_its_own_pcap(monkeypatch):
    # The session must be closed *inside* the arm's capture window. Closing it at
    # the start of the next arm instead puts this arm's FIN in the next arm's pcap,
    # which is exactly what a stray FIN from "the previous test" is.
    log = _install_ordered(monkeypatch)
    experiment.run_comparison("gemini-3.1-flash-lite", turns=1,
                              arms=["stateless", "cached"],
                              want_capture=True, timestamp="2026-07-13T00:00:00")
    assert log == [
        "reset",                                  # drop anything pooled earlier
        "open:stateless", "reset", "close:stateless",
        "open:cached", "reset", "close:cached",
    ]


def test_arms_still_get_a_fresh_connection_without_capture(monkeypatch):
    # No capture, but each arm must still open its own TCP connection, or its wire
    # bytes are not attributable to it.
    log = _install_ordered(monkeypatch)
    experiment.run_comparison("gemini-3.1-flash-lite", turns=1,
                              arms=["stateless", "cached"])
    assert log.count("reset") == 3   # once before the loop, once after each arm


# --- the pcap must name the arm it captured -------------------------------

def test_every_arm_name_survives_into_its_filename():
    # The label alphabet excluded '_', so `interaction_inline` fell through to the
    # "stateless" fallback: a 4-arm capture wrote two files called
    # capture_stateless_*, one of which was the inline arm. A pcap that lies about
    # which arm it holds defeats the only thing a per-arm pcap is for.
    for arm in experiment.COMPARE_ARMS:
        cap = capture.Capture("2026-07-13T00:00:00", mode=arm)
        assert cap.mode == arm, f"{arm} was relabelled {cap.mode}"
        assert f"capture_{arm}_" in cap.path.name


def test_generated_names_are_downloadable():
    for arm in experiment.COMPARE_ARMS:
        cap = capture.Capture("2026-07-13T00:00:00", mode=arm)
        assert capture._SAFE_NAME.match(cap.path.name), cap.path.name


def test_a_label_that_cannot_be_named_is_an_error_not_a_silent_rename():
    # Falling back to "stateless" is how the mislabelling hid. Refuse instead.
    cap = capture.Capture("2026-07-13T00:00:00", mode="../etc/passwd")
    assert cap.error
    assert cap.result()["ok"] is False


def test_traversal_is_still_rejected_by_the_download_guard():
    assert capture.safe_pcap_path("../../etc/passwd") is None
    assert capture.safe_pcap_path("capture_a/b_2026_0000000000000000.pcap") is None
