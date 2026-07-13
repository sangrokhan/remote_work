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
    #
    # Every arm now also runs a prep stage *before* its capture window opens (a
    # no-op for everything except cached), and the connection is reset right after
    # prep, before the window opens -- otherwise prep's own leftover connection (or
    # its FIN) would land inside the steady capture.
    log = _install_ordered(monkeypatch)
    experiment.run_comparison("gemini-3.1-flash-lite", turns=1,
                              arms=["stateless", "cached"],
                              want_capture=True, timestamp="2026-07-13T00:00:00")
    assert log == [
        "reset",                                  # drop anything pooled earlier
        "reset", "open:stateless", "reset", "close:stateless",
        "reset", "open:cached", "reset", "close:cached",
    ]


def test_arms_still_get_a_fresh_connection_without_capture(monkeypatch):
    # No capture, but each arm must still open its own TCP connection, or its wire
    # bytes are not attributable to it. Two resets per arm now: one after prep
    # (before the steady stage), one after the steady stage itself.
    log = _install_ordered(monkeypatch)
    experiment.run_comparison("gemini-3.1-flash-lite", turns=1,
                              arms=["stateless", "cached"])
    assert log.count("reset") == 5   # once before the loop, twice per arm


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


# --- the cached arm's pcap/wall_ms must cover only the steady queries ------

def _install_full_ordered(monkeypatch):
    """Like _install_ordered, but also logs create_cache/delete_cache/call_gemini,
    so the whole prep -> capture -> teardown sequence can be pinned in one list."""
    log = _install_ordered(monkeypatch)
    real_create_cache = experiment.create_cache
    real_delete_cache = experiment.delete_cache
    real_call_gemini = experiment.call_gemini

    def fake_create_cache(*a, **k):
        log.append("create_cache")
        return real_create_cache(*a, **k)

    def fake_delete_cache(*a, **k):
        log.append("delete_cache")
        return real_delete_cache(*a, **k)

    def fake_call_gemini(*a, **k):
        log.append("query")
        return real_call_gemini(*a, **k)

    monkeypatch.setattr(experiment, "create_cache", fake_create_cache)
    monkeypatch.setattr(experiment, "delete_cache", fake_delete_cache)
    monkeypatch.setattr(experiment, "call_gemini", fake_call_gemini)
    return log


def test_cached_capture_window_brackets_only_the_steady_queries(monkeypatch):
    # Cache creation (prep) and cache deletion (teardown) must both happen outside
    # the pcap window: create_cache is preparation, delete_cache is teardown, and
    # neither is the thing being measured.
    log = _install_full_ordered(monkeypatch)
    experiment.run_comparison("gemini-3.1-flash-lite", turns=2, arms=["cached"],
                              want_capture=True, timestamp="2026-07-13T00:00:00")

    open_i = log.index("open:cached")
    close_i = log.index("close:cached")
    bracket = log[open_i:close_i + 1]

    create_indices = [i for i, e in enumerate(log) if e == "create_cache"]
    delete_indices = [i for i, e in enumerate(log) if e == "delete_cache"]
    assert create_indices and all(i < open_i for i in create_indices)
    assert delete_indices and all(i > close_i for i in delete_indices)

    # Everything inside the bracket is a query -- no cache create/delete leaks in.
    assert "create_cache" not in bracket
    assert "delete_cache" not in bracket
    assert bracket.count("query") == 2   # the two steady turns


def test_cached_wall_ms_excludes_prep_and_teardown_time(monkeypatch):
    # Make prep and teardown slow (a known, large sleep) and confirm wall_ms only
    # reflects the steady stage, not the whole arm.
    _install_ordered(monkeypatch)  # patches experiment.time.sleep to a no-op

    # Patch time.monotonic to advance a lot during prep/teardown-like calls and
    # only a little during the steady queries, by wrapping call_gemini/create_cache
    # /delete_cache to bump a shared fake clock.
    clock = {"t": 0.0}

    def fake_monotonic():
        return clock["t"]

    def bump(seconds):
        clock["t"] += seconds

    real_create_cache = experiment.create_cache
    real_delete_cache = experiment.delete_cache
    real_call_gemini = experiment.call_gemini

    def slow_create_cache(*a, **k):
        bump(10.0)   # prep is slow
        return real_create_cache(*a, **k)

    def slow_delete_cache(*a, **k):
        bump(10.0)   # teardown is slow
        return real_delete_cache(*a, **k)

    def fast_call_gemini(*a, **k):
        bump(0.01)   # steady queries are fast
        return real_call_gemini(*a, **k)

    monkeypatch.setattr(experiment.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(experiment, "create_cache", slow_create_cache)
    monkeypatch.setattr(experiment, "delete_cache", slow_delete_cache)
    monkeypatch.setattr(experiment, "call_gemini", fast_call_gemini)

    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=2, arms=["cached"])
    assert out["wall_ms"]["cached"] < 100          # far below the ~20s of prep+teardown
    assert clock["t"] >= 20.0                       # the slow stages really did run


def test_cachegen_records_and_steady_shape_are_unchanged(monkeypatch):
    # The restructuring must not change what gets recorded: cachegen records still
    # carry phase="cachegen", steady records are still shaped the same way.
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=2, arms=["cached"])
    gen = [r for r in out["records"] if r["arm"] == "cached" and r["phase"] == "cachegen"]
    steady = [r for r in out["records"] if r["arm"] == "cached" and r["phase"] == "steady"]
    assert len(gen) == 2
    assert all(r.get("cache_id") is not None or r.get("skipped") for r in gen)
    assert [r["turn"] for r in steady] == [1, 2]
    assert steady[0]["cache_id"] is None
    assert steady[1]["cache_id"] == gen[0]["cache_id"]


def test_other_arms_pcap_and_wall_ms_are_unaffected(monkeypatch):
    # stateless/interaction/interaction_stateless have no-op prep and teardown: their
    # capture must still bracket the whole arm, exactly as before.
    log = _install_ordered(monkeypatch)
    experiment.run_comparison("gemini-3.1-flash-lite", turns=1,
                              arms=["stateless"],
                              want_capture=True, timestamp="2026-07-13T00:00:00")
    assert log == ["reset", "reset", "open:stateless", "reset", "close:stateless"]
