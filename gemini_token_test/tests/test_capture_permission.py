"""available() has to mean "a capture will actually work".

tcpdump being installed says nothing: capturing needs CAP_NET_RAW, and without it
every capture dies with "Operation not permitted". Reporting "ready" on the
strength of the binary existing hands the operator a checkbox that silently
produces nothing, which is worse than not offering it.
"""

import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import capture


def test_missing_binary_is_unavailable(monkeypatch):
    monkeypatch.delenv("PCAP_DISABLE", raising=False)
    monkeypatch.setattr(capture, "tcpdump_path", lambda: None)
    ok, reason = capture.available()
    assert ok is False
    assert "not installed" in reason


def test_binary_without_raw_socket_capability_is_unavailable(monkeypatch):
    monkeypatch.delenv("PCAP_DISABLE", raising=False)
    monkeypatch.setattr(capture, "tcpdump_path", lambda: "/usr/bin/tcpdump")
    monkeypatch.setattr(capture, "can_raw_capture", lambda: False)
    ok, reason = capture.available()
    assert ok is False
    # The reason must be actionable: name the capability and how to grant it.
    assert "NET_RAW" in reason
    assert "setcap" in reason


def test_binary_with_capability_is_available(monkeypatch):
    monkeypatch.delenv("PCAP_DISABLE", raising=False)
    monkeypatch.setattr(capture, "tcpdump_path", lambda: "/usr/bin/tcpdump")
    monkeypatch.setattr(capture, "can_raw_capture", lambda: True)
    assert capture.available() == (True, "ready")


def test_disable_flag_still_wins(monkeypatch):
    monkeypatch.setenv("PCAP_DISABLE", "1")
    monkeypatch.setattr(capture, "can_raw_capture", lambda: True)
    ok, _ = capture.available()
    assert ok is False


def test_raw_capture_probe_reports_false_without_permission(monkeypatch):
    def deny(*a, **k):
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(socket, "socket", deny)
    assert capture.can_raw_capture() is False


def test_raw_capture_probe_closes_the_socket_it_opens(monkeypatch):
    closed = {"n": 0}

    class _Sock:
        def close(self):
            closed["n"] += 1

    monkeypatch.setattr(socket, "socket", lambda *a, **k: _Sock())
    assert capture.can_raw_capture() is True
    assert closed["n"] == 1
