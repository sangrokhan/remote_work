"""capture: tcpdump packet capture around one conversation run.

Adapted from token_traffic/core/capture.py, simplified to one capture per
run (this project has one connection, not multiple provider/arm combos).
NIC offload toggling is intentionally left out here: this lab already runs
one connection at a time on loopback/docker-bridge interfaces where segment
counts are not the primary evidence (cwnd comes from netlink, not the pcap).
"""

import re
import secrets
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest

from tcp_congestion import capture


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture(autouse=True)
def _fresh_cache():
    capture.reset_capability_cache()
    yield
    capture.reset_capability_cache()


def test_tcpdump_path_returns_none_when_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert capture.tcpdump_path() is None


def test_available_false_when_tcpdump_missing(monkeypatch):
    monkeypatch.setattr(capture, "tcpdump_path", lambda: None)
    ok, reason = capture.available()
    assert ok is False
    assert "tcpdump" in reason.lower()


def test_available_false_when_disabled(monkeypatch):
    monkeypatch.setenv("TRAFFIC_PCAP_DISABLE", "1")
    ok, reason = capture.available()
    assert ok is False
    assert "disabled" in reason.lower()


def test_safe_pcap_path_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFIC_PCAP_DIR", str(tmp_path))
    assert capture.safe_pcap_path("../../etc/passwd") is None
    assert capture.safe_pcap_path("not_a_valid_name.pcap") is None


def test_safe_pcap_path_accepts_generated_name(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFIC_PCAP_DIR", str(tmp_path))
    token = secrets.token_hex(8)
    name = f"capture_conversation_2026-01-01_{token}.pcap"
    (tmp_path / name).write_bytes(b"\x00" * 10)
    assert capture.safe_pcap_path(name) == tmp_path / name


def test_parse_tcpdump_stats():
    text = ("123 packets captured\n130 packets received by filter\n"
            "7 packets dropped by kernel\n")
    stats = capture._parse_tcpdump_stats(text)
    assert stats["captured"] == 123
    assert stats["received_by_filter"] == 130
    assert stats["dropped_by_kernel"] == 7


def test_capture_rejects_unsafe_label(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFIC_PCAP_DIR", str(tmp_path))
    cap = capture.Capture(timestamp="2026-01-01T00:00:00", label="bad label!",
                          host="127.0.0.1")
    assert cap.error


def test_capture_result_reports_error_without_touching_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFIC_PCAP_DIR", str(tmp_path))
    cap = capture.Capture(timestamp="2026-01-01T00:00:00", label="bad label!",
                          host="127.0.0.1")
    result = cap.result()
    assert result["ok"] is False
    assert result["error"]


@pytest.mark.skipif(not shutil.which("tcpdump"), reason="tcpdump not installed")
def test_capture_produces_a_pcap_with_real_traffic(tmp_path, monkeypatch):
    ok, reason = capture.available()
    if not ok:
        pytest.skip(reason)
    monkeypatch.setenv("TRAFFIC_PCAP_DIR", str(tmp_path))

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    import threading
    def serve():
        conn, _ = srv.accept()
        conn.recv(1024)
        conn.close()
    threading.Thread(target=serve, daemon=True).start()

    cap = capture.Capture(timestamp="2026-01-01T00:00:00", label="conversation",
                          host="127.0.0.1", port=port, interface="lo")
    cap.__enter__()
    try:
        time.sleep(0.3)
        client = socket.create_connection(("127.0.0.1", port))
        client.sendall(b"hello")
        time.sleep(0.3)
        client.close()
    finally:
        cap.__exit__(None, None, None)
    srv.close()

    result = cap.result()
    assert result["ok"] is True
    assert result["bytes"] > 0
