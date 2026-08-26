"""tests/gateway/test_forwarding.py -- aipt.gateway.forwarding tests
(DESIGN.md 4.7 확정 설계 1: L3 IP forwarding availability check). Reads a
scratch file standing in for /proc/sys/net/ipv4/ip_forward -- never touches
the real sysctl path, matching the rest of tests/gateway/'s "never actually
run privileged operations" posture."""

import pytest

from aipt.gateway import forwarding


def _write(tmp_path, content):
    p = tmp_path / "ip_forward"
    p.write_text(content)
    return str(p)


def test_read_ip_forward_true_when_one(tmp_path):
    path = _write(tmp_path, "1\n")
    ok, reason = forwarding.read_ip_forward(path)
    assert ok is True
    assert reason == "ready"


def test_read_ip_forward_false_when_zero(tmp_path):
    path = _write(tmp_path, "0\n")
    ok, reason = forwarding.read_ip_forward(path)
    assert ok is False
    assert "ip_forward" in reason
    assert "sysctl" in reason or "docker-compose" in reason


def test_read_ip_forward_false_when_missing_file(tmp_path):
    path = str(tmp_path / "does_not_exist")
    ok, reason = forwarding.read_ip_forward(path)
    assert ok is False
    assert "could not be read" in reason


def test_available_matches_read_ip_forward(tmp_path):
    path = _write(tmp_path, "1")
    assert forwarding.available(path) == forwarding.read_ip_forward(path)


def test_status_returns_dict_shape(tmp_path):
    path = _write(tmp_path, "1")
    result = forwarding.status(path)
    assert result == {"ok": True, "reason": "ready"}


def test_status_dict_shape_when_disabled(tmp_path):
    path = _write(tmp_path, "0")
    result = forwarding.status(path)
    assert result["ok"] is False
    assert "reason" in result


def test_default_path_is_standard_proc_sysctl():
    assert forwarding.IP_FORWARD_PATH == "/proc/sys/net/ipv4/ip_forward"


def test_read_ip_forward_permission_error_reports_unreadable(tmp_path, monkeypatch):
    path = _write(tmp_path, "1")

    def fake_open(*a, **k):
        raise PermissionError("denied")

    monkeypatch.setattr("builtins.open", fake_open)
    ok, reason = forwarding.read_ip_forward(path)
    assert ok is False
    assert "could not be read" in reason
