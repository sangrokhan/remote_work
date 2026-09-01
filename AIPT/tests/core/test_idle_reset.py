"""tests/core/test_idle_reset.py -- aipt.core.idle_reset tests (idle-reset
TTFT experiment control, 2026-09-01 ooo interview). Reads/writes a scratch
file standing in for /proc/sys/net/ipv4/tcp_slow_start_after_idle -- never
touches the real sysctl path, matching tests/gateway/test_forwarding.py's
"never actually run privileged operations" posture."""

import pytest

from aipt.core import idle_reset


def _write(tmp_path, content):
    p = tmp_path / "tcp_slow_start_after_idle"
    p.write_text(content)
    return str(p)


def test_read_true_when_one(tmp_path):
    path = _write(tmp_path, "1\n")
    enabled, reason = idle_reset.read(path)
    assert enabled is True
    assert reason == "ready"


def test_read_false_when_zero(tmp_path):
    path = _write(tmp_path, "0\n")
    enabled, reason = idle_reset.read(path)
    assert enabled is False
    assert reason == "ready"


def test_read_none_when_missing_file(tmp_path):
    path = str(tmp_path / "does_not_exist")
    enabled, reason = idle_reset.read(path)
    assert enabled is None
    assert "could not be read" in reason


def test_read_none_on_unexpected_value(tmp_path):
    path = _write(tmp_path, "garbage")
    enabled, reason = idle_reset.read(path)
    assert enabled is None
    assert "unexpected value" in reason


def test_write_then_read_round_trips(tmp_path):
    path = _write(tmp_path, "1")
    ok, reason = idle_reset.write(False, path)
    assert ok is True
    assert reason == "ready"
    enabled, _ = idle_reset.read(path)
    assert enabled is False

    ok, reason = idle_reset.write(True, path)
    assert ok is True
    enabled, _ = idle_reset.read(path)
    assert enabled is True


def test_write_false_when_unwritable(tmp_path, monkeypatch):
    path = _write(tmp_path, "1")
    real_open = open

    def fake_open(file, mode="r", *a, **k):
        if file == path and "w" in mode:
            raise PermissionError("denied")
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)
    ok, reason = idle_reset.write(True, path)
    assert ok is False
    assert "could not be written" in reason


def test_status_returns_dict_shape(tmp_path):
    path = _write(tmp_path, "1")
    result = idle_reset.status(path)
    assert result == {"ok": True, "enabled": True, "reason": "ready"}


def test_status_dict_shape_when_missing(tmp_path):
    path = str(tmp_path / "missing")
    result = idle_reset.status(path)
    assert result["ok"] is False
    assert result["enabled"] is None
    assert "reason" in result


def test_default_path_is_standard_proc_sysctl():
    assert idle_reset.IDLE_RESET_PATH == "/proc/sys/net/ipv4/tcp_slow_start_after_idle"
