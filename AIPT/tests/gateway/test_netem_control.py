"""aipt.gateway.netem_control -- tc command construction, mocked
execution (DESIGN.md 4.7 B9). Never actually runs `tc` -- subprocess.run
is monkeypatched throughout, matching aipt.core.offload's own
test-doesn't-need-real-NET_ADMIN posture."""

import subprocess

import pytest

from aipt.gateway import netem_control
from aipt.gateway.profiles import PRESETS, custom_profile


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_build_commands_clean_profile_is_del_only():
    cmds = netem_control.build_commands("eth0", PRESETS["clean"])
    assert len(cmds) == 1
    assert cmds[0] == ["tc", "qdisc", "del", "dev", "eth0", "root"]


def test_build_commands_delay_only():
    p = custom_profile(delay_ms=100)
    cmds = netem_control.build_commands("eth0", p)
    assert cmds[0] == ["tc", "qdisc", "del", "dev", "eth0", "root"]
    assert "netem" in cmds[1]
    assert "delay" in cmds[1] and "100ms" in cmds[1]
    # fq chained underneath, same as aipt.core.netem
    assert cmds[2][:2] == ["tc", "qdisc"]
    assert "fq" in cmds[2]


def test_build_commands_delay_with_jitter():
    p = custom_profile(delay_ms=100, jitter_ms=20)
    cmds = netem_control.build_commands("eth0", p)
    netem_cmd = cmds[1]
    idx = netem_cmd.index("delay")
    assert netem_cmd[idx + 1] == "100ms"
    assert netem_cmd[idx + 2] == "20ms"


def test_build_commands_loss():
    p = custom_profile(loss_pct=5.0)
    cmds = netem_control.build_commands("eth0", p)
    netem_cmd = cmds[1]
    idx = netem_cmd.index("loss")
    assert netem_cmd[idx + 1] == "5.0%"


def test_build_commands_reorder_without_delay_adds_minimal_delay():
    # netem's reorder only has meaning relative to a delay -- 0-delay
    # reorder is silently dropped, so we inject 1ms as a floor.
    p = custom_profile(reorder_pct=2.0)
    cmds = netem_control.build_commands("eth0", p)
    netem_cmd = cmds[1]
    assert "delay" in netem_cmd
    assert "reorder" in netem_cmd


def test_build_commands_reorder_with_explicit_delay_not_overridden():
    p = custom_profile(delay_ms=50, reorder_pct=2.0)
    cmds = netem_control.build_commands("eth0", p)
    netem_cmd = cmds[1]
    idx = netem_cmd.index("delay")
    assert netem_cmd[idx + 1] == "50ms"


def test_build_commands_full_profile_has_all_knobs():
    p = custom_profile(delay_ms=150, jitter_ms=40, loss_pct=1.0, reorder_pct=0.5)
    cmds = netem_control.build_commands("eth0", p)
    netem_cmd = cmds[1]
    assert "delay" in netem_cmd
    assert "loss" in netem_cmd
    assert "reorder" in netem_cmd


def test_available_reports_missing_tc(monkeypatch):
    monkeypatch.setattr(netem_control.shutil, "which", lambda name: None)
    ok, reason = netem_control.available()
    assert ok is False
    assert "iproute2" in reason or "tc" in reason


def test_available_true_when_tc_present(monkeypatch):
    monkeypatch.setattr(netem_control.shutil, "which", lambda name: "/sbin/tc")
    ok, reason = netem_control.available()
    assert ok is True


class TestApplyProfile:
    def setup_method(self):
        netem_control._STATE.clear()

    def test_apply_profile_without_tc_reports_ok_false(self, monkeypatch):
        monkeypatch.setattr(netem_control.shutil, "which", lambda name: None)
        result = netem_control.apply_profile("eth0", PRESETS["3g"])
        assert result["ok"] is False
        assert "iproute2" in result["reason"] or "tc" in result["reason"]
        assert result["profile"]["profile"] == "3g"
        assert len(result["commands"]) >= 1

    def test_apply_profile_dry_run_does_not_execute(self, monkeypatch):
        monkeypatch.setattr(netem_control.shutil, "which", lambda name: "/sbin/tc")
        calls = []
        monkeypatch.setattr(netem_control.subprocess, "run", lambda *a, **k: calls.append(a))
        result = netem_control.apply_profile("eth0", PRESETS["broadband"], dry_run=True)
        assert result["ok"] is True
        assert result["dry_run"] is True
        assert calls == []

    def test_apply_profile_success_runs_all_commands(self, monkeypatch):
        monkeypatch.setattr(netem_control.shutil, "which", lambda name: "/sbin/tc")
        calls = []

        def fake_run(argv, **kwargs):
            calls.append(argv)
            return _Proc(returncode=0)

        monkeypatch.setattr(netem_control.subprocess, "run", fake_run)
        result = netem_control.apply_profile("eth0", PRESETS["3g"])
        assert result["ok"] is True
        assert len(calls) == 3  # del, add netem, add fq
        assert netem_control.current_profile("eth0").name == "3g"

    def test_apply_profile_permission_denied_reports_ok_false(self, monkeypatch):
        # Simulates the sandbox's real condition: tc exists but the
        # process lacks CAP_NET_ADMIN, so the qdisc add fails.
        monkeypatch.setattr(netem_control.shutil, "which", lambda name: "/sbin/tc")

        def fake_run(argv, **kwargs):
            if argv[:3] == ["tc", "qdisc", "del"]:
                return _Proc(returncode=0)
            return _Proc(returncode=2, stderr="RTNETLINK answers: Operation not permitted")
        monkeypatch.setattr(netem_control.subprocess, "run", fake_run)

        result = netem_control.apply_profile("eth0", PRESETS["3g"])
        assert result["ok"] is False
        assert "Operation not permitted" in result["reason"]
        assert "NET_ADMIN" in result["reason"]

    def test_apply_profile_del_failure_on_empty_qdisc_is_swallowed(self, monkeypatch):
        # `tc qdisc del ... root` on a pristine interface returns nonzero
        # ("Cannot delete qdisc with handle of zero") -- that's not a real
        # failure and must not block applying a fresh profile.
        monkeypatch.setattr(netem_control.shutil, "which", lambda name: "/sbin/tc")

        def fake_run(argv, **kwargs):
            if argv[:3] == ["tc", "qdisc", "del"]:
                return _Proc(returncode=2, stderr="Cannot delete qdisc with handle of zero.")
            return _Proc(returncode=0)
        monkeypatch.setattr(netem_control.subprocess, "run", fake_run)

        result = netem_control.apply_profile("eth0", PRESETS["broadband"])
        assert result["ok"] is True

    def test_apply_profile_subprocess_exception_reports_ok_false(self, monkeypatch):
        monkeypatch.setattr(netem_control.shutil, "which", lambda name: "/sbin/tc")

        def fake_run(argv, **kwargs):
            raise OSError("boom")
        monkeypatch.setattr(netem_control.subprocess, "run", fake_run)

        result = netem_control.apply_profile("eth0", PRESETS["3g"])
        assert result["ok"] is False
        assert "boom" in result["reason"]

    def test_current_profile_defaults_to_clean(self):
        assert netem_control.current_profile("nonexistent0").name == "clean"

    def test_clear_applies_clean_profile(self, monkeypatch):
        monkeypatch.setattr(netem_control.shutil, "which", lambda name: "/sbin/tc")
        monkeypatch.setattr(netem_control.subprocess, "run", lambda *a, **k: _Proc(returncode=0))
        result = netem_control.clear("eth0")
        assert result["ok"] is True
        assert result["profile"]["profile"] == "clean"
        assert netem_control.current_profile("eth0").name == "clean"
