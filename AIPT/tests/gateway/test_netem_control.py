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
        result = netem_control.apply_profile("eth0", PRESETS["wireless"])
        assert result["ok"] is False
        assert "iproute2" in result["reason"] or "tc" in result["reason"]
        assert result["profile"]["profile"] == "wireless"
        assert len(result["commands"]) >= 1

    def test_apply_profile_dry_run_does_not_execute(self, monkeypatch):
        monkeypatch.setattr(netem_control.shutil, "which", lambda name: "/sbin/tc")
        calls = []
        monkeypatch.setattr(netem_control.subprocess, "run", lambda *a, **k: calls.append(a))
        result = netem_control.apply_profile("eth0", PRESETS["wired"], dry_run=True)
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
        result = netem_control.apply_profile("eth0", PRESETS["wireless"])
        assert result["ok"] is True
        assert len(calls) == 3  # del, add netem, add fq
        assert netem_control.current_profile("eth0").name == "wireless"

    def test_apply_profile_permission_denied_reports_ok_false(self, monkeypatch):
        # Simulates the sandbox's real condition: tc exists but the
        # process lacks CAP_NET_ADMIN, so the qdisc add fails.
        monkeypatch.setattr(netem_control.shutil, "which", lambda name: "/sbin/tc")

        def fake_run(argv, **kwargs):
            if argv[:3] == ["tc", "qdisc", "del"]:
                return _Proc(returncode=0)
            return _Proc(returncode=2, stderr="RTNETLINK answers: Operation not permitted")
        monkeypatch.setattr(netem_control.subprocess, "run", fake_run)

        result = netem_control.apply_profile("eth0", PRESETS["wireless"])
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

        result = netem_control.apply_profile("eth0", PRESETS["wired"])
        assert result["ok"] is True

    def test_apply_profile_subprocess_exception_reports_ok_false(self, monkeypatch):
        monkeypatch.setattr(netem_control.shutil, "which", lambda name: "/sbin/tc")

        def fake_run(argv, **kwargs):
            raise OSError("boom")
        monkeypatch.setattr(netem_control.subprocess, "run", fake_run)

        result = netem_control.apply_profile("eth0", PRESETS["wireless"])
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


class TestApplyGatewayProfile:
    """2026-09 client-link-only redesign: only the client-facing leg
    carries the user-selected profile (in both directions, egress direct +
    ingress via IFB); the backend-facing leg always gets the fixed
    ETHERNET_BASELINE regardless of what was requested."""

    def setup_method(self):
        netem_control._STATE.clear()

    def test_apply_gateway_profile_shapes_client_link_both_directions(self, monkeypatch):
        monkeypatch.setattr(netem_control.shutil, "which", lambda name: "/sbin/tc")
        calls = []

        def fake_run(argv, **kwargs):
            calls.append(argv)
            return _Proc(returncode=0)

        monkeypatch.setattr(netem_control.subprocess, "run", fake_run)
        result = netem_control.apply_gateway_profile("eth0", "eth1", "ifb0", PRESETS["wireless"])

        assert result["ok"] is True
        assert result["client_iface"] == "eth0"
        assert result["backend_iface"] == "eth1"
        assert result["ifb_dev"] == "ifb0"
        assert result["client"]["ok"] is True
        assert result["client"]["egress"]["ok"] is True
        assert result["client"]["ingress"]["ok"] is True
        # client egress (del/netem/fq) + IFB setup (modprobe/link add/link
        # set) + ingress redirect (del/qdisc add/filter add) + ifb0 netem
        # (del/netem/fq) + backend baseline egress (del/netem/fq, since
        # ETHERNET_BASELINE has a nonzero delay) = 3+3+3+3+3 = 15
        assert len(calls) == 15
        assert netem_control.current_profile("eth0").name == "wireless"
        assert netem_control.current_ingress_profile("eth0").name == "wireless"

    def test_apply_gateway_profile_always_applies_ethernet_baseline_to_backend(self, monkeypatch):
        # Even when the caller asks for "wireless" on the client link, the
        # backend leg must get ETHERNET_BASELINE -- not "wireless".
        monkeypatch.setattr(netem_control.shutil, "which", lambda name: "/sbin/tc")
        monkeypatch.setattr(netem_control.subprocess, "run", lambda *a, **k: _Proc(returncode=0))

        result = netem_control.apply_gateway_profile("eth0", "eth1", "ifb0", PRESETS["wireless"])

        assert result["backend"]["profile"]["profile"] == "ethernet_baseline"
        assert netem_control.current_profile("eth1").name == "ethernet_baseline"
        # The top-level "profile" field reports what was requested for the
        # client link, not what the backend actually got.
        assert result["profile"]["profile"] == "wireless"

    def test_apply_gateway_profile_dry_run_does_not_execute(self, monkeypatch):
        monkeypatch.setattr(netem_control.shutil, "which", lambda name: "/sbin/tc")
        calls = []
        monkeypatch.setattr(netem_control.subprocess, "run", lambda *a, **k: calls.append(a))
        result = netem_control.apply_gateway_profile("eth0", "eth1", "ifb0", PRESETS["wired"], dry_run=True)
        assert result["ok"] is True
        assert result["client"]["egress"]["dry_run"] is True
        assert result["client"]["ingress"]["dry_run"] is True
        assert result["backend"]["dry_run"] is True
        assert calls == []

    def test_apply_gateway_profile_without_tc_reports_ok_false(self, monkeypatch):
        monkeypatch.setattr(netem_control.shutil, "which", lambda name: None)
        result = netem_control.apply_gateway_profile("eth0", "eth1", "ifb0", PRESETS["wireless"])
        assert result["ok"] is False
        assert "eth0" in result["reason"]

    def test_apply_gateway_profile_ok_gated_by_client_link_only(self, monkeypatch):
        # Backend baseline failing (e.g. no CAP_NET_ADMIN on that leg)
        # must not be silently hidden, but it also must not flip top-level
        # "ok" -- only the client link (what the caller actually asked to
        # change) gates "ok" (module docstring's explicit rationale).
        monkeypatch.setattr(netem_control.shutil, "which", lambda name: "/sbin/tc")

        def fake_run(argv, **kwargs):
            if argv[:3] == ["tc", "qdisc", "del"]:
                return _Proc(returncode=0)
            if "eth1" in argv:
                return _Proc(returncode=2, stderr="RTNETLINK answers: Operation not permitted")
            return _Proc(returncode=0)

        monkeypatch.setattr(netem_control.subprocess, "run", fake_run)
        result = netem_control.apply_gateway_profile("eth0", "eth1", "ifb0", PRESETS["wireless"])

        assert result["client"]["ok"] is True
        assert result["backend"]["ok"] is False
        assert result["ok"] is True  # client link succeeded, so overall ok
        assert "reason" not in result  # only set when client link fails

    def test_current_gateway_profile_defaults_to_clean_and_baseline_reported_separately(self):
        result = netem_control.current_gateway_profile("eth0", "eth1", "ifb0")
        assert result["client"]["egress"]["profile"] == "clean"
        assert result["client"]["ingress"]["profile"] == "clean"
        # current_profile() has no special-case for the backend leg -- it
        # just reports whatever was last applied there (defaults to
        # "clean" until a real apply_gateway_profile call installs the
        # baseline; the *policy* that it should always be
        # ETHERNET_BASELINE lives in apply_backend_link_baseline, not in
        # this read path).
        assert result["backend"]["profile"] == "clean"

    def test_clear_gateway_reapplies_clean_client_and_baseline_backend(self, monkeypatch):
        monkeypatch.setattr(netem_control.shutil, "which", lambda name: "/sbin/tc")
        monkeypatch.setattr(netem_control.subprocess, "run", lambda *a, **k: _Proc(returncode=0))
        result = netem_control.clear_gateway("eth0", "eth1", "ifb0")
        assert result["ok"] is True
        assert netem_control.current_profile("eth0").name == "clean"
        assert netem_control.current_ingress_profile("eth0").name == "clean"
        assert netem_control.current_profile("eth1").name == "ethernet_baseline"

    def test_default_client_backend_ifb_constants_exist(self):
        assert netem_control.DEFAULT_CLIENT_IFACE
        assert netem_control.DEFAULT_BACKEND_IFACE
        assert netem_control.DEFAULT_IFB_DEV
