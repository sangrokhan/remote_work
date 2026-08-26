"""offload.py: env-driven ethtool -K TSO/GSO/SG/GRO/LRO toggle run at
container start.

Tests cover:
  - environment parsing (truthy strings, missing, off)
  - command generation (disabled vs enabled)
  - dry-run mode produces commands without executing them
"""

from tcp_congestion import offload


def test_flag_recognizes_truthy_strings():
    assert offload._flag("1") is True
    assert offload._flag("true") is True
    assert offload._flag("True") is True
    assert offload._flag("yes") is True
    assert offload._flag("on") is True


def test_flag_recognizes_falsy_or_missing():
    assert offload._flag("0") is False
    assert offload._flag("") is False
    assert offload._flag(None) is False  # type: ignore[arg-type]
    assert offload._flag("false") is False


def test_build_commands_disabled_false_returns_empty():
    cmds = offload.build_commands(iface="eth0", disable=False)
    assert cmds == []


def test_build_commands_disable_true_turns_off_all_features():
    cmds = offload.build_commands(iface="eth0", disable=True)
    assert len(cmds) == 1
    cmd = cmds[0]
    assert "ethtool -K eth0" in cmd
    for feat in offload.FEATURES:
        assert f"{feat} off" in cmd


def test_build_commands_covers_tso_gso_sg_gro_lro():
    assert set(offload.FEATURES) == {"tso", "gso", "sg", "gro", "lro"}


def test_build_commands_uses_given_interface():
    cmds = offload.build_commands(iface="eth1", disable=True)
    assert "eth1" in cmds[0]


def test_apply_dry_run_returns_commands_without_executing(monkeypatch):
    executed = []
    monkeypatch.setattr(offload, "_run", lambda cmd: executed.append(cmd))
    cmds = offload.apply(iface="eth0", disable=True, dry_run=True)
    assert executed == []
    assert len(cmds) > 0


def test_apply_wet_run_executes_commands(monkeypatch):
    executed = []
    monkeypatch.setattr(offload, "_run", lambda cmd: executed.append(cmd))
    offload.apply(iface="eth0", disable=True, dry_run=False)
    assert len(executed) > 0


def test_apply_disabled_false_executes_nothing(monkeypatch):
    executed = []
    monkeypatch.setattr(offload, "_run", lambda cmd: executed.append(cmd))
    offload.apply(iface="eth0", disable=False, dry_run=False)
    assert executed == []


def test_from_env_reads_disable_env(monkeypatch):
    monkeypatch.setenv("NIC_OFFLOAD_DISABLE", "1")
    monkeypatch.setenv("NIC_OFFLOAD_IFACE", "eth0")
    cfg = offload.from_env()
    assert cfg["disable"] is True
    assert cfg["iface"] == "eth0"


def test_from_env_defaults_to_disabled_false_and_eth0(monkeypatch):
    monkeypatch.delenv("NIC_OFFLOAD_DISABLE", raising=False)
    monkeypatch.delenv("NIC_OFFLOAD_IFACE", raising=False)
    cfg = offload.from_env()
    assert cfg["disable"] is False
    assert cfg["iface"] == "eth0"
