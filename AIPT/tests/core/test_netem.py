"""netem.py: env-driven tc netem setup run at container start.

Tests cover:
  - environment parsing (valid, missing, zero)
  - command generation for client-side and server-side delay
  - dry-run mode produces commands without executing them

Migrated unchanged from tcp_congestion/tests/test_netem.py -- token_traffic
has no equivalent module.
"""

import pytest
from aipt.core import netem


def test_parse_zero_delay_is_disabled():
    assert netem.parse_delay("0") == 0


def test_parse_positive_delay_returns_ms():
    assert netem.parse_delay("30") == 30


def test_parse_empty_string_returns_zero():
    assert netem.parse_delay("") == 0


def test_parse_negative_clamps_to_zero():
    assert netem.parse_delay("-5") == 0


def test_parse_non_numeric_raises():
    with pytest.raises(ValueError):
        netem.parse_delay("banana")


def test_build_commands_zero_delay_returns_empty():
    cmds = netem.build_commands(iface="eth0", delay_ms=0)
    assert cmds == []


def test_build_commands_positive_delay_adds_netem():
    cmds = netem.build_commands(iface="eth0", delay_ms=20)
    assert any("netem" in c for c in cmds)
    assert any("20ms" in c for c in cmds)


def test_build_commands_chains_fq_as_child_of_netem():
    """A bare `netem` at root replaces the interface's qdisc outright, which
    silently discards fq (and the BBR pacing it provides) -- see netem.py's
    docstring. fq must be added as netem's child, not left implicit."""
    cmds = netem.build_commands(iface="eth0", delay_ms=20)
    joined = " ".join(cmds)
    assert "parent" in joined
    assert " fq" in joined or joined.endswith("fq")


def test_build_commands_includes_qdisc_del_before_add():
    cmds = netem.build_commands(iface="eth0", delay_ms=30)
    # del must come before add so re-runs are idempotent
    joined = " ".join(cmds)
    assert joined.index("del") < joined.index("add")


def test_build_commands_uses_given_interface():
    cmds = netem.build_commands(iface="eth1", delay_ms=10)
    assert all("eth1" in c for c in cmds)


def test_apply_dry_run_returns_commands_without_executing(monkeypatch):
    executed = []
    monkeypatch.setattr(netem, "_run", lambda cmd: executed.append(cmd))
    cmds = netem.apply(iface="eth0", delay_ms=25, dry_run=True)
    assert executed == []
    assert len(cmds) > 0


def test_apply_wet_run_executes_commands(monkeypatch):
    executed = []
    monkeypatch.setattr(netem, "_run", lambda cmd: executed.append(cmd))
    netem.apply(iface="eth0", delay_ms=25, dry_run=False)
    assert len(executed) > 0


def test_from_env_reads_delay_env(monkeypatch):
    monkeypatch.setenv("NETEM_DELAY_MS", "40")
    monkeypatch.setenv("NETEM_IFACE", "eth0")
    cfg = netem.from_env()
    assert cfg["delay_ms"] == 40
    assert cfg["iface"] == "eth0"


def test_from_env_defaults_to_zero_and_eth0(monkeypatch):
    monkeypatch.delenv("NETEM_DELAY_MS", raising=False)
    monkeypatch.delenv("NETEM_IFACE", raising=False)
    cfg = netem.from_env()
    assert cfg["delay_ms"] == 0
    assert cfg["iface"] == "eth0"
