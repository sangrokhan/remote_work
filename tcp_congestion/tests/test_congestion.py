"""congestion: readiness check for the 4-way algorithm comparison lab.

Pure-function tests inject fake paths/subprocess results so these run
without root, without a real kernel state, and identically on a dev laptop
or in CI -- the live "does the kernel actually have bbr/vegas loaded"
answer is exercised manually via the web UI's /api/config banner.
"""

from __future__ import annotations

import subprocess

from tcp_congestion import congestion


def test_available_algorithms_parses_whitespace_separated_list(tmp_path):
    p = tmp_path / "avail"
    p.write_text("reno cubic bbr vegas\n")
    assert congestion.available_algorithms(p) == ["reno", "cubic", "bbr", "vegas"]


def test_available_algorithms_missing_file_returns_empty(tmp_path):
    p = tmp_path / "does_not_exist"
    assert congestion.available_algorithms(p) == []


def test_current_default_algorithm_reads_file(tmp_path):
    p = tmp_path / "current"
    p.write_text("cubic\n")
    assert congestion.current_default_algorithm(p) == "cubic"


def test_current_default_algorithm_missing_file_returns_empty(tmp_path):
    p = tmp_path / "does_not_exist"
    assert congestion.current_default_algorithm(p) == ""


def test_qdisc_kind_missing_tc_returns_empty_kind(monkeypatch):
    monkeypatch.setattr(congestion.shutil, "which", lambda name: None)
    kind, raw = congestion.qdisc_kind("eth0")
    assert kind == ""
    assert "tc" in raw.lower()


def test_qdisc_kind_parses_fq(monkeypatch):
    monkeypatch.setattr(congestion.shutil, "which", lambda name: "/sbin/tc")

    class _Proc:
        returncode = 0
        stdout = "qdisc fq 0: root refcnt 2 limit 10000p flow_limit 100p\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _Proc())
    kind, raw = congestion.qdisc_kind("eth0")
    assert kind == "fq"
    assert "fq" in raw


def test_qdisc_kind_parses_fq_codel(monkeypatch):
    monkeypatch.setattr(congestion.shutil, "which", lambda name: "/sbin/tc")

    class _Proc:
        returncode = 0
        stdout = "qdisc fq_codel 0: root refcnt 2 limit 10240p\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _Proc())
    kind, _ = congestion.qdisc_kind("eth0")
    assert kind == "fq_codel"


def test_qdisc_kind_detects_fq_chained_under_netem(monkeypatch):
    """netem.build_commands chains fq as netem's child (parent 1:) so RTT
    injection doesn't silently discard BBR's pacing qdisc. `tc qdisc show`
    then lists BOTH lines; the effective/child qdisc (fq) is what matters,
    not netem sitting at root."""
    monkeypatch.setattr(congestion.shutil, "which", lambda name: "/sbin/tc")

    class _Proc:
        returncode = 0
        stdout = (
            "qdisc netem 1: root refcnt 2 limit 1000 delay 20ms\n"
            "qdisc fq 10: parent 1: limit 10000p flow_limit 100p\n"
        )
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _Proc())
    kind, raw = congestion.qdisc_kind("eth0")
    assert kind == "fq"
    assert "netem" in raw  # raw output still shows the full chain for debugging


def test_qdisc_kind_netem_without_chained_fq_reports_netem(monkeypatch):
    """A netem-only root (the old, broken pre-chaining behaviour) must be
    surfaced as 'netem', not silently reported as fq -- this is exactly the
    misconfiguration the readiness banner needs to catch and flag."""
    monkeypatch.setattr(congestion.shutil, "which", lambda name: "/sbin/tc")

    class _Proc:
        returncode = 0
        stdout = "qdisc netem 1: root refcnt 2 limit 1000 delay 20ms\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _Proc())
    kind, _ = congestion.qdisc_kind("eth0")
    assert kind == "netem"


def test_qdisc_kind_nonzero_exit_returns_empty_kind(monkeypatch):
    monkeypatch.setattr(congestion.shutil, "which", lambda name: "/sbin/tc")

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "Cannot find device \"eth0\"\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _Proc())
    kind, raw = congestion.qdisc_kind("eth0")
    assert kind == ""
    assert "eth0" in raw


def test_status_ready_when_all_algorithms_and_fq_present(monkeypatch):
    monkeypatch.setattr(congestion, "available_algorithms",
                        lambda path=None: ["reno", "cubic", "bbr", "vegas"])
    monkeypatch.setattr(congestion, "current_default_algorithm", lambda path=None: "cubic")
    monkeypatch.setattr(congestion, "qdisc_kind", lambda iface="eth0": ("fq", "qdisc fq 0: root"))

    result = congestion.status(iface="eth0")
    assert result["ready"] is True
    assert result["missing"] == []
    assert result["qdisc_ok"] is True
    assert result["guidance"] == []


def test_status_reports_missing_algorithms_with_modprobe_guidance(monkeypatch):
    monkeypatch.setattr(congestion, "available_algorithms",
                        lambda path=None: ["reno", "cubic"])
    monkeypatch.setattr(congestion, "current_default_algorithm", lambda path=None: "cubic")
    monkeypatch.setattr(congestion, "qdisc_kind", lambda iface="eth0": ("fq", "qdisc fq 0: root"))

    result = congestion.status(iface="eth0")
    assert result["ready"] is False
    assert set(result["missing"]) == {"bbr", "vegas"}
    guidance_text = " ".join(result["guidance"])
    assert "modprobe" in guidance_text
    assert "tcp_bbr" in guidance_text
    assert "tcp_vegas" in guidance_text


def test_status_reports_wrong_qdisc_with_tc_guidance(monkeypatch):
    monkeypatch.setattr(congestion, "available_algorithms",
                        lambda path=None: ["reno", "cubic", "bbr", "vegas"])
    monkeypatch.setattr(congestion, "current_default_algorithm", lambda path=None: "cubic")
    monkeypatch.setattr(congestion, "qdisc_kind",
                        lambda iface="eth0": ("fq_codel", "qdisc fq_codel 0: root"))

    result = congestion.status(iface="eth0")
    assert result["ready"] is False
    assert result["qdisc_ok"] is False
    guidance_text = " ".join(result["guidance"])
    assert "fq_codel" in guidance_text
    assert "tc qdisc replace dev eth0 root fq" in guidance_text


def test_status_unknown_qdisc_is_not_ok_but_does_not_crash(monkeypatch):
    monkeypatch.setattr(congestion, "available_algorithms",
                        lambda path=None: ["reno", "cubic", "bbr", "vegas"])
    monkeypatch.setattr(congestion, "current_default_algorithm", lambda path=None: "cubic")
    monkeypatch.setattr(congestion, "qdisc_kind",
                        lambda iface="eth0": ("", "tc not installed"))

    result = congestion.status(iface="eth0")
    assert result["ready"] is False
    assert result["qdisc"] == ""
    assert "unknown" in " ".join(result["guidance"])


def test_offload_status_missing_ethtool(monkeypatch):
    monkeypatch.setattr(congestion.shutil, "which", lambda name: None)
    result = congestion.offload_status("eth0")
    assert result["available"] is False
    assert "ethtool" in result["reason"].lower()
    assert result["all_off"] is False


def test_offload_status_parses_all_on(monkeypatch):
    monkeypatch.setattr(congestion.shutil, "which", lambda name: "/sbin/ethtool")

    class _Proc:
        returncode = 0
        stdout = (
            "Features for eth0:\n"
            "tcp-segmentation-offload: on\n"
            "generic-segmentation-offload: on\n"
            "generic-receive-offload: off\n"
            "large-receive-offload: off [fixed]\n"
            "scatter-gather: on\n"
        )
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _Proc())
    result = congestion.offload_status("eth0")
    assert result["available"] is True
    assert result["features"]["tso"] == "on"
    assert result["features"]["gro"] == "off"
    assert result["all_off"] is False


def test_offload_status_parses_all_off(monkeypatch):
    monkeypatch.setattr(congestion.shutil, "which", lambda name: "/sbin/ethtool")

    class _Proc:
        returncode = 0
        stdout = (
            "Features for eth0:\n"
            "tcp-segmentation-offload: off\n"
            "generic-segmentation-offload: off\n"
            "generic-receive-offload: off\n"
            "large-receive-offload: off [fixed]\n"
            "scatter-gather: off\n"
        )
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _Proc())
    result = congestion.offload_status("eth0")
    assert result["all_off"] is True


def test_status_includes_offload_key(monkeypatch):
    monkeypatch.setattr(congestion, "available_algorithms",
                        lambda path=None: ["reno", "cubic", "bbr", "vegas"])
    monkeypatch.setattr(congestion, "current_default_algorithm", lambda path=None: "cubic")
    monkeypatch.setattr(congestion, "qdisc_kind", lambda iface="eth0": ("fq", "qdisc fq 0: root"))
    monkeypatch.setattr(congestion, "offload_status",
                        lambda iface="eth0": {"available": True, "reason": "",
                                              "features": {}, "all_off": True})

    result = congestion.status(iface="eth0")
    assert result["offload"]["all_off"] is True
