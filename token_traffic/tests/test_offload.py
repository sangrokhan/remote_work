"""Segmentation offload: parsing what ethtool says, and putting back what we changed.

Two failures matter more than the rest, and both are silent.

Leaving the machine changed. A capture that turns GRO off and does not put it back has
altered the box for every process on it, permanently, with nothing in the run document
admitting to it. Worse: restoring to `on` rather than to what it *was* would enable a
feature the operator had deliberately disabled. The tests below pin restoration to the
recorded prior value.

Reporting offload as off when it is on. Then a reader takes 64 KB kernel super-packets
for wire frames and counts slow-start bursts that are not there. `describe()` and the
recorded state exist for that reader, so they are tested harder than the plumbing.

ethtool is faked out throughout: this has to pass on a box with no ethtool, no NIC and
no CAP_NET_ADMIN, and a test that silently skips is worse than no test.
"""

import subprocess

import pytest

from core import offload


# Real `ethtool -k` output, trimmed to the lines that matter, from the box this was
# developed on. Real rather than invented because the parser's job is to survive the
# actual format -- the trailing "[fixed]" in particular.
ETHTOOL_K = """Features for enp1s0f0:
rx-checksumming: on
tx-checksumming: on
scatter-gather: on
tcp-segmentation-offload: on
\ttx-tcp-segmentation: on
\ttx-tcp-ecn-segmentation: on
\ttx-tcp-mangleid-segmentation: off
\ttx-tcp6-segmentation: on
generic-segmentation-offload: on
generic-receive-offload: on
large-receive-offload: off [fixed]
"""

ETHTOOL_K_ALL_OFF = """Features for eth0:
tcp-segmentation-offload: off
generic-segmentation-offload: off
generic-receive-offload: off
large-receive-offload: off [fixed]
"""


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def ethtool(monkeypatch):
    """Fake ethtool. Records every invocation; answers -k from a settable state."""
    calls = []
    state = {"out": ETHTOOL_K, "set_rc": 0, "set_err": ""}

    monkeypatch.setattr(offload, "ethtool_path", lambda: "/usr/sbin/ethtool")

    def run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:2] == ["ip", "route"]:
            return _Proc(0, "1.1.1.1 via 192.168.1.1 dev enp1s0f0 src 192.168.1.5 \n")
        if "-k" in cmd:
            return _Proc(0, state["out"])
        if "-K" in cmd:
            return _Proc(state["set_rc"], "", state["set_err"])
        return _Proc(0, "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.delenv("TRAFFIC_PCAP_NO_OFFLOAD", raising=False)
    return {"calls": calls, "state": state}


def _sets(calls):
    """Just the `ethtool -K` invocations, as {feature: value} maps in order."""
    out = []
    for c in calls:
        if "-K" in c:
            args = c[c.index("-K") + 2:]
            out.append(dict(zip(args[::2], args[1::2])))
    return out


# --- reading --------------------------------------------------------------

def test_read_parses_state_and_fixedness(ethtool):
    state = offload.read("enp1s0f0")
    assert state["tso"] == {"on": True, "fixed": False}
    assert state["gso"] == {"on": True, "fixed": False}
    assert state["gro"] == {"on": True, "fixed": False}
    # The sub-features (tx-tcp-segmentation and friends) are not top-level knobs and
    # must not be mistaken for them.
    assert set(state) == {"tso", "gso", "gro"}


def test_read_without_ethtool_is_empty_not_an_exception(monkeypatch):
    monkeypatch.setattr(offload, "ethtool_path", lambda: None)
    assert offload.read("enp1s0f0") == {}


def test_egress_iface_comes_from_the_routing_table(ethtool):
    """`any` is a tcpdump pseudo-interface that ethtool cannot address, so the device
    has to be discovered rather than configured."""
    assert offload.egress_iface("1.1.1.1") == "enp1s0f0"


# --- the window -----------------------------------------------------------

def test_disabled_by_default_it_only_observes(ethtool):
    """The default must not touch the NIC: turning offload off changes the timings
    this package measures, so it cannot happen because somebody asked for a pcap."""
    with offload.Window("1.1.1.1") as w:
        pass
    assert _sets(ethtool["calls"]) == []
    result = w.result()
    assert result["disabled"] == []
    assert result["requested"] is False
    assert result["during_capture"] == {"tso": True, "gso": True, "gro": True}


def test_when_asked_it_turns_the_three_off_in_one_call(ethtool, monkeypatch):
    monkeypatch.setenv("TRAFFIC_PCAP_NO_OFFLOAD", "1")
    with offload.Window("1.1.1.1") as w:
        during = _sets(ethtool["calls"])
        assert during == [{"tso": "off", "gso": "off", "gro": "off"}], (
            "one ethtool call, not three: each change reinitialises the ring, and the "
            "link being bounced is the one carrying the traffic under measurement")
        assert w.applied is True
    assert w.result()["during_capture"] == {"tso": False, "gso": False, "gro": False}


def test_it_restores_what_was_there_not_what_is_conventional(ethtool, monkeypatch):
    """The failure that would matter most. A box with GRO deliberately off must not
    come out of a run with GRO on."""
    ethtool["state"]["out"] = """Features for eth0:
tcp-segmentation-offload: on
generic-segmentation-offload: on
generic-receive-offload: off
"""
    monkeypatch.setenv("TRAFFIC_PCAP_NO_OFFLOAD", "1")
    with offload.Window("1.1.1.1"):
        pass

    sets = _sets(ethtool["calls"])
    assert sets[0] == {"tso": "off", "gso": "off"}, "gro was already off; leave it"
    assert sets[1] == {"tso": "on", "gso": "on"}, "restore to prior, and only what we changed"
    assert all("gro" not in s for s in sets)


def test_a_fixed_feature_is_left_alone(ethtool, monkeypatch):
    """`[fixed]` cannot be changed. Asking anyway makes ethtool fail the whole call,
    which would take the changeable features down with it."""
    ethtool["state"]["out"] = """Features for eth0:
tcp-segmentation-offload: on [fixed]
generic-segmentation-offload: on
generic-receive-offload: on
"""
    monkeypatch.setenv("TRAFFIC_PCAP_NO_OFFLOAD", "1")
    with offload.Window("1.1.1.1") as w:
        pass
    assert _sets(ethtool["calls"])[0] == {"gso": "off", "gro": "off"}
    assert w.result()["fixed"] == ["tso"]
    # tso stayed on, and the record says so rather than claiming a clean capture.
    assert w.result()["during_capture"]["tso"] is True


def test_already_off_means_nothing_to_do_and_nothing_to_restore(ethtool, monkeypatch):
    ethtool["state"]["out"] = ETHTOOL_K_ALL_OFF
    monkeypatch.setenv("TRAFFIC_PCAP_NO_OFFLOAD", "1")
    with offload.Window("1.1.1.1") as w:
        pass
    assert _sets(ethtool["calls"]) == []
    assert w.result()["disabled"] == []


def test_a_refused_change_is_recorded_and_the_capture_goes_on(ethtool, monkeypatch):
    """No CAP_NET_ADMIN, or a driver that says no. A run must not die configuring a
    NIC -- a harder-to-read pcap beats no run."""
    ethtool["state"]["set_rc"] = 1
    ethtool["state"]["set_err"] = "Cannot change generic-segmentation-offload"
    monkeypatch.setenv("TRAFFIC_PCAP_NO_OFFLOAD", "1")
    with offload.Window("1.1.1.1") as w:
        pass
    result = w.result()
    assert "Cannot change" in result["error"]
    assert result["disabled"] == []
    # And the state it reports is the truth: still on.
    assert result["during_capture"] == {"tso": True, "gso": True, "gro": True}


def test_a_failed_restore_is_shouted_about(ethtool, monkeypatch):
    """The machine is now not how we found it and nothing else in the run will notice."""
    monkeypatch.setenv("TRAFFIC_PCAP_NO_OFFLOAD", "1")
    w = offload.Window("1.1.1.1")
    w.__enter__()
    ethtool["state"]["set_rc"] = 1
    ethtool["state"]["set_err"] = "busy"
    w.__exit__(None, None, None)
    assert "RESTORE FAILED" in w.result()["error"]


def test_restore_is_idempotent(ethtool, monkeypatch):
    monkeypatch.setenv("TRAFFIC_PCAP_NO_OFFLOAD", "1")
    w = offload.Window("1.1.1.1")
    w.__enter__()
    w.restore()
    w.restore()
    assert len(_sets(ethtool["calls"])) == 2, "one disable, one restore, no more"


def test_missing_ethtool_when_asked_says_which_is_missing(monkeypatch):
    monkeypatch.setattr(offload, "ethtool_path", lambda: None)
    monkeypatch.setenv("TRAFFIC_PCAP_NO_OFFLOAD", "1")
    with offload.Window("1.1.1.1", iface="eth0") as w:
        pass
    assert w.result()["error"] == "ethtool not installed"


# --- what the reader is told ----------------------------------------------

def test_describe_warns_while_offload_is_on():
    text = offload.describe({"iface": "eth0",
                             "during_capture": {"tso": True, "gso": True, "gro": False}})
    assert "offload ON" in text
    assert "tso" in text and "gso" in text
    assert "not wire frames" in text
    assert "segment counts" in text


def test_describe_confirms_real_frames_when_all_off():
    text = offload.describe({"iface": "eth0",
                             "during_capture": {"tso": False, "gso": False, "gro": False}})
    assert "offload off" in text
    assert "real wire frames" in text


def test_current_is_shaped_the_way_describe_reads_it(ethtool):
    """The bug this test exists for, found in the running container: the preflight
    passed `read()` straight to `describe()`. Different shape, no exception -- it just
    reported "unknown" on a box where offload was plainly on, which is the one answer
    that must never be wrong by accident."""
    state = offload.current("1.1.1.1")
    assert state["during_capture"] == {"tso": True, "gso": True, "gro": True}
    assert state["iface"] == "enp1s0f0"
    text = offload.describe(state)
    assert "unknown" not in text
    assert "offload ON" in text


def test_current_survives_a_box_with_no_ethtool(monkeypatch):
    monkeypatch.setattr(offload, "ethtool_path", lambda: None)
    assert "unknown" in offload.describe(offload.current("1.1.1.1"))


def test_describe_admits_when_it_does_not_know():
    """Silence would read as "off". Unknown has to say unknown, or a pcap gets trusted
    for something nobody checked."""
    assert "unknown" in offload.describe({})
    assert "unknown" in offload.describe({"iface": "eth0", "during_capture": {}})
