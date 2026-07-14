"""Capture helpers, with tcpdump faked out.

None of this needs tcpdump, root, or a network: `available()` has to be able to say
"no, and here is the knob" on a machine that has none of them, and the filename and
filter rules are pure string work. The one thing that would need a real tcpdump --
capturing packets -- is not tested here, because a test that silently skips is worse
than no test at all.

Two ways to get `available()` wrong, and an earlier version got the second:

1. tcpdump being installed says nothing; capturing needs CAP_NET_RAW.
2. Asking whether *we* can open a raw socket also says nothing. setcap grants the
   capability to the tcpdump binary, not to this process, so probing our own socket
   reports "unavailable" forever no matter what the operator does.

So the probe starts the capture it would start and sees whether it survives.
"""

import getpass
import subprocess
from pathlib import Path

import pytest

from core import capture


class _Stderr:
    def __init__(self, text=b"tcpdump: any: You don't have permission to perform this capture"):
        self._text = text

    def read(self):
        return self._text


class _Proc:
    """A fake tcpdump. alive=False means it died on startup (no permission)."""

    def __init__(self, alive=True, stderr=None):
        self._alive = alive
        self.stderr = stderr or _Stderr()
        self.killed = False

    def poll(self):
        return None if self._alive else 1

    def send_signal(self, sig):
        pass

    def terminate(self):
        self.killed = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("TRAFFIC_PCAP_DISABLE", raising=False)
    monkeypatch.delenv("TRAFFIC_CAPTURE_PROBE_TTL", raising=False)
    capture.reset_capability_cache()
    yield
    capture.reset_capability_cache()


def _fake_tcpdump(monkeypatch, alive=True):
    proc = _Proc(alive=alive)
    monkeypatch.setattr(capture, "tcpdump_path", lambda: "/usr/bin/tcpdump")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(capture.time, "sleep", lambda s: None)
    capture.reset_capability_cache()
    return proc


# --- the filter ----------------------------------------------------------------

def test_filter_narrows_to_the_api_host():
    assert capture._filter_expr(["1.2.3.4", "5.6.7.8"]) == (
        "tcp port 443 and (host 1.2.3.4 or host 5.6.7.8)")


def test_filter_falls_back_to_all_of_443_when_the_name_did_not_resolve():
    # A noisier pcap beats no pcap.
    assert capture._filter_expr([]) == "tcp port 443"


# --- naming: one pcap per (provider, arm) --------------------------------------

def test_the_pcap_is_named_for_the_provider_and_the_arm():
    c = capture.Capture("2026-07-14T00:00:00", "gemini", "interaction_inline",
                        "generativelanguage.googleapis.com")
    assert c.error == ""
    assert c.path.name.startswith("capture_gemini_interaction_inline_")
    assert capture._SAFE_NAME.match(c.path.name)


def test_the_name_capture_writes_is_a_name_the_download_route_accepts():
    """A run stamps its arms with `datetime.now(timezone.utc).isoformat()`, which carries
    a dot and a plus (2026-07-14T09:46:57.837905+00:00). Only the colons used to be
    replaced, so both survived into the filename and _SAFE_NAME then refused to match the
    file capture had just written: tcpdump wrote a good pcap, the run recorded it, and
    every download of it 404'd. The two must agree, so they are tested together.
    """
    c = capture.Capture("2026-07-14T09:46:57.837905+00:00", "gemini", "stateless",
                        "generativelanguage.googleapis.com")
    assert c.error == ""
    assert capture._SAFE_NAME.match(c.path.name), c.path.name


def test_two_arms_of_two_providers_do_not_collide():
    names = {
        capture.Capture("2026-07-14T00:00:00", p, a, "h").path.name
        for p, a in [("gemini", "stateless"), ("openai", "chat_stateless"),
                     ("gemini", "cached"), ("openai", "responses_stateful")]
    }
    assert len(names) == 4


def test_an_unspellable_label_is_refused_not_renamed():
    # The predecessor substituted a default label for anything it could not spell,
    # which is how an arm shipped a pcap claiming to be a different arm.
    c = capture.Capture("2026-07-14T00:00:00", "gemini", "../../etc/passwd", "h")
    assert c.error.startswith("unsafe_capture_label")


def test_a_download_name_cannot_traverse_out_of_the_pcap_dir():
    for bad in ["../etc/passwd", "capture_x.pcap/../y", "run.json",
                "capture_; rm -rf.pcap", ""]:
        assert capture.safe_pcap_path(bad) is None


# --- drop counters --------------------------------------------------------------

def test_the_drop_counters_are_parsed_off_tcpdumps_exit_summary():
    text = ("130 packets captured\n"
            "137 packets received by filter\n"
            "7 packets dropped by kernel\n"
            "0 packets dropped by interface\n")
    assert capture._parse_tcpdump_stats(text) == {
        "captured": 130, "received_by_filter": 137,
        "dropped_by_kernel": 7, "dropped_by_interface": 0,
    }


def test_partial_and_empty_summaries_parse():
    # Some platforms never print "dropped by interface".
    assert capture._parse_tcpdump_stats("42 packets captured\n") == {"captured": 42}
    assert capture._parse_tcpdump_stats("") == {}


def test_a_lossy_capture_says_so_rather_than_passing_for_a_complete_one(monkeypatch,
                                                                       tmp_path):
    monkeypatch.setenv("TRAFFIC_PCAP_DIR", str(tmp_path))
    c = capture.Capture("2026-07-14T00:00:00", "gemini", "cached", "h")
    c.path.write_bytes(b"pcapdata")
    c.stats = {"captured": 130, "received_by_filter": 137, "dropped_by_kernel": 7}
    res = c.result()
    assert res["dropped"] == 7
    assert res["ok"] is True
    assert any("dropped" in line for line in res["log"])


def test_a_capture_that_got_nothing_is_not_ok(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAFFIC_PCAP_DIR", str(tmp_path))
    c = capture.Capture("2026-07-14T00:00:00", "gemini", "cached", "h")
    res = c.result()
    assert res["ok"] is False
    assert "no packets" in res["note"]


# --- availability ---------------------------------------------------------------

def test_a_tcpdump_that_survives_startup_can_capture(monkeypatch):
    _fake_tcpdump(monkeypatch, alive=True)
    assert capture.can_raw_capture() is True


def test_a_tcpdump_that_dies_on_startup_cannot(monkeypatch):
    _fake_tcpdump(monkeypatch, alive=False)
    assert capture.can_raw_capture() is False


def test_the_probe_does_not_leave_tcpdump_running(monkeypatch):
    proc = _fake_tcpdump(monkeypatch, alive=True)
    capture.can_raw_capture()
    assert proc.killed is True


def test_the_probe_is_cached(monkeypatch):
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return _Proc(alive=True)

    monkeypatch.setattr(capture, "tcpdump_path", lambda: "/usr/bin/tcpdump")
    monkeypatch.setattr(subprocess, "Popen", counting)
    monkeypatch.setattr(capture.time, "sleep", lambda s: None)
    capture.can_raw_capture()
    capture.can_raw_capture()
    assert calls["n"] == 1


def test_granting_caps_is_picked_up_once_the_cache_expires(monkeypatch):
    # The operator runs setcap while the server is up; the page must stop saying
    # "unavailable" without a restart.
    monkeypatch.setenv("TRAFFIC_CAPTURE_PROBE_TTL", "0")
    _fake_tcpdump(monkeypatch, alive=False)
    assert capture.can_raw_capture() is False
    _fake_tcpdump(monkeypatch, alive=True)
    assert capture.can_raw_capture() is True


def test_a_missing_binary_is_unavailable(monkeypatch):
    monkeypatch.setattr(capture, "tcpdump_path", lambda: None)
    ok, reason = capture.available()
    assert ok is False and "not installed" in reason


def test_a_capless_tcpdump_names_the_fix(monkeypatch):
    _fake_tcpdump(monkeypatch, alive=False)
    ok, reason = capture.available()
    assert ok is False
    assert "NET_RAW" in reason and "setcap" in reason


def test_a_capable_tcpdump_is_available(monkeypatch):
    _fake_tcpdump(monkeypatch, alive=True)
    assert capture.available() == (True, "ready")


def test_the_disable_flag_wins(monkeypatch):
    _fake_tcpdump(monkeypatch, alive=True)
    monkeypatch.setenv("TRAFFIC_PCAP_DISABLE", "1")
    ok, reason = capture.available()
    assert ok is False and "disabled" in reason


def test_tcpdump_keeps_our_uid_so_it_can_write_the_pcap(monkeypatch, tmp_path):
    # A setcap'd tcpdump drops privileges to the unprivileged `tcpdump` user, which
    # cannot write into our pcap directory -- so the capture dies with "Permission
    # denied" on the output file even though it had NET_RAW all along.
    argv = {}
    monkeypatch.setenv("TRAFFIC_PCAP_DIR", str(tmp_path))
    monkeypatch.setattr(capture, "tcpdump_path", lambda: "/usr/bin/tcpdump")
    monkeypatch.setattr(capture.time, "sleep", lambda s: None)
    monkeypatch.setattr(capture, "_resolve_ips", lambda host: ["1.2.3.4"])

    def popen(cmd, **k):
        argv["cmd"] = cmd
        return _Proc(alive=True)

    monkeypatch.setattr(subprocess, "Popen", popen)
    with capture.Capture("2026-07-14T00:00:00", "gemini", "stateless", "h"):
        pass
    cmd = argv["cmd"]
    assert cmd[cmd.index("-Z") + 1] == getpass.getuser()
    assert cmd[cmd.index("-s") + 1] == str(capture.PCAP_SNAPLEN)
    assert cmd[-1] == "tcp port 443 and (host 1.2.3.4)"


def test_a_second_capture_never_overwrites_the_first(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAFFIC_PCAP_DIR", str(tmp_path))
    monkeypatch.setattr(capture, "tcpdump_path", lambda: "/usr/bin/tcpdump")
    monkeypatch.setattr(capture.time, "sleep", lambda s: None)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _Proc(alive=True))
    c = capture.Capture("2026-07-14T00:00:00", "gemini", "stateless", "h")
    c.path.write_bytes(b"someone else's capture")
    with c:
        pass
    assert c.error == "pcap_name_collision"
    assert c.path.read_bytes() == b"someone else's capture"


# --- AppArmor: where tcpdump is allowed to write --------------------------------

def test_a_dot_directory_under_home_is_rejected():
    # Ubuntu's tcpdump profile carries `audit deny @{HOME}/.*/** mrwkl`, and deny
    # beats allow -- so a project under ~/.openclaw cannot be written to even though
    # the profile otherwise permits any *.pcap path. This is what turns a working
    # capture into an opaque "Permission denied".
    home = Path.home()
    assert capture.apparmor_blocks(str(home / ".openclaw/work/data/pcaps")) is True
    assert capture.apparmor_blocks(str(home / ".config/x/y.pcap")) is True


def test_an_ordinary_home_path_is_allowed():
    assert capture.apparmor_blocks(str(Path.home() / "work/data/pcaps")) is False


def test_tmp_is_allowed():
    assert capture.apparmor_blocks("/tmp/traffic_pcaps") is False


def test_available_reports_the_apparmor_block_and_names_the_knob(monkeypatch):
    _fake_tcpdump(monkeypatch, alive=True)
    monkeypatch.setenv("TRAFFIC_PCAP_DIR", str(Path.home() / ".openclaw/data/pcaps"))
    ok, reason = capture.available()
    assert ok is False
    assert "AppArmor" in reason
    assert "TRAFFIC_PCAP_DIR" in reason


def test_the_default_pcap_dir_is_somewhere_tcpdump_can_actually_write():
    assert capture.apparmor_blocks(str(capture.pcap_dir().resolve())) is False
