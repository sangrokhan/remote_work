"""available() has to mean "a capture will actually work".

Two ways to get this wrong, and the first version got the second one:

1. tcpdump being installed says nothing -- capturing needs CAP_NET_RAW.
2. Asking whether *we* can open a raw socket also says nothing. `setcap` grants
   the capability to the tcpdump binary, not to this process: with caps set,
   tcpdump captures happily while a Python AF_PACKET socket is still refused.
   Probing our own socket therefore reports "unavailable" forever, no matter what
   the operator does.

So the probe has to ask tcpdump, by starting the capture it would start and
seeing whether it survives.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import capture


class _Proc:
    """A fake tcpdump. alive=False means it died on startup (no permission)."""

    def __init__(self, alive=True):
        self._alive = alive
        self.stderr = _Stderr()
        self.killed = False

    def poll(self):
        return None if self._alive else 1

    def terminate(self):
        self.killed = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


class _Stderr:
    def read(self):
        return b"tcpdump: any: You don't have permission to perform this capture"


def _fake_tcpdump(monkeypatch, alive):
    proc = _Proc(alive=alive)
    monkeypatch.setattr(capture, "tcpdump_path", lambda: "/usr/bin/tcpdump")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(capture.time, "sleep", lambda s: None)
    capture.reset_capability_cache()
    monkeypatch.delenv("PCAP_DISABLE", raising=False)
    return proc


def test_tcpdump_that_survives_startup_can_capture(monkeypatch):
    _fake_tcpdump(monkeypatch, alive=True)
    assert capture.can_raw_capture() is True


def test_tcpdump_that_dies_on_startup_cannot_capture(monkeypatch):
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
    monkeypatch.delenv("PCAP_DISABLE", raising=False)
    capture.reset_capability_cache()
    capture.can_raw_capture()
    capture.can_raw_capture()
    assert calls["n"] == 1


def test_granting_caps_is_picked_up_once_the_cache_expires(monkeypatch):
    # The operator runs setcap while the server is up; the page must stop saying
    # "unavailable" without a restart.
    monkeypatch.setenv("CAPTURE_PROBE_TTL", "0")
    _fake_tcpdump(monkeypatch, alive=False)
    assert capture.can_raw_capture() is False
    _fake_tcpdump(monkeypatch, alive=True)
    assert capture.can_raw_capture() is True


def test_missing_binary_is_unavailable(monkeypatch):
    monkeypatch.delenv("PCAP_DISABLE", raising=False)
    monkeypatch.setattr(capture, "tcpdump_path", lambda: None)
    capture.reset_capability_cache()
    ok, reason = capture.available()
    assert ok is False
    assert "not installed" in reason


def test_no_capability_is_reported_with_the_fix(monkeypatch):
    _fake_tcpdump(monkeypatch, alive=False)
    ok, reason = capture.available()
    assert ok is False
    assert "NET_RAW" in reason
    assert "setcap" in reason


def test_capable_tcpdump_is_available(monkeypatch):
    _fake_tcpdump(monkeypatch, alive=True)
    assert capture.available() == (True, "ready")


def test_disable_flag_still_wins(monkeypatch):
    _fake_tcpdump(monkeypatch, alive=True)
    monkeypatch.setenv("PCAP_DISABLE", "1")
    ok, _ = capture.available()
    assert ok is False


def test_tcpdump_keeps_our_uid_so_it_can_write_the_pcap(monkeypatch):
    # A setcap'd tcpdump drops privileges to the unprivileged `tcpdump` user by
    # default. That user cannot write into our pcap directory, so the capture dies
    # with "Permission denied" on the output file even though it had NET_RAW. Pin
    # it to the uid that owns the directory.
    import getpass
    argv = {}
    monkeypatch.setattr(capture, "tcpdump_path", lambda: "/usr/bin/tcpdump")

    class _P:
        def __init__(self, cmd, **k):
            argv["cmd"] = cmd
            self.stderr = _Stderr()

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(subprocess, "Popen", _P)
    monkeypatch.setattr(capture.time, "sleep", lambda s: None)
    cap = capture.Capture("2026-07-13T00:00:00", mode="stateless")
    with cap:
        pass
    cmd = argv["cmd"]
    assert "-Z" in cmd
    assert cmd[cmd.index("-Z") + 1] == getpass.getuser()


# --- AppArmor: where tcpdump is allowed to write --------------------------

def test_a_dot_directory_under_home_is_rejected():
    # Ubuntu's tcpdump AppArmor profile carries `audit deny @{HOME}/.*/** mrwkl`,
    # and deny beats allow -- so a project living under e.g. ~/.openclaw cannot be
    # written to even though the profile otherwise permits any *.pcap path. This is
    # what turns a working capture into an opaque "Permission denied".
    assert capture.apparmor_blocks("/home/han/.openclaw/work/data/pcaps") is True
    assert capture.apparmor_blocks("/home/han/.config/x/y.pcap") is True


def test_an_ordinary_home_path_is_allowed():
    assert capture.apparmor_blocks("/home/han/work/data/pcaps") is False


def test_tmp_is_allowed():
    assert capture.apparmor_blocks("/tmp/gemini_pcaps") is False


def test_available_reports_the_apparmor_block_with_the_fix(monkeypatch):
    _fake_tcpdump(monkeypatch, alive=True)
    monkeypatch.setattr(capture, "PCAP_DIR", Path("/home/han/.openclaw/data/pcaps"))
    ok, reason = capture.available()
    assert ok is False
    assert "AppArmor" in reason
    assert "PCAP_DIR" in reason      # name the knob that fixes it


def test_default_pcap_dir_is_not_apparmor_blocked():
    # Whatever the default is, it must be somewhere tcpdump can actually write.
    assert capture.apparmor_blocks(str(capture.PCAP_DIR.resolve())) is False
