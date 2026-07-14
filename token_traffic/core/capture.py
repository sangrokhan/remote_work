"""tcpdump around one arm: the packets themselves, as evidence the numbers are real.

A byte count taken inside the process is a claim. A pcap taken outside it, on the
interface, is the thing the claim is about -- the TLS payload is encrypted, but
packet sizes and timing are exactly the traffic being argued over, and they can be
opened in Wireshark by somebody who does not trust this code.

One capture per (provider, arm), because the arms are what the run compares and a
single pcap spanning all of them cannot be attributed after the fact.

Needs the tcpdump binary and raw-socket capability (NET_RAW). Works locally and in
Docker with `--cap-add=NET_RAW`; does not work under gVisor (Cloud Run), where
capture reports itself unavailable and the run proceeds without it.

Knobs: TRAFFIC_PCAP_DIR, TRAFFIC_PCAP_IFACE, TRAFFIC_PCAP_SNAPLEN,
TRAFFIC_PCAP_DISABLE.
"""

from __future__ import annotations

import getpass
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path


def apparmor_blocks(path: str) -> bool:
    """Whether Ubuntu's tcpdump AppArmor profile forbids writing here.

    The profile permits any *.pcap path, but it also carries

        audit deny @{HOME}/.*  mrwkl
        audit deny @{HOME}/.*/** mrwkl

    to keep tcpdump out of dotfiles -- and in AppArmor, deny beats allow. So a
    checkout living under, say, ~/.openclaw/ cannot host the pcap directory at all:
    tcpdump gets NET_RAW, opens the socket, and then dies on the output file with a
    bare "Permission denied" that reads exactly like the capability never took.
    Hours were spent on that misreading; this function exists so nobody spends them
    again.
    """
    home = Path.home()
    try:
        rel = Path(path).resolve().relative_to(home)
    except ValueError:
        return False          # not under $HOME; the deny rules do not apply
    return bool(rel.parts) and rel.parts[0].startswith(".")


def _default_pcap_dir() -> Path:
    """data/pcaps next to the code, unless AppArmor would refuse it -- which it does
    whenever the checkout lives under a dot-directory in $HOME. Fall back to a temp
    dir, which the profile allows, rather than shipping a default that silently
    produces empty captures."""
    local = Path("data/pcaps")
    if apparmor_blocks(str(local.resolve())):
        return Path(tempfile.gettempdir()) / "traffic_pcaps"
    return local


def pcap_dir() -> Path:
    """Where captures are written, read from the environment on every call.

    A module constant frozen at import is a knob that silently does nothing whenever
    the import happens first -- which, in a Flask app and in a test suite, it always
    does. `core.store` re-reads its directory per call for the same reason.
    """
    env = os.environ.get("TRAFFIC_PCAP_DIR")
    return Path(env) if env else _default_pcap_dir()


# Snaplen: bytes kept per packet. The TLS payload is encrypted and therefore useless
# to read, so only the L2-L4 headers and the TLS record header are worth storing.
# Truncating to ~100 bytes slashes the disk I/O per packet, which is the main cause
# of kernel drops (the "previous segment not captured" warnings) under load. Each
# frame still records its original on-wire length, so packet sizes stay exact.
PCAP_SNAPLEN = int(os.environ.get("TRAFFIC_PCAP_SNAPLEN", "100"))
# Filename = label + timestamp + a high-entropy token, so concurrent arms never
# collide and a download URL is not guessable from another one.
#
# The timestamp is an ISO-8601 string, and it arrives carrying `:`, `.` and `+`
# (2026-07-14T09:46:57.837905+00:00). Only the colons used to be replaced, so the dot
# and the plus survived into the filename -- and this pattern, which every download goes
# through, then refused to match the name capture had just written. tcpdump wrote a
# perfectly good pcap, the run recorded it, and /api/pcaps/<name> answered 404 for a file
# sitting on disk. So the timestamp is squeezed through `_stamp` on the way in, and this
# accepts what `_stamp` can emit -- the two must be read together or the bug comes back.
_SAFE_NAME = re.compile(r"^capture_[a-z0-9_-]{1,64}_[0-9A-Za-z\-]{1,64}_[0-9a-f]{16}\.pcap$")

# Anything not alphanumeric becomes a dash, runs collapse, edges are trimmed. Lossy on
# purpose: the filename is a label, and the authoritative timestamp is in the run
# document next to the pcap's entry.
_UNSAFE_STAMP = re.compile(r"[^0-9A-Za-z]+")


def _stamp(timestamp: str) -> str:
    return _UNSAFE_STAMP.sub("-", timestamp or "").strip("-")[:64] or "0"
# The label is provider_arm. Anything outside this alphabet could escape the pcap
# directory once it lands in a filename. Underscores belong here: arm names carry
# them (interaction_inline, responses_stateful), and an earlier version that left
# them out quietly relabelled such arms and shipped a pcap claiming to be a
# different one.
_SAFE_LABEL = re.compile(r"^[a-z0-9_-]{1,64}$")
# tcpdump prints its capture stats to stderr on exit. Parse the drop counts, so a
# lossy pcap announces itself instead of being read as a complete record of the run.
_STAT_RE = re.compile(
    r"(\d+)\s+packets\s+(captured|received by filter|dropped by kernel|dropped by interface)"
)


def _parse_tcpdump_stats(text: str) -> dict:
    """Extract the packet counters from tcpdump's exit summary.

        123 packets captured
        130 packets received by filter
        7 packets dropped by kernel
        0 packets dropped by interface

    Returns whichever of those lines were present; platforms differ on the last one.
    """
    stats: dict = {}
    for m in _STAT_RE.finditer(text):
        stats[m.group(2).replace(" ", "_")] = int(m.group(1))
    return stats


def tcpdump_path() -> str | None:
    return shutil.which("tcpdump")


# A filter no packet can match, so the probe capture stays idle and costs nothing.
_NEVER_MATCHES = "ether proto 0x9999"
_CAP_CACHE: dict = {}


def _iface() -> str:
    return os.environ.get("TRAFFIC_PCAP_IFACE", "any")


def _keep_uid() -> list[str]:
    """A setcap'd tcpdump drops privileges to the unprivileged `tcpdump` user the
    moment it has its socket. That user cannot write into our pcap directory, so the
    capture dies on the output file with "Permission denied" despite having held
    NET_RAW all along. Pin it to the uid that owns the directory."""
    return ["-Z", getpass.getuser()]


def reset_capability_cache() -> None:
    _CAP_CACHE.clear()


def _probe_tcpdump(path: str) -> bool:
    """Start the capture we would start, and see whether it survives."""
    try:
        proc = subprocess.Popen(
            [path, "-i", _iface(), "-w", os.devnull, "-c", "1", *_keep_uid(),
             _NEVER_MATCHES],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except Exception:
        return False
    time.sleep(0.4)  # long enough for it to die on a permission error
    alive = proc.poll() is None
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    return alive


def can_raw_capture() -> bool:
    """Whether *tcpdump* can capture -- not whether this process can.

    setcap grants CAP_NET_RAW to the tcpdump binary, not to us: once granted,
    tcpdump captures happily while a Python AF_PACKET socket here is still refused.
    Probing our own socket would therefore report "unavailable" forever no matter
    what the operator does. Ask tcpdump instead, and cache the answer, because the
    probe spawns a process and this is called on every page render.
    """
    path = tcpdump_path()
    if path is None:
        return False
    ttl = float(os.environ.get("TRAFFIC_CAPTURE_PROBE_TTL", "30"))
    now = time.monotonic()
    if "ok" in _CAP_CACHE and (now - _CAP_CACHE["at"]) < ttl:
        return _CAP_CACHE["ok"]
    ok = _probe_tcpdump(path)
    _CAP_CACHE.update(ok=ok, at=now)
    return ok


_NO_CAP = ("tcpdump installed but it lacks NET_RAW — every capture would fail with "
           "'Operation not permitted'. Grant it with: sudo setcap "
           "cap_net_raw,cap_net_admin+eip $(which tcpdump), run as root, or start "
           "the container with --cap-add=NET_RAW.")


def available() -> tuple[bool, str]:
    """Whether a capture can actually run. Returns (ok, reason_if_not).

    Every "no" names the knob that turns it into a yes: an unexplained unavailable
    is indistinguishable from a bug.
    """
    if os.environ.get("TRAFFIC_PCAP_DISABLE") == "1":
        return False, "capture disabled (TRAFFIC_PCAP_DISABLE=1)"
    if tcpdump_path() is None:
        return False, "tcpdump not installed"
    if not can_raw_capture():
        return False, _NO_CAP
    if apparmor_blocks(str(pcap_dir().resolve())):
        return False, (
            f"AppArmor forbids tcpdump writing to {pcap_dir()} — its profile denies "
            f"@{{HOME}}/.*/** outright, and this path sits under a dot-directory in "
            f"$HOME. Point TRAFFIC_PCAP_DIR at a path outside it "
            f"(e.g. TRAFFIC_PCAP_DIR=/tmp/pcaps)."
        )
    return True, "ready"


def _resolve_ips(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        return sorted({i[4][0] for i in infos})
    except Exception:
        return []


def _filter_expr(ips: list[str]) -> str:
    """tcpdump filter: tcp/443 to the API host's IP(s), or all of 443 if the name
    did not resolve -- a noisier pcap beats no pcap."""
    if not ips:
        return "tcp port 443"
    hosts = " or ".join(f"host {ip}" for ip in ips)
    return f"tcp port 443 and ({hosts})"


def safe_pcap_path(name: str) -> Path | None:
    """Validate a download filename and map it into the pcap directory, no traversal."""
    if not _SAFE_NAME.match(name or ""):
        return None
    root = pcap_dir()
    p = (root / name).resolve()
    if p.parent != root.resolve():
        return None
    return p if p.exists() else None


class Capture:
    """Context manager around one arm: start tcpdump on enter, stop and read its
    stats on exit. One instance, one pcap, one (provider, arm)."""

    def __init__(self, timestamp: str, provider: str, arm: str, host: str,
                 interface: str | None = None):
        self.timestamp = timestamp
        self.provider = provider or ""
        self.arm = arm or ""
        self.label = f"{self.provider}_{self.arm}"
        self.host = host or ""
        self.interface = interface or _iface()
        self.ips: list[str] = []
        self.snaplen = PCAP_SNAPLEN
        self.proc: subprocess.Popen | None = None
        self.error = ""
        self.stats: dict = {}
        # Refuse a label we cannot spell in a filename rather than substituting one
        # we can. The predecessor renamed anything unspellable to a default, which is
        # how an arm once shipped a pcap claiming to be another arm -- silently,
        # which is the only way a mislabelled capture does real damage.
        if not _SAFE_LABEL.match(self.label):
            self.error = f"unsafe_capture_label: {self.label!r}"
            self.path = pcap_dir() / "invalid.pcap"
            return
        token = secrets.token_hex(8)  # 64-bit: unguessable and collision-proof
        self.path = pcap_dir() / f"capture_{self.label}_{_stamp(timestamp)}_{token}.pcap"

    def __enter__(self) -> "Capture":
        if self.error:          # rejected label; never touch the filesystem
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Reserve the path atomically (O_EXCL), so a collision is rejected instead of
        # silently overwriting another arm's capture.
        try:
            os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        except FileExistsError:
            self.error = "pcap_name_collision"
            return self
        except Exception as exc:
            self.error = f"pcap_open_failed: {exc}"
            return self
        self.ips = _resolve_ips(self.host)
        cmd = [
            tcpdump_path(), "-i", self.interface, "-w", str(self.path),
            "-s", str(self.snaplen), "-U", "-n", *_keep_uid(),
            _filter_expr(self.ips),
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except Exception as exc:
            self.error = f"start_failed: {exc}"
            return self
        time.sleep(0.4)  # let tcpdump initialize before any traffic flows
        if self.proc.poll() is not None:
            err = (self.proc.stderr.read() or b"").decode(errors="replace").strip()
            self.error = f"tcpdump_exited: {err[:200]}"
            self.proc = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.proc is None:
            return
        try:
            self.proc.send_signal(signal.SIGINT)  # clean flush of the buffer
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        try:
            err = (self.proc.stderr.read() or b"").decode(errors="replace")
            self.stats = _parse_tcpdump_stats(err)
        except Exception:
            pass
        finally:
            self.proc = None

    def result(self) -> dict:
        """What the capture actually got, for the run document and the UI."""
        if self.error:
            return {"ok": False, "provider": self.provider, "arm": self.arm,
                    "error": self.error, "host": self.host}
        size = self.path.stat().st_size if self.path.exists() else 0
        dropped = (self.stats.get("dropped_by_kernel", 0)
                   + self.stats.get("dropped_by_interface", 0))
        return {
            "ok": size > 0,
            "provider": self.provider,
            "arm": self.arm,
            "file": self.path.name,
            "bytes": size,
            "host": self.host,
            "ips": self.ips,
            "filter": _filter_expr(self.ips),
            "snaplen": self.snaplen,
            "stats": self.stats,
            "dropped": dropped,
            "log": self._log_lines(dropped),
            "error": "",
            "note": "" if size > 0 else "no packets captured (a mock run makes no real traffic)",
        }

    def _log_lines(self, dropped: int) -> list[str]:
        if not self.stats:
            return []
        s = self.stats
        lines = [
            f"tcpdump[{self.label}]: {s.get('captured', '?')} captured, "
            f"{s.get('received_by_filter', '?')} received by filter, "
            f"{dropped} dropped (snaplen={self.snaplen})"
        ]
        if dropped:
            lines.append(
                f"⚠ {dropped} packet(s) dropped during capture — the pcap may show "
                "'previous segment not captured'. The capture was overloaded; try a "
                "quieter host or a smaller TRAFFIC_PCAP_SNAPLEN."
            )
        return lines
