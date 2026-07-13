"""Packet capture around an experiment run, via tcpdump.

Captures real on-wire packets to the Gemini API host on tcp/443 while the experiment
runs, writing a .pcap the user can download and open in Wireshark. The TLS payload
is encrypted, but packet sizes + timing are exactly the "real traffic" proof.

Requires the tcpdump binary and raw-socket capability (NET_RAW). Works locally and
in Docker with `--cap-add=NET_RAW`. Does NOT work on Cloud Run (gVisor sandbox has
no raw sockets) — capture is reported unavailable there and the experiment still
runs normally.
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

from gemini_client import api_host  # endpoint host for the filter

def apparmor_blocks(path: str) -> bool:
    """Whether Ubuntu's tcpdump AppArmor profile forbids writing here.

    The profile permits any *.pcap path, but it also carries

        audit deny @{HOME}/.*  mrwkl
        audit deny @{HOME}/.*/** mrwkl

    to keep tcpdump out of dotfiles -- and in AppArmor deny beats allow. So a
    project checked out under, say, ~/.openclaw/ cannot host the pcap directory at
    all: tcpdump gets NET_RAW, opens the socket, and then dies on the output file
    with a bare "Permission denied" that looks like the capability never took.
    """
    home = Path.home()
    try:
        rel = Path(path).resolve().relative_to(home)
    except ValueError:
        return False          # not under $HOME; the deny rules do not apply
    return bool(rel.parts) and rel.parts[0].startswith(".")


def _default_pcap_dir() -> Path:
    """data/pcaps next to the code, unless AppArmor would refuse it -- which it
    does whenever the checkout lives under a dot-directory in $HOME. Fall back to
    a temp dir, which the profile allows, rather than shipping a default that
    silently produces empty captures."""
    local = Path("data/pcaps")
    if apparmor_blocks(str(local.resolve())):
        return Path(tempfile.gettempdir()) / "gemini_pcaps"
    return local


PCAP_DIR = Path(os.environ["PCAP_DIR"]) if os.environ.get("PCAP_DIR") else _default_pcap_dir()
# Snaplen: bytes captured per packet. TLS payload is encrypted (unreadable), so we
# only need L2-L4 headers + TLS record header for sizes/timing. Truncating to ~100
# bytes slashes disk I/O per packet, which is the main cause of kernel drops (the
# "ACKed unseen segment" / "previous segment not captured" warnings) under load.
# The original on-wire length is still recorded in each frame, so packet sizes stay
# exact in Wireshark.
PCAP_SNAPLEN = int(os.environ.get("PCAP_SNAPLEN", "100"))
# Filename = timestamp + a high-entropy token so concurrent runs never collide
# and download URLs are unguessable across requests.
_SAFE_NAME = re.compile(r"^capture_[a-z0-9-]{1,32}_[0-9T\-]+_[0-9a-f]{16}\.pcap$")
# The label names the run being captured (an arm, a stage). Anything outside this
# alphabet could escape the pcap directory once it lands in a filename.
_SAFE_LABEL = re.compile(r"^[a-z0-9-]{1,32}$")
# tcpdump prints capture stats to stderr on exit (SIGINT). Parse the drop counts so
# the UI can surface capture loss instead of silently producing a lossy pcap.
_STAT_RE = re.compile(
    r"(\d+)\s+packets\s+(captured|received by filter|dropped by kernel|dropped by interface)"
)


def _parse_tcpdump_stats(text: str) -> dict:
    """Extract packet counters from tcpdump's stderr summary.

    tcpdump exit summary looks like:
        123 packets captured
        130 packets received by filter
        7 packets dropped by kernel
        0 packets dropped by interface
    Returns {captured, received_by_filter, dropped_by_kernel, dropped_by_interface}
    for whichever lines are present.
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


def _keep_uid() -> list[str]:
    """A setcap'd tcpdump drops privileges to the unprivileged `tcpdump` user as
    soon as it has the socket. That user cannot write into our pcap directory, so
    the capture dies on the output file with "Permission denied" despite having had
    NET_RAW all along. Pin it to the uid that owns the directory."""
    return ["-Z", getpass.getuser()]


def reset_capability_cache() -> None:
    _CAP_CACHE.clear()


def _probe_tcpdump(path: str) -> bool:
    """Start the capture we would start, and see whether it survives."""
    try:
        proc = subprocess.Popen(
            [path, "-i", os.environ.get("PCAP_IFACE", "any"), "-w", os.devnull,
             "-c", "1", *_keep_uid(), _NEVER_MATCHES],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except Exception:
        return False
    time.sleep(0.4)  # give it long enough to die on a permission error
    alive = proc.poll() is None
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    return alive


def can_raw_capture() -> bool:
    """Whether *tcpdump* can capture — not whether this process can.

    setcap grants CAP_NET_RAW to the tcpdump binary, not to us: once it is granted,
    tcpdump captures happily while a Python AF_PACKET socket here is still refused.
    So probing our own socket would report "unavailable" forever no matter what the
    operator does. Ask tcpdump instead, and cache the answer — the probe spawns a
    process, and this is called on every page render.
    """
    path = tcpdump_path()
    if path is None:
        return False
    ttl = float(os.environ.get("CAPTURE_PROBE_TTL", "30"))
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
    """Whether a capture can run. Returns (ok, reason_if_not)."""
    if os.environ.get("PCAP_DISABLE") == "1":
        return False, "capture disabled (PCAP_DISABLE=1)"
    if tcpdump_path() is None:
        return False, "tcpdump not installed"
    if not can_raw_capture():
        return False, _NO_CAP
    if apparmor_blocks(str(PCAP_DIR.resolve())):
        return False, (
            f"AppArmor forbids tcpdump writing to {PCAP_DIR} — its profile denies "
            f"@{{HOME}}/.*/** outright, and this path sits under a dot-directory in "
            f"$HOME. Point PCAP_DIR at a path outside it (e.g. PCAP_DIR=/tmp/pcaps)."
        )
    return True, "ready"


def _resolve_ips(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        return sorted({i[4][0] for i in infos})
    except Exception:
        return []


def _filter_expr(ips: list[str]) -> str:
    """tcpdump filter: tcp/443 to the API host IP(s), or all 443 if unresolved."""
    if not ips:
        return "tcp port 443"
    hosts = " or ".join(f"host {ip}" for ip in ips)
    return f"tcp port 443 and ({hosts})"


def safe_pcap_path(name: str) -> Path | None:
    """Validate a download filename and map it into PCAP_DIR (no traversal)."""
    if not _SAFE_NAME.match(name):
        return None
    p = (PCAP_DIR / name).resolve()
    if p.parent != PCAP_DIR.resolve():
        return None
    return p if p.exists() else None


class Capture:
    """Context manager: start tcpdump on enter, stop + finalize on exit."""

    def __init__(self, timestamp: str, mode: str = "stateless",
                 interface: str | None = None):
        self.timestamp = timestamp
        self.mode = mode if _SAFE_LABEL.match(mode or "") else "stateless"
        self.interface = interface or os.environ.get("PCAP_IFACE", "any")
        self.host = api_host()
        self.ips: list[str] = []
        token = secrets.token_hex(8)  # 64-bit: unguessable + collision-proof
        ts = timestamp.replace(":", "-")
        self.path = PCAP_DIR / f"capture_{self.mode}_{ts}_{token}.pcap"
        self.snaplen = PCAP_SNAPLEN
        self.proc: subprocess.Popen | None = None
        self.error = ""
        self.stats: dict = {}

    def __enter__(self) -> "Capture":
        PCAP_DIR.mkdir(parents=True, exist_ok=True)
        # Reserve the path atomically (O_EXCL) so a collision is rejected, never
        # silently overwriting another request's capture.
        try:
            os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        except FileExistsError:
            self.error = "pcap_name_collision"
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
        time.sleep(0.4)  # let tcpdump initialize before traffic flows
        # If it died immediately (e.g. permission denied), capture stderr.
        if self.proc.poll() is not None:
            err = (self.proc.stderr.read() or b"").decode(errors="replace").strip()
            self.error = f"tcpdump_exited: {err[:200]}"
            self.proc = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.proc is None:
            return
        try:
            self.proc.send_signal(signal.SIGINT)  # clean flush
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        # tcpdump writes its capture/drop summary to stderr on exit; read it so the
        # UI can report packets dropped by kernel/interface (capture loss).
        try:
            err = (self.proc.stderr.read() or b"").decode(errors="replace")
            self.stats = _parse_tcpdump_stats(err)
        except Exception:
            pass
        finally:
            self.proc = None

    def result(self) -> dict:
        """Summary for the API response."""
        if self.error:
            return {"ok": False, "error": self.error, "host": self.host}
        size = self.path.stat().st_size if self.path.exists() else 0
        dropped = (self.stats.get("dropped_by_kernel", 0)
                   + self.stats.get("dropped_by_interface", 0))
        log = self._log_lines(dropped)
        return {
            "ok": size > 0,
            "file": self.path.name,
            "bytes": size,
            "host": self.host,
            "ips": self.ips,
            "filter": _filter_expr(self.ips),
            "snaplen": self.snaplen,
            "stats": self.stats,
            "dropped": dropped,
            "log": log,
            "note": "" if size > 0 else "no packets captured (mock has no real traffic)",
        }

    def _log_lines(self, dropped: int) -> list[str]:
        """Human-readable capture log lines for the UI."""
        if not self.stats:
            return []
        s = self.stats
        lines = [
            f"tcpdump[{self.mode}]: {s.get('captured', '?')} captured, "
            f"{s.get('received_by_filter', '?')} received by filter, "
            f"{dropped} dropped (snaplen={self.snaplen})"
        ]
        if dropped:
            lines.append(
                f"⚠ {dropped} packet(s) dropped during capture — pcap may show "
                "'ACKed unseen segment' / 'previous segment not captured'. "
                "Capture overloaded; try larger PCAP_SNAPLEN headroom or a quieter host."
            )
        return lines
