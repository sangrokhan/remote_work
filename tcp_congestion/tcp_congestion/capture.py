"""capture: tcpdump packet capture around one conversation run.

Adapted from token_traffic/core/capture.py, simplified to one capture per
run: this project drives a single keep-alive connection per experiment, not
multiple provider/arm combinations, so there is one pcap per run rather than
one per (provider, arm, kind).

Needs the tcpdump binary and raw-socket capability (NET_RAW). Works locally
and in Docker with --cap-add=NET_RAW.

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


def _default_pcap_dir() -> Path:
    return Path(tempfile.gettempdir()) / "tcp_congestion_pcaps"


def pcap_dir() -> Path:
    env = os.environ.get("TRAFFIC_PCAP_DIR")
    return Path(env) if env else _default_pcap_dir()


PCAP_SNAPLEN = int(os.environ.get("TRAFFIC_PCAP_SNAPLEN", "100"))

# Filename = label + timestamp + a high-entropy token, so a download URL is
# not guessable and concurrent runs never collide.
_SAFE_NAME = re.compile(r"^capture_[a-z0-9_-]{1,64}_[0-9A-Za-z\-]{1,64}_[0-9a-f]{16}\.pcap$")
_UNSAFE_STAMP = re.compile(r"[^0-9A-Za-z]+")
_SAFE_LABEL = re.compile(r"^[a-z0-9_-]{1,64}$")

_STAT_RE = re.compile(
    r"(\d+)\s+packets\s+(captured|received by filter|dropped by kernel|dropped by interface)"
)


def _stamp(timestamp: str) -> str:
    return _UNSAFE_STAMP.sub("-", timestamp or "").strip("-")[:64] or "0"


def _parse_tcpdump_stats(text: str) -> dict:
    stats: dict = {}
    for m in _STAT_RE.finditer(text):
        stats[m.group(2).replace(" ", "_")] = int(m.group(1))
    return stats


def tcpdump_path() -> str | None:
    return shutil.which("tcpdump")


_NEVER_MATCHES = "ether proto 0x9999"
_CAP_CACHE: dict = {}


def _iface() -> str:
    return os.environ.get("TRAFFIC_PCAP_IFACE", "any")


def _keep_uid() -> list[str]:
    return ["-Z", getpass.getuser()]


def reset_capability_cache() -> None:
    _CAP_CACHE.clear()


def _probe_tcpdump(path: str) -> bool:
    try:
        proc = subprocess.Popen(
            [path, "-i", _iface(), "-w", os.devnull, "-c", "1", *_keep_uid(),
             _NEVER_MATCHES],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except Exception:
        return False
    time.sleep(0.4)
    alive = proc.poll() is None
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    return alive


def can_raw_capture() -> bool:
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
    if os.environ.get("TRAFFIC_PCAP_DISABLE") == "1":
        return False, "capture disabled (TRAFFIC_PCAP_DISABLE=1)"
    if tcpdump_path() is None:
        return False, "tcpdump not installed"
    if not can_raw_capture():
        return False, _NO_CAP
    return True, "ready"


def _resolve_ips(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        return sorted({i[4][0] for i in infos})
    except Exception:
        return []


def _filter_expr(ips: list[str], port: int) -> str:
    if not ips:
        return f"tcp port {port}"
    hosts = " or ".join(f"host {ip}" for ip in ips)
    return f"tcp port {port} and ({hosts})"


def safe_pcap_path(name: str) -> Path | None:
    if not _SAFE_NAME.match(name or ""):
        return None
    root = pcap_dir()
    p = (root / name).resolve()
    if p.parent != root.resolve():
        return None
    return p if p.exists() else None


class Capture:
    """Context manager around one run's traffic: start tcpdump on enter, stop
    and read its stats on exit. One instance, one pcap, one connection."""

    def __init__(self, timestamp: str, label: str, host: str, port: int = 8888,
                 interface: str | None = None):
        self.timestamp = timestamp
        self.label = label or "conversation"
        self.host = host or ""
        self.port = port
        self.interface = interface or _iface()
        self.ips: list[str] = []
        self.snaplen = PCAP_SNAPLEN
        self.proc: subprocess.Popen | None = None
        self.error = ""
        self.stats: dict = {}
        if not _SAFE_LABEL.match(self.label):
            self.error = f"unsafe_capture_label: {self.label!r}"
            self.path = pcap_dir() / "invalid.pcap"
            return
        token = secrets.token_hex(8)
        self.path = pcap_dir() / f"capture_{self.label}_{_stamp(timestamp)}_{token}.pcap"

    def __enter__(self) -> "Capture":
        if self.error:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        except FileExistsError:
            self.error = "pcap_name_collision"
            return self
        except Exception as exc:
            self.error = f"pcap_open_failed: {exc}"
            return self
        self.ips = _resolve_ips(self.host, self.port)
        cmd = [
            tcpdump_path(), "-i", self.interface, "-w", str(self.path),
            "-s", str(self.snaplen), "-U", "-n", *_keep_uid(),
            _filter_expr(self.ips, self.port),
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except Exception as exc:
            self.error = f"start_failed: {exc}"
            return self
        time.sleep(0.4)
        if self.proc.poll() is not None:
            err = (self.proc.stderr.read() or b"").decode(errors="replace").strip()
            self.error = f"tcpdump_exited: {err[:200]}"
            self.proc = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.proc is None:
            return
        try:
            self.proc.send_signal(signal.SIGINT)
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
        if self.error:
            return {"ok": False, "label": self.label, "error": self.error,
                    "host": self.host, "port": self.port}
        size = self.path.stat().st_size if self.path.exists() else 0
        dropped = (self.stats.get("dropped_by_kernel", 0)
                   + self.stats.get("dropped_by_interface", 0))
        return {
            "ok": size > 0,
            "label": self.label,
            "file": self.path.name,
            "bytes": size,
            "host": self.host,
            "port": self.port,
            "ips": self.ips,
            "filter": _filter_expr(self.ips, self.port),
            "snaplen": self.snaplen,
            "stats": self.stats,
            "dropped": dropped,
            "error": "",
            "note": "" if size > 0 else "no packets captured",
        }
