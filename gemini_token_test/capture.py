"""Packet capture around an experiment run, via tcpdump.

Captures real on-wire packets to the Vertex host on tcp/443 while the experiment
runs, writing a .pcap the user can download and open in Wireshark. The TLS payload
is encrypted, but packet sizes + timing are exactly the "real traffic" proof.

Requires the tcpdump binary and raw-socket capability (NET_RAW). Works locally and
in Docker with `--cap-add=NET_RAW`. Does NOT work on Cloud Run (gVisor sandbox has
no raw sockets) — capture is reported unavailable there and the experiment still
runs normally.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

from gemini_client import _vertex_host, LOCATION  # endpoint host for the filter

PCAP_DIR = Path(os.environ.get("PCAP_DIR", "data/pcaps"))
# Filename = timestamp + a high-entropy token so concurrent runs never collide
# and download URLs are unguessable across requests.
_SAFE_NAME = re.compile(r"^capture_(stateless|stateful)_[0-9T\-]+_[0-9a-f]{16}\.pcap$")


def tcpdump_path() -> str | None:
    return shutil.which("tcpdump")


def available() -> tuple[bool, str]:
    """Whether a capture can run. Returns (ok, reason_if_not)."""
    if os.environ.get("PCAP_DISABLE") == "1":
        return False, "capture disabled (PCAP_DISABLE=1)"
    if tcpdump_path() is None:
        return False, "tcpdump not installed"
    return True, "ready"


def _resolve_ips(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        return sorted({i[4][0] for i in infos})
    except Exception:
        return []


def _filter_expr(ips: list[str]) -> str:
    """tcpdump filter: tcp/443 to the Vertex host IP(s), or all 443 if unresolved."""
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
        self.mode = mode if mode in ("stateless", "stateful") else "stateless"
        self.interface = interface or os.environ.get("PCAP_IFACE", "any")
        self.host = _vertex_host()
        self.ips: list[str] = []
        token = secrets.token_hex(8)  # 64-bit: unguessable + collision-proof
        ts = timestamp.replace(":", "-")
        self.path = PCAP_DIR / f"capture_{self.mode}_{ts}_{token}.pcap"
        self.proc: subprocess.Popen | None = None
        self.error = ""

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
            "-U", "-n", _filter_expr(self.ips),
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
        finally:
            self.proc = None

    def result(self) -> dict:
        """Summary for the API response."""
        if self.error:
            return {"ok": False, "error": self.error, "host": self.host,
                    "location": LOCATION}
        size = self.path.stat().st_size if self.path.exists() else 0
        return {
            "ok": size > 0,
            "file": self.path.name,
            "bytes": size,
            "host": self.host,
            "ips": self.ips,
            "filter": _filter_expr(self.ips),
            "note": "" if size > 0 else "no packets captured (mock has no real traffic)",
        }
