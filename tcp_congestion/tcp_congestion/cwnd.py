"""cwnd: continuous congestion-window monitor for one socket's lifetime.

Adapted from token_traffic/core/cwnd.py, with two changes:
  - no dependency on core.config / core.wire (this project is standalone)
  - `Monitor.announce(sock)` replaces the global connect-watcher mechanism:
    the experiment code calls it once, right after connecting, instead of
    subscribing through a shared registry.

Why netlink and not per-request getsockopt: the reset this project measures
happens *during* the idle gap, when no Python code is running (blocked on a
socket read). A monitor sampling from the request loop would sleep through
the event. The C helper runs on its own clock and its own process, reading
kernel state via sock_diag (the same interface `ss -ti` uses) for a socket
this process does not own -- so nothing about the traffic being measured is
perturbed by measuring it.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

DEFAULT_INTERVAL_MS = 2
DEFAULT_MAX_SAMPLES = 400_000
MAX_SECONDS = 3600

SAMPLE_FIELDS = [
    "t_ms", "wall", "local", "remote", "state", "ca_state",
    "snd_cwnd", "snd_ssthresh", "rcv_ssthresh",
    "rtt_us", "rttvar_us", "min_rtt_us", "rto_us", "ato_us",
    "snd_mss", "rcv_mss", "advmss", "pmtu",
    "unacked", "sacked", "lost", "retrans", "total_retrans", "reordering",
    "bytes_sent", "bytes_acked", "bytes_received", "bytes_retrans",
    "segs_out", "segs_in", "delivered", "delivery_rate", "pacing_rate",
    "snd_wnd", "rwnd_limited_us", "sndbuf_limited_us", "busy_time_us",
    "last_data_sent_ms", "last_data_recv_ms", "last_ack_recv_ms",
    "inode",
]

_SOURCE = Path(__file__).resolve().parent.parent / "native" / "cwnd_monitor.c"
_BINARY = Path(__file__).resolve().parent.parent / "native" / "cwnd_monitor"

_probe: tuple[bool, str] | None = None


def _flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def binary_path() -> Path:
    override = (os.environ.get("TRAFFIC_CWND_BIN") or "").strip()
    return Path(override) if override else _BINARY


def interval_ms() -> int:
    try:
        n = int(os.environ.get("TRAFFIC_CWND_INTERVAL_MS") or DEFAULT_INTERVAL_MS)
    except ValueError:
        return DEFAULT_INTERVAL_MS
    return n if n >= 1 else DEFAULT_INTERVAL_MS


def max_samples() -> int:
    try:
        n = int(os.environ.get("TRAFFIC_CWND_MAX_SAMPLES") or DEFAULT_MAX_SAMPLES)
    except ValueError:
        return DEFAULT_MAX_SAMPLES
    return n if n > 0 else DEFAULT_MAX_SAMPLES


def reset_capability_cache() -> None:
    global _probe
    _probe = None


def build() -> tuple[bool, str]:
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return False, "no C compiler (cc/gcc) on PATH"
    if not _SOURCE.exists():
        return False, f"source missing: {_SOURCE}"
    out = binary_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run([cc, "-O2", "-Wall", "-o", str(out), str(_SOURCE)],
                              capture_output=True, text=True, timeout=120)
    except Exception as exc:
        return False, f"compile failed: {exc}"
    if proc.returncode != 0:
        return False, f"compile failed: {(proc.stderr or '').strip()[:400]}"
    reset_capability_cache()
    return True, f"built {out}"


def available() -> tuple[bool, str]:
    global _probe
    if _probe is not None:
        return _probe

    def answer(ok: bool, why: str) -> tuple[bool, str]:
        global _probe
        _probe = (ok, why)
        return _probe

    if _flag("TRAFFIC_CWND_DISABLE"):
        return answer(False, "disabled by TRAFFIC_CWND_DISABLE")
    if not sys.platform.startswith("linux"):
        return answer(False, f"netlink sock_diag is Linux-only (this is {sys.platform})")

    path = binary_path()
    if not path.exists():
        ok, msg = build()
        if not ok:
            return answer(False, f"helper not built: {msg}")
    if not os.access(str(path), os.X_OK):
        return answer(False, f"helper not executable: {path}")

    try:
        proc = subprocess.run([str(path), "--port", "443", "--max-seconds", "0.02"],
                              capture_output=True, text=True, timeout=15)
    except Exception as exc:
        return answer(False, f"helper would not run: {exc}")
    if proc.returncode != 0:
        return answer(False, f"helper exited {proc.returncode}: "
                             f"{(proc.stderr or '').strip()[:200]}")
    if '"type":"end"' not in (proc.stdout or ""):
        return answer(False, "helper produced no trailer; netlink probably unavailable")
    return answer(True, f"ready ({path})")


def resolve_ips(host: str, port: int = 443) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        return sorted({i[4][0] for i in infos})
    except Exception:
        return []


class Monitor:
    """One connection's congestion samples over its whole lifetime.

    Unlike a per-request TCP_INFO snapshot, this runs continuously on its own
    thread from __enter__ to stop(), so the reset that happens *during* an
    idle gap (while the request loop is blocked on a socket read) is not
    missed. `announce(sock)` tells the helper about a socket immediately, so
    the first (idle-window) samples are not missed either.
    """

    def __init__(self, label: str, host: str, port: int = 443,
                 interval: int | None = None):
        self.label = label
        self.host = host or ""
        self.port = port
        self.interval = interval or interval_ms()

        self.ips: list[str] = []
        self.proc: subprocess.Popen | None = None
        self.samples: list[dict] = []
        self.meta: dict = {}
        self.end: dict = {}
        self.error = ""
        self.truncated = False
        self._reader: threading.Thread | None = None
        self._stderr: list[str] = []
        self._announce_lock = threading.Lock()
        self.announced = 0

    def __enter__(self) -> "Monitor":
        self.ips = resolve_ips(self.host, self.port)
        argv = [str(binary_path()),
                "--port", str(self.port),
                "--interval-ms", str(self.interval),
                "--max-seconds", str(MAX_SECONDS),
                "--label", self.label]
        if self.ips:
            argv += ["--dst", ",".join(self.ips)]

        try:
            self.proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1)
        except Exception as exc:
            self.error = f"monitor would not start: {exc}"
            return self

        self._reader = threading.Thread(target=self._drain, daemon=True,
                                        name=f"cwnd:{self.label}")
        self._reader.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def announce(self, sock: socket.socket) -> None:
        """Tell the helper about a socket the caller just opened, so its very
        first samples (before the next rediscovery tick) are not missed."""
        proc = self.proc
        if proc is None or proc.stdin is None:
            return
        try:
            local_ip, local_port = sock.getsockname()[:2]
            peer_ip, peer_port = sock.getpeername()[:2]
        except Exception:
            return
        if peer_port != self.port:
            return
        local_ip = local_ip.replace("::ffff:", "")
        peer_ip = peer_ip.replace("::ffff:", "")
        line = f"track {local_ip} {local_port} {peer_ip} {peer_port}\n"
        try:
            with self._announce_lock:
                proc.stdin.write(line)
                proc.stdin.flush()
                self.announced += 1
        except Exception:
            pass

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.send_signal(signal.SIGTERM)
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        if self._reader is not None:
            self._reader.join(timeout=5)
            self._reader = None
        try:
            err = (proc.stderr.read() or "").strip() if proc.stderr else ""
        except Exception:
            err = ""
        if err:
            self._stderr.append(err)
        for f in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if f:
                    f.close()
            except Exception:
                pass

    def _drain(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        cap = max_samples()
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                kind = row.get("type")
                if kind == "sample":
                    if len(self.samples) < cap:
                        self.samples.append(row)
                    else:
                        self.truncated = True
                elif kind == "meta":
                    self.meta = row
                elif kind == "end":
                    self.end = row
        except Exception as exc:
            self._stderr.append(f"reader: {exc}")

    def result(self) -> dict:
        err = self.error
        if not err and self._stderr:
            err = "; ".join(self._stderr)[:400]

        sockets = sorted({s.get("local", "") for s in self.samples if s.get("local")})
        out = {
            "label": self.label,
            "host": self.host,
            "port": self.port,
            "ips": self.ips,
            "interval_ms": self.interval,
            "samples": self.samples,
            "sample_count": len(self.samples),
            "ticks": self.end.get("ticks", 0),
            "seconds": self.end.get("seconds", 0),
            "announced": self.announced,
            "sockets": sockets,
            "truncated": self.truncated,
            "error": err,
        }
        out.update(idle_resets(self.samples))
        return out


INIT_CWND = 10


def idle_resets(samples: list[dict]) -> dict:
    """Count the drops from a grown window back to the initial one, per socket.

    Also returns `reset_events` with the exact t_ms of each reset -- the
    signal that "the *next* transmission after idle re-enters slow start".
    """
    peak: dict[str, int] = {}
    state: dict[str, int] = {}
    resets = 0
    reset_rows: list[dict] = []

    for s in samples:
        key = s.get("local") or ""
        cwnd = s.get("snd_cwnd")
        if not key or not isinstance(cwnd, int):
            continue
        prev = state.get(key)
        peak[key] = max(peak.get(key, 0), cwnd)
        if (prev is not None and prev > INIT_CWND and cwnd <= INIT_CWND
                and s.get("ca_state") == "open"):
            resets += 1
            reset_rows.append({"t_ms": s.get("t_ms"), "local": key,
                               "from": prev, "to": cwnd,
                               "idle_ms": s.get("last_data_sent_ms")})
        state[key] = cwnd

    return {
        "idle_resets": resets,
        "reset_events": reset_rows,
        "peak_cwnd": max(peak.values()) if peak else 0,
        "final_cwnd": max(state.values()) if state else 0,
    }
