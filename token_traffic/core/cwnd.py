"""The kernel's congestion state during one arm, sampled by a C helper on its own thread.

What this answers: an LLM turn is mostly idle. The client uploads a prompt in a few
milliseconds and then waits seconds for the server to think. With
`net.ipv4.tcp_slow_start_after_idle=1` -- the Linux default -- the kernel throws away
the congestion window whenever the idle gap exceeds one RTO, so every turn after the
first re-enters slow start and re-earns a window it already had. On a long conversation
that is a round-trip tax on every turn, and it is invisible to everything else this
package measures: the byte counts are identical either way, and the latency marks show
the *symptom* without naming the cause.

`native/cwnd_monitor.c` reads that state over netlink sock_diag -- the same interface
`ss -ti` uses -- so the Python client is not touched and not perturbed. This module owns
its lifecycle: start it before the measured window, read its NDJSON on a reader thread
while the turns run, stop it after, and hand back the samples.

Why a thread and not a poll at turn boundaries: the reset happens *during* the idle
gap, which is precisely when no Python code is running -- the request is blocked on a
socket read. Anything sampling from the main thread would sleep through the event it
is trying to observe. The helper runs as its own process on its own clock, and the
reader thread only has to drain a pipe.

Absence is not failure. Monitoring is best-effort in exactly the way capture is: a box
without the compiled binary, or a container without netlink, runs the experiment
anyway and says why the column is empty.

Knobs: TRAFFIC_CWND_BIN, TRAFFIC_CWND_INTERVAL_MS, TRAFFIC_CWND_MAX_SAMPLES,
TRAFFIC_CWND_DISABLE.
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

from core import config

# 10 ms. Fast enough that an idle gap of one RTO (200 ms and up) is resolved into
# dozens of samples, slow enough that the helper costs a fraction of a core.
DEFAULT_INTERVAL_MS = 10

# A ceiling on what one arm may keep, because the samples ride in the run document and
# a run document is written to disk and served over HTTP. At 10 ms with a couple of
# sockets in flight, this is roughly half an hour of monitoring -- far past any run --
# and an arm that hits it says so in its result rather than quietly dropping the tail.
DEFAULT_MAX_SAMPLES = 400_000

# The helper is killed when the arm ends. This is the backstop for the case where it
# is not: a monitor orphaned by a crashed run must not outlive the machine's patience.
MAX_SECONDS = 3600

# Every numeric field the helper emits, in the order a reader wants them: what the
# window is doing first, then why, then the counters that corroborate it.
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

# available() shells out to the helper, and /api/config asks on every page load. The
# answer only changes when somebody rebuilds the binary, so it is cached the way
# capture caches its tcpdump probe -- with a reset for the tests that toggle it.
_probe: tuple[bool, str] | None = None


def binary_path() -> Path:
    """The helper binary. TRAFFIC_CWND_BIN overrides, for a container that builds it
    somewhere else."""
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
    """Forget the probe result. For tests, and for a rebuild mid-session."""
    global _probe
    _probe = None


def build() -> tuple[bool, str]:
    """Compile the helper. Returns (ok, message).

    Kept here rather than left to the Makefile alone so that a checkout which has never
    run `make cwnd` can still be asked to monitor -- and so the compiler's own error
    reaches the operator instead of a bare "binary missing".
    """
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
    except Exception as exc:                      # compiler missing mid-flight, timeout
        return False, f"compile failed: {exc}"
    if proc.returncode != 0:
        return False, f"compile failed: {(proc.stderr or '').strip()[:400]}"
    reset_capability_cache()
    return True, f"built {out}"


def available() -> tuple[bool, str]:
    """Whether this box can monitor congestion state, and if not, why not.

    The check actually runs the helper for 20 ms rather than merely stat-ing the
    binary. A binary that exists but cannot open a netlink socket -- gVisor, a locked
    down container, a seccomp profile without AF_NETLINK -- would otherwise be
    discovered at the start of a paid run, which is the worst possible moment.
    """
    global _probe
    if _probe is not None:
        return _probe

    def answer(ok: bool, why: str) -> tuple[bool, str]:
        global _probe
        _probe = (ok, why)
        return _probe

    if config.flag("TRAFFIC_CWND_DISABLE"):
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
    """The addresses the client will actually connect to.

    Resolved here and passed to the helper as literals, so the helper never does DNS
    of its own: if the name resolves differently in the two processes, the monitor
    would watch a socket the client never opened and report an empty run as a quiet
    zero.
    """
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        return sorted({i[4][0] for i in infos})
    except Exception:
        return []


class Monitor:
    """One arm's congestion samples: start the helper on enter, stop it on exit.

    Mirrors core.capture.Capture on purpose -- same lifecycle, same "record the failure
    and carry on" contract -- because the runner drives both from the same window and a
    monitor that raised where a capture merely reported would turn a missing column
    into a failed run.
    """

    def __init__(self, provider: str, arm: str, host: str, kind: str = "",
                 port: int = 443, interval: int | None = None):
        self.provider = provider or ""
        self.arm = arm or ""
        self.kind = kind or ""
        self.host = host or ""
        self.port = port
        self.interval = interval or interval_ms()
        self.label = f"{self.provider}:{self.arm}" + (f":{self.kind}" if self.kind else "")

        self.ips: list[str] = []
        self.proc: subprocess.Popen | None = None
        self.samples: list[dict] = []
        self.meta: dict = {}
        self.end: dict = {}
        self.error = ""
        self.truncated = False
        self._reader: threading.Thread | None = None
        self._stderr: list[str] = []

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Monitor":
        self.ips = resolve_ips(self.host, self.port)
        argv = [str(binary_path()),
                "--port", str(self.port),
                "--interval-ms", str(self.interval),
                "--max-seconds", str(MAX_SECONDS),
                "--label", self.label]
        if self.ips:
            argv += ["--dst", ",".join(self.ips)]
        # No --dst when the name did not resolve: watching every socket on :443 is
        # noisy, and the run document says which host was meant, so a reader can tell
        # the noise from the signal. Watching nothing would just be a silent zero.

        try:
            self.proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1)
        except Exception as exc:
            self.error = f"monitor would not start: {exc}"
            return self

        self._reader = threading.Thread(target=self._drain, daemon=True,
                                        name=f"cwnd:{self.label}")
        self._reader.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def stop(self) -> None:
        """Idempotent. SIGTERM, not kill: the helper writes its trailer on the way out,
        and the trailer is how we learn how many ticks it actually managed -- which is
        the only way to tell a monitor that watched an idle socket from one that was
        never running."""
        proc, self.proc = self.proc, None
        if proc is None:
            return
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
        for f in (proc.stdout, proc.stderr):
            try:
                if f:
                    f.close()
            except Exception:
                pass

    def _drain(self) -> None:
        """Read the helper's NDJSON until it closes. Runs on the reader thread.

        A malformed line is skipped rather than fatal: the helper writes diagnostics to
        stderr precisely so stdout stays parseable, but a monitor that killed an arm
        over one bad line would be worse than one that lost a sample.
        """
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
        except Exception as exc:                  # pipe torn down under us on stop()
            self._stderr.append(f"reader: {exc}")

    # -- result ------------------------------------------------------------

    def result(self) -> dict:
        """What the run document keeps for this arm.

        `sockets` is the distinct peers seen, `resets` the number of times a socket's
        window dropped to a value at or below the initial 10 after having been larger --
        the event this whole module exists to count, computed here rather than in the
        UI so the CSV and the summary cannot disagree about it.
        """
        err = self.error
        if not err and self._stderr:
            err = "; ".join(self._stderr)[:400]

        sockets = sorted({s.get("local", "") for s in self.samples if s.get("local")})
        out = {
            "label": self.label,
            "provider": self.provider,
            "arm": self.arm,
            "kind": self.kind,
            "host": self.host,
            "port": self.port,
            "ips": self.ips,
            "interval_ms": self.interval,
            "samples": self.samples,
            "sample_count": len(self.samples),
            "ticks": self.end.get("ticks", 0),
            "seconds": self.end.get("seconds", 0),
            "sockets": sockets,
            "truncated": self.truncated,
            "error": err,
        }
        out.update(idle_resets(self.samples))
        return out


# The value Linux starts a connection at, and returns it to after idle: 10 segments,
# from RFC 6928 and unchanged in mainline since 2.6.39. A window that was larger and
# is now at or below this has been reset, not merely reduced.
INIT_CWND = 10


def idle_resets(samples: list[dict]) -> dict:
    """Count the drops from a grown window back to the initial one, per socket.

    A drop only counts when the congestion-control state is `open` -- i.e. the kernel
    is not in recovery. A window that collapsed because of loss is a different finding
    with a different fix, and folding the two together would let a lossy network
    masquerade as the idle-reset problem (or hide it).
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

