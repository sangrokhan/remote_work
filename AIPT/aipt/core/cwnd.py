"""cwnd: continuous congestion-window monitor for one socket's lifetime.

What this answers: an LLM turn (or any long-lived request/response exchange) is
mostly idle. The client uploads a prompt/request in a few milliseconds and then
waits seconds for the server to think/respond. With
`net.ipv4.tcp_slow_start_after_idle=1` -- the Linux default -- the kernel throws
away the congestion window whenever the idle gap exceeds one RTO, so every turn
after the first re-enters slow start and re-earns a window it already had. On a
long conversation that is a round-trip tax on every turn, and it is invisible to
everything else this package measures: the byte counts are identical either way,
and the latency marks show the *symptom* without naming the cause.

`native/cwnd_monitor.c` reads that state over netlink sock_diag -- the same
interface `ss -ti` uses -- so the Python client is not touched and not
perturbed. This module owns its lifecycle: start it before the measured window,
read its NDJSON on a reader thread while the turns run, stop it after, and hand
back the samples.

Why a thread and not a poll at turn boundaries: the reset happens *during* the
idle gap, which is precisely when no Python code is running -- the request is
blocked on a socket read. Anything sampling from the main thread would sleep
through the event it is trying to observe. The helper runs as its own process on
its own clock, and the reader thread only has to drain a pipe.

Absence is not failure. Monitoring is best-effort in exactly the way capture is: a
box without the compiled binary, or a container without netlink, runs the
experiment anyway and says why the column is empty.

Cost. The helper does not dump the kernel's socket table every tick -- it dumps
once to find the sockets talking to the measured host, then asks for those by
4-tuple, which is a hash lookup rather than a walk of every bucket. Measured on
this package's bench, one tracked socket: a 2 ms period costs 7% of a core, where
dumping every tick could not hold 2 ms at all and burned 100%. `dumps` and
`exact_queries` in `result()` are how a caller checks its own run stayed on the
cheap path instead of taking that on faith.

Label: a single opaque string, not a structured (provider, arm, kind) tuple.
This module has no opinion about what the label means -- a caller that wants
`"openai:stateless:bytes"` assembles that string itself (e.g.
`f"{provider}:{arm}:{kind}"`); a caller with one connection per experiment
(no per-arm structure) just passes a run id. Keeping the label opaque here is
what lets one `cwnd.py` serve both shapes of caller without either carrying
fields it does not need.

Adapted from two prior, independently-forked copies of this module
(`token_traffic/core/cwnd.py`, `tcp_congestion/tcp_congestion/cwnd.py`) that had
drifted apart: this version keeps the simplified, dependency-free interface
(`Monitor.announce(sock)` instead of a shared connect-watcher registry, no
dependency on `core.config`/`core.wire`) and restores the detailed design
rationale and the `dumps`/`exact_queries`/`tracked` instrumentation fields that
had been dropped along the way.

Why netlink and not per-request getsockopt: the reset happens *during* the idle
gap, when no Python code is running (blocked on a socket read). A monitor
sampling from the request loop would sleep through the event. The C helper runs
on its own clock and its own process, reading kernel state via sock_diag (the
same interface `ss -ti` uses) for a socket this process does not own -- so
nothing about the traffic being measured is perturbed by measuring it.

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

# 2 ms. Not chosen against the idle gap -- that is seconds long and a slow sampler
# would find it -- but against the path's RTT, because the reset is only visible until
# slow start doubles the window back. A CDN edge measured a few ms away can climb from
# 10 segments to 65 in about 10 ms: a 10 ms sampler steps over the event entirely,
# which is exactly what a run against a real API host showed.
#
# 2 ms used to be unaffordable and silently became 5 ms under load, because every tick
# dumped the whole TCP hash table. Now that a tick is hash lookups, 2 ms costs 7% of a
# core against 100% before -- measured, one tracked socket, in this package's bench.
DEFAULT_INTERVAL_MS = 2

# A ceiling on what one monitor may keep, because the samples ride in the run
# document and a run document is written to disk and served over HTTP. At 2 ms
# with a couple of sockets in flight this is roughly six minutes of monitoring,
# and a run that hits it says so in its result rather than quietly dropping the
# tail.
DEFAULT_MAX_SAMPLES = 400_000

# The helper is killed when the monitored window ends. This is the backstop for
# the case where it is not: a monitor orphaned by a crashed run must not outlive
# the machine's patience.
MAX_SECONDS = 3600

# Every numeric field the helper emits, in the order a reader wants them: what the
# window is doing first, then why, then the counters that corroborate it. This is
# NOT the literal order native/cwnd_monitor.c's emit_sample() prints them in --
# the C helper's printf puts rto_us/ato_us right before inode (after
# last_data_sent_ms/last_data_recv_ms/last_ack_recv_ms), while this list places
# them right after min_rtt_us instead. That's harmless: `_drain()` parses the
# NDJSON line with json.loads() into a dict, so the JSON key/value mapping does
# not depend on printf order at all -- this list only fixes the column order
# used when exporting to CSV (see CONNECTION_COLUMNS in
# aipt/export/connection.py), which is a reader-ergonomics choice independent
# of the wire order. The two field *sets* are identical (40 keys each).
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

_SOURCE = Path(__file__).resolve().parent.parent.parent / "native" / "cwnd_monitor.c"
_BINARY = Path(__file__).resolve().parent.parent.parent / "native" / "cwnd_monitor"

# available() shells out to the helper, and a config endpoint may ask on every page
# load. The answer only changes when somebody rebuilds the binary, so it is cached
# the way capture caches its tcpdump probe -- with a reset for the tests that
# toggle it.
_probe: tuple[bool, str] | None = None


def _flag(name: str) -> bool:
    """Read a boolean env flag without depending on a shared config module.

    This project's core has no external dependency (deliberately -- see module
    docstring), so the small `_flag` helper is duplicated here rather than
    imported from a `core.config` module.
    """
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


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


# B12 (DESIGN.md 4.9): the fixed 2 ms period above is right for the paths this
# module was tuned against (a CDN edge a few ms away), but a backend whose RTT is
# an order of magnitude shorter or longer needs a period scaled to *its* path, not
# to that one. `interval_from_rtt` is that scaling, and it is opt-in: a caller
# that does not pass `rtt_hint_ms` to `Monitor` gets exactly the old fixed-2ms
# behaviour, unchanged.
#
# K, the divisor: this module's own docstring justifies 2 ms against a path where
# the window climbs from 10 segments to 65 in about 10 ms, and a 2 ms sampler
# lands roughly 5 samples across that 10 ms recovery window (10 / 2 = 5). K=5.0
# reproduces that same "5 samples across the burst" ratio directly from the RTT
# instead of from a hardcoded constant: `interval_ms = rtt_ms / K` means a burst
# whose visible window is on the order of one RTT gets sampled about 5 times
# before slow start has doubled it away. A caller with a good RTT estimate (a
# gateway-injected delay, or a measured RTT from a prior turn) gets a period that
# tracks its own path instead of inheriting the CDN-edge number; a caller with no
# such estimate keeps using the fixed default.
DEFAULT_RTT_K = 5.0


def interval_from_rtt(rtt_ms: float, k: float = DEFAULT_RTT_K,
                       min_interval_ms: int = 1) -> tuple[int, str]:
    """Adaptive cwnd sampling period from a path's expected RTT (DESIGN.md 4.9, B12).

    `interval_ms = max(min_interval_ms, rtt_ms / k)`. K is how many times a
    slow-start burst on the order of one RTT should be sampled before it is
    over -- see `DEFAULT_RTT_K` for why 5.0 is the default.

    Extremely short paths (loopback, a container talking to itself, RTT well
    under 1 ms) push `rtt_ms / k` below `min_interval_ms`, the netlink-tick
    floor below which a sample costs more than the thing it is measuring is
    worth. When that happens the result is clamped to `min_interval_ms` and
    the reason is `"floor_clamped"` rather than `"adaptive:..."` -- the caller
    is told the number is a floor, not a measurement, so nothing downstream
    reports a precision the sampler could not actually deliver.

    Returns `(interval_ms, reason)`:
      - `rtt_ms` missing/non-positive: `(DEFAULT_INTERVAL_MS, "fixed")` -- no
        usable hint, fall back to the module default rather than guessing.
      - clamped to the floor: `(min_interval_ms, "floor_clamped")`.
      - otherwise: `(round(rtt_ms / k), "adaptive:rtt=<rtt_ms>ms")`.
    """
    if rtt_ms is None or rtt_ms <= 0:
        return DEFAULT_INTERVAL_MS, "fixed"
    computed = rtt_ms / k
    if computed < min_interval_ms:
        return min_interval_ms, "floor_clamped"
    interval = int(round(computed))
    if interval < min_interval_ms:
        interval = min_interval_ms
    return interval, f"adaptive:rtt={rtt_ms}ms"


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

    Kept here rather than left to the Makefile alone so that a checkout which has
    never run `make cwnd` can still be asked to monitor -- and so the compiler's
    own error reaches the operator instead of a bare "binary missing".
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
    binary. A binary that exists but cannot open a netlink socket -- gVisor, a
    locked-down container, a seccomp profile without AF_NETLINK -- would
    otherwise be discovered at the start of a run, which is the worst possible
    moment.
    """
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
    """The addresses the client will actually connect to.

    Resolved here and passed to the helper as literals, so the helper never does
    DNS of its own: if the name resolves differently in the two processes, the
    monitor would watch a socket the client never opened and report an empty run
    as a quiet zero.
    """
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        return sorted({i[4][0] for i in infos})
    except Exception:
        return []


class Monitor:
    """One connection's (or one arm's) congestion samples over its lifetime.

    Unlike a per-request TCP_INFO snapshot, this runs continuously on its own
    thread from `__enter__` to `stop()`, so the reset that happens *during* an
    idle gap (while the request loop is blocked on a socket read) is not missed.
    `announce(sock)` tells the helper about a socket immediately, so the first
    (idle-window) samples are not missed either -- rather than waiting for the
    helper's own periodic rediscovery dump.

    Mirrors `core.capture.Capture` on purpose -- same lifecycle, same "record the
    failure and carry on" contract -- because a runner that drives both from the
    same window should not have a monitor that raised where a capture merely
    reported; that would turn a missing column into a failed run.

    `label` is a single opaque string. A caller that wants structured labelling
    (e.g. provider/arm/kind) assembles it before constructing the Monitor, for
    example `Monitor(f"{provider}:{arm}:{kind}", host)`.
    """

    def __init__(self, label: str, host: str, port: int = 443,
                 interval: int | None = None, rtt_hint_ms: float | None = None):
        self.label = label
        self.host = host or ""
        self.port = port

        # B12 (DESIGN.md 4.9): `rtt_hint_ms` is opt-in adaptive sampling. An
        # explicit `interval` always wins (a caller that knows exactly what it
        # wants gets exactly that); with no `interval` and no `rtt_hint_ms` the
        # behaviour is byte-for-byte what it was before this knob existed --
        # `interval_ms()` (env override or DEFAULT_INTERVAL_MS), reason "fixed",
        # confidence "high". PublicAIBackend never passes `rtt_hint_ms`, so it
        # keeps the plain fixed-2ms path unconditionally.
        if interval:
            self.interval = interval
            self.interval_reason = "fixed"
            self.measurement_confidence = "high"
        elif rtt_hint_ms is not None:
            self.interval, self.interval_reason = interval_from_rtt(rtt_hint_ms)
            self.measurement_confidence = (
                "degraded" if self.interval_reason == "floor_clamped" else "high")
        else:
            self.interval = interval_ms()
            self.interval_reason = "fixed"
            self.measurement_confidence = "high"

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
        # No --dst when the name did not resolve: watching every socket on the
        # port is noisy, and the run document says which host was meant, so a
        # reader can tell the noise from the signal. Watching nothing would just
        # be a silent zero.

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
        """Tell the helper about a socket the caller just opened.

        Runs on whichever thread made the connection, so the write is
        serialised. Only sockets going to the port being monitored are
        announced: the caller may talk to other things, and a monitor that
        followed all of them would fill the samples with traffic the run did
        not produce. This is how the very first (idle-window) samples for a
        socket are not missed, instead of waiting on the helper's own
        rediscovery dump.
        """
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
        # v4-mapped v6 (::ffff:a.b.c.d) is spelled back as v4, because that is the
        # family the socket is really in and what the helper will match against.
        local_ip = local_ip.replace("::ffff:", "")
        peer_ip = peer_ip.replace("::ffff:", "")
        line = f"track {local_ip} {local_port} {peer_ip} {peer_port}\n"
        try:
            with self._announce_lock:
                proc.stdin.write(line)
                proc.stdin.flush()
                self.announced += 1
        except Exception:
            # A dead helper is not a reason to fail a request. The samples are
            # best-effort; the run being measured is not.
            pass

    def stop(self) -> None:
        """Idempotent. SIGTERM, not kill: the helper writes its trailer on the way
        out, and the trailer is how we learn how many ticks it actually
        managed -- which is the only way to tell a monitor that watched an idle
        socket from one that was never running."""
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()      # EOF: no more announcements are coming
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
        """Read the helper's NDJSON until it closes. Runs on the reader thread.

        A malformed line is skipped rather than fatal: the helper writes
        diagnostics to stderr precisely so stdout stays parseable, but a monitor
        that killed a run over one bad line would be worse than one that lost a
        sample.
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
        """What the run document keeps for this monitor.

        `sockets` is the distinct peers seen, `resets` (via `idle_resets`) the
        number of times a socket's window dropped to a value at or below the
        initial 10 after having been larger -- the event this whole module
        exists to count, computed here rather than in a caller so the CSV and
        any summary cannot disagree about it.
        """
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
            # B12 (DESIGN.md 4.9): why the period is what it is, and how much to
            # trust samples taken at it. "fixed" is the pre-B12 default path
            # (fixed_2ms/env override); "adaptive:rtt=<x>ms" means it was derived
            # from `rtt_hint_ms` via `interval_from_rtt`; "floor_clamped" means
            # the RTT-derived period was below the netlink-tick floor and got
            # clamped -- confidence is "degraded" only in that last case.
            "interval_reason": self.interval_reason,
            "measurement_confidence": self.measurement_confidence,
            "samples": self.samples,
            "sample_count": len(self.samples),
            "ticks": self.end.get("ticks", 0),
            "seconds": self.end.get("seconds", 0),
            # How the ticks were paid for. A dump walks the kernel's whole socket
            # table (~2.4ms); an exact query is a hash lookup (~3us). If `dumps`
            # approaches `ticks`, the helper lost track of its socket and is paying
            # the walk every time -- which stretches the period and misplaces every
            # event, silently, unless the number is on the record.
            "dumps": self.end.get("dumps", 0),
            "exact_queries": self.end.get("exact_queries", 0),
            "tracked": self.end.get("tracked", 0),
            # How many sockets the caller named for us, versus how many the helper
            # had to go looking for. A run where these differ has connections
            # opening outside `announce()`, and those are the ones whose first
            # window is missed.
            "announced": self.announced,
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

    A drop only counts when the congestion-control state is `open` -- i.e. the
    kernel is not in recovery. A window that collapsed because of loss is a
    different finding with a different fix, and folding the two together would
    let a lossy network masquerade as the idle-reset problem (or hide it).

    Also returns `reset_events` with the exact `t_ms` of each reset -- the signal
    that "the *next* transmission after idle re-enters slow start".
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
