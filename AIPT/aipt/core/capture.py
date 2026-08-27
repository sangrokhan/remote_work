"""tcpdump around a run's traffic: the packets themselves, as evidence the
numbers are real.

A byte count taken inside the process is a claim. A pcap taken outside it, on
the interface, is the thing the claim is about -- the TLS payload is
encrypted, but packet sizes and timing are exactly the traffic being argued
over, and they can be opened in Wireshark by somebody who does not trust this
code.

Two callers, two granularities of "one capture":

* external_api (real provider calls, `token_traffic/core/capture.py`): one
  pcap per (provider, arm, kind), because the arms are what the run compares
  and a `both` run's blocking/streamed passes interleave on the same host and
  port, so one capture spanning either cannot be attributed after the fact.
* synthetic_mock (`tcp_congestion/tcp_congestion/capture.py`): one pcap per
  run -- a single keep-alive connection per experiment, not multiple
  provider/arm combinations.

`label` is the parameter this module was generalized around: pass it
directly for a synthetic_mock-style simple label (defaults to
"conversation"), or leave it unset and pass `provider`/`arm`/`kind` and the
external_api-style compound label (`{provider}_{arm}_{kind}`) is built for
you. Either way it ends up in the pcap filename and in `result()["label"]`.

Needs the tcpdump binary and raw-socket capability (NET_RAW). Works locally
and in Docker with `--cap-add=NET_RAW`; does not work under gVisor (Cloud
Run), where capture reports itself unavailable and the run proceeds without
it.

AppArmor pitfall (do not remove `apparmor_blocks()` -- see its docstring):
Ubuntu's tcpdump AppArmor profile carries `audit deny @{HOME}/.*/** mrwkl`,
so a checkout living under a dot-directory in $HOME (e.g. ~/.openclaw/...)
cannot host the pcap directory: tcpdump gets NET_RAW, opens the socket, and
then dies on the *output file* with a bare "Permission denied" that reads
exactly like the capability never took. Real hours were lost to that
misreading before this function existed; `available()` and
`_default_pcap_dir()` both consult it so the failure explains itself instead
of repeating.

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

from aipt.core import offload

# Default TLS port used when no explicit `port` is given (external_api mode:
# real providers, always 443). synthetic_mock passes its own `port` (e.g.
# the mock server's 8888) explicitly.
_DEFAULT_PORT = 443


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
        return Path(tempfile.gettempdir()) / "aipt_pcaps"
    return local


def pcap_dir() -> Path:
    """Where captures are written, read from the environment on every call.

    A module constant frozen at import is a knob that silently does nothing whenever
    the import happens first -- which, in a web app and in a test suite, it always
    does. `core.store` re-reads its directory per call for the same reason.
    """
    env = os.environ.get("TRAFFIC_PCAP_DIR")
    return Path(env) if env else _default_pcap_dir()


# Snaplen: bytes kept per packet. The TLS payload is encrypted and therefore useless
# to read, so only the L2-L4 headers and the TLS record header are worth storing.
# Truncating to ~200 bytes (Ethernet+IP+TCP headers with options, ~54-66 bytes,
# plus room for an HTTP/TLS record header on top) slashes the disk I/O per packet,
# which is the main cause of kernel drops (the "previous segment not captured"
# warnings) under load, while still leaving the MTU/MSS-relevant evidence intact:
# with segmentation offload disabled (see aipt.core.offload.Window, applied around
# every Capture window) each wire-real frame tops out at the path MTU (typically
# 1500 bytes / ~1448-1460 byte MSS payload), and 200 bytes is enough to see the
# frame boundary and header stack without paying to store the (often encrypted,
# always disk-costly) bulk of the payload. Each frame still records its original
# on-wire length, so packet *sizes* (and therefore MTU/MSS-boundary evidence) stay
# exact regardless of snaplen -- only the *stored bytes* are truncated.
PCAP_SNAPLEN = int(os.environ.get("TRAFFIC_PCAP_SNAPLEN", "200"))

# Filename = label + timestamp + a high-entropy token, so concurrent runs never
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


# The label identifies the run/arm in the filename. Anything outside this alphabet
# could escape the pcap directory once it lands in a filename. Underscores belong
# here: external_api arm names carry them (interaction_inline, responses_inline),
# and an earlier version that left them out quietly relabelled such arms and shipped
# a pcap claiming to be a different one.
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


def _resolve_ips(host: str, port: int = _DEFAULT_PORT) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        return sorted({i[4][0] for i in infos})
    except Exception:
        return []


def _filter_expr(ips: list[str], port: int = _DEFAULT_PORT) -> str:
    """tcpdump filter: tcp/port to the target host's IP(s), or all of that port if
    the name did not resolve -- a noisier pcap beats no pcap."""
    if not ips:
        return f"tcp port {port}"
    hosts = " or ".join(f"host {ip}" for ip in ips)
    return f"tcp port {port} and ({hosts})"


def ethtool_path() -> str | None:
    return shutil.which("ethtool")


# `ethtool -T <iface>` prints a "Capabilities:" block with one flag per line,
# indented, e.g.:
#     Capabilities:
#             hardware-transmit
#             hardware-receive
#             hardware-raw-clock
#             software-transmit
#             software-receive
#             software-system-clock
# Any `hardware-*` line under Capabilities means the NIC itself timestamps
# packets (nanosecond resolution, no kernel/GIL jitter); only `software-*`
# lines means the kernel timestamps them when it gets around to processing
# them (microsecond-ish resolution, jittery under load) -- see DESIGN.md
# 4.9's B13: this is the reliability floor for packets.csv's gap_ms on short
# RTT paths (Mock/LocalLLM backends, sub-ms gaps), where that jitter is on
# the same order as the signal being measured.
_HW_TS_RE = re.compile(r"^hardware-(transmit|receive)\b")


def timestamp_source(iface: str = "eth0") -> dict:
    """Whether `iface` timestamps packets in hardware or software.

    DESIGN.md 4.9's B13: `packets.csv`'s `gap_ms` (inter-arrival gap) is only
    as trustworthy as the clock that stamped each packet. A software
    timestamp is taken when the kernel gets around to the packet -- fine
    against a multi-ms RTT, but on the sub-ms paths the v2 Mock/LocalLLM
    backends produce, that jitter can be the same size as the gap being
    measured. A hardware timestamp comes from the NIC itself and does not
    have that problem. This reports which one `iface` has, following the
    same "detect availability, explain why not, never raise" pattern as
    `available()`/`offload.describe()` above -- a caller trying to judge
    packets.csv's precision needs an answer, not a stack trace, on a laptop
    or a container with no `ethtool` at all.
    """
    tool = ethtool_path()
    if tool is None:
        return {"iface": iface, "hardware_timestamping": False, "raw": "",
                 "available": False, "reason": "ethtool not installed"}
    try:
        proc = subprocess.run([tool, "-T", iface], capture_output=True,
                               text=True, timeout=10)
    except Exception as exc:
        return {"iface": iface, "hardware_timestamping": False, "raw": "",
                 "available": False, "reason": f"ethtool failed to run: {exc}"}
    raw = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return {"iface": iface, "hardware_timestamping": False, "raw": raw,
                 "available": False,
                 "reason": f"ethtool -T {iface} exited {proc.returncode}: "
                           f"{raw.strip()[:200]}"}
    hardware = any(_HW_TS_RE.match(line.strip()) for line in raw.splitlines())
    return {"iface": iface, "hardware_timestamping": hardware, "raw": raw,
             "available": True,
             "reason": ("hardware timestamping available" if hardware else
                        "software timestamping only (no hardware-transmit/"
                        "hardware-receive capability reported)")}


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
    """Context manager around one capture's traffic: start tcpdump on enter, stop
    and read its stats on exit. One instance, one pcap.

    Two ways to name it, matching the two callers this was merged from:

    * synthetic_mock style -- pass `label` directly (defaults to
      "conversation" if omitted along with provider/arm). One capture per
      run.
    * external_api style -- pass `provider`/`arm`/(`kind`) and leave `label`
      unset; the label is built as `{provider}_{arm}_{kind}` (or
      `{provider}_{arm}` with no kind). `kind` is the measure the pcap
      covers -- "bytes" or "latency" -- and is in the label, and so in the
      filename, because a `both` run captures the blocking and the streamed
      passes into *separate* pcaps: the two passes interleave on the same
      host and port, so one capture holding both cannot be read against
      either the bytes number or the latency number.

    `port` defaults to 443 (external_api: real TLS providers). synthetic_mock
    passes its mock server's port explicitly (e.g. 8888).
    """

    def __init__(self, timestamp: str, provider: str = "", arm: str = "",
                 host: str = "", kind: str = "", *, label: str | None = None,
                 port: int | None = None, interface: str | None = None):
        self.timestamp = timestamp
        self.provider = provider or ""
        self.arm = arm or ""
        self.kind = kind or ""
        self.host = host or ""
        self.port = _DEFAULT_PORT if port is None else port
        if label is not None:
            self.label = label
        elif self.kind:
            self.label = f"{self.provider}_{self.arm}_{self.kind}"
        elif self.provider or self.arm:
            self.label = f"{self.provider}_{self.arm}"
        else:
            self.label = "conversation"
        self.interface = interface or _iface()
        self.ips: list[str] = []
        self.snaplen = PCAP_SNAPLEN
        self.proc: subprocess.Popen | None = None
        # Exists from construction, so restore() and result() never have to check for
        # None -- including on the rejected-label path, which returns before __enter__
        # does anything.
        self.offload = offload.Window("", iface="")
        self.error = ""
        self.stats: dict = {}
        # Queried once here, not on every result() call: it shells out to
        # ethtool, and the interface's timestamp source cannot change over
        # the life of one capture, so there is nothing to gain by re-asking.
        self.timestamp_source = timestamp_source(self.interface)
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
        # silently overwriting another run's capture.
        try:
            os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        except FileExistsError:
            self.error = "pcap_name_collision"
            return self
        except Exception as exc:
            self.error = f"pcap_open_failed: {exc}"
            return self
        self.ips = _resolve_ips(self.host, self.port)
        # Before tcpdump, not after. A capture that started under offload and then had
        # it turned off mid-window would hold two kinds of packet in one file with
        # nothing in the file to say where the boundary is.
        self.offload = offload.Window(self.ips[0] if self.ips else self.host)
        self.offload.__enter__()
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
            self.offload.restore()
            return self
        time.sleep(0.4)  # let tcpdump initialize before any traffic flows
        if self.proc.poll() is not None:
            err = (self.proc.stderr.read() or b"").decode(errors="replace").strip()
            self.error = f"tcpdump_exited: {err[:200]}"
            self.proc = None
            self.offload.restore()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.proc is None:
            self.offload.restore()
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
            # After tcpdump has stopped, so the last packet in the file was still taken
            # under the settings this capture's result reports.
            self.offload.restore()

    def result(self) -> dict:
        """What the capture actually got, for the run document and the UI."""
        if self.error:
            return {"ok": False, "label": self.label, "provider": self.provider,
                    "arm": self.arm, "kind": self.kind, "error": self.error,
                    "host": self.host, "port": self.port,
                    "offload": self.offload.result(),
                    "timestamp_source": self.timestamp_source}
        size = self.path.stat().st_size if self.path.exists() else 0
        dropped = (self.stats.get("dropped_by_kernel", 0)
                   + self.stats.get("dropped_by_interface", 0))
        out = {
            "ok": size > 0,
            "label": self.label,
            "provider": self.provider,
            "arm": self.arm,
            "file": self.path.name,
            "bytes": size,
            "host": self.host,
            "port": self.port,
            "ips": self.ips,
            "filter": _filter_expr(self.ips, self.port),
            "snaplen": self.snaplen,
            # Without this a reader cannot tell a 64 KB kernel super-packet from a
            # jumbo frame, or a missing slow-start burst from one that never happened.
            "offload": self.offload.result(),
            "stats": self.stats,
            "dropped": dropped,
            "log": self._log_lines(dropped),
            "error": "",
            "note": "" if size > 0 else "no packets captured (a mock run makes no real traffic)",
            # B13: whether gap_ms in packets.csv can be trusted on a short-RTT
            # path -- see timestamp_source()'s docstring.
            "timestamp_source": self.timestamp_source,
        }
        if self.kind:
            out["kind"] = self.kind
        return out

    def _log_lines(self, dropped: int) -> list[str]:
        state = self.offload.result()
        # First line, before the counts. Somebody reading "3412 captured" needs to know
        # whether those were packets or kernel super-packets before the number means
        # anything.
        lines = [f"offload[{self.label}]: {offload.describe(state)}"]
        if state.get("error"):
            lines.append(f"⚠ offload: {state['error']}")
        if not self.stats:
            return lines
        s = self.stats
        lines += [
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
