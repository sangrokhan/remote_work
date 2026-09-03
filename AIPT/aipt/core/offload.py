"""Segmentation offload: read it, optionally turn it off around a capture, and always
say which it was.

This module merges two independent lineages:

  - `token_traffic/core/offload.py` -- the detailed, capture-time API:
    `read()`/`Window`/`describe()`/`current()`. Built for tcpdump captures
    where offload state has to be *observed*, selectively disabled for the
    duration of one capture, and *restored exactly* afterwards (not just
    switched back "on").
  - `tcp_congestion/tcp_congestion/offload.py` -- the entrypoint-time API:
    `build_commands()`/`apply()`/`from_env()`. Built for "disable offload
    once, for good, at container start" via a single `ethtool -K` call
    covering a broader feature list (tso/gso/sg/gro/lro).

Both solve the same underlying problem (offload distorts what a packet
capture -- or a cwnd sampler timed to packet arrivals -- actually sees) but
at different times in a run's lifecycle, so both APIs are kept, and both env
var names are honoured. As of T7 (2026-09), the two APIs also share one
feature-set (`FEATURES`, both names) and one restore contract: the
capture-time `Window` restores automatically at the end of its `with`
block, and the entrypoint-time `Toggle` offers the same
observe-before/restore-only-what-changed contract for callers that want a
container's NIC put back the way it was found (`build_commands()`/`apply()`
keep their original "disable once, no restore" signatures unchanged, for
backward compatibility with existing entrypoint scripts):

  NIC_OFFLOAD_DISABLE     — canonical name (from tcp_congestion). "1"/"true"/
                            "yes"/"on" means disable.
  TRAFFIC_PCAP_NO_OFFLOAD — deprecated alias (from token_traffic), kept for
                            existing docker-compose.yml/docs compatibility.
                            Either name being truthy is enough.
  NIC_OFFLOAD_IFACE       — interface for the entrypoint-time bulk toggle
                            (default: eth0).

The problem this solves. tcpdump taps AF_PACKET, and on egress that tap sits *before*
the NIC segments anything. With TSO or GSO on, what lands in the pcap is the super-
packet the kernel handed the card -- up to 64 KB -- not the ~1448-byte frames that
actually crossed the wire. GRO does the mirror image on receive: separate frames are
already merged by the time the tap sees them. So a pcap from an offloaded interface
shows packets that never existed.

Byte totals survive this. Offload splits a payload, it does not change it. Two things
do not survive:

  Header overhead is undercounted. One super-packet records one 40-byte header; the
  45 real frames it becomes each pay their own.

  Segment counts are meaningless, and segment counts are how congestion control is
  read. Slow start's signature -- bursts of 10, then 20, then 40 -- is invisible when
  40 frames arrive as one pcap record. This is exactly the evidence a cwnd argument
  needs, and offload erases it.

So why is turning it off not the default? Because it changes the thing being measured.
Segmenting in software costs CPU per packet and alters pacing, and this package times
LLM turns to the millisecond. A latency arm captured with offload disabled is not the
same experiment as one captured with it on. Truth about the packets is bought with a
confound in the timings, and which of those matters depends on the question -- so it is
a knob, off by default, and the state is recorded either way.

Recorded either way is the part that is not optional. A pcap whose segmentation state
is unknown cannot be read as evidence of anything: the reader cannot tell a 64 KB
record from a jumbo frame, or a missing burst from a burst that never happened.

Where it applies. Under Docker the capture runs in the container's own network
namespace, so the interface that matters is the container's `eth0` -- turning offload
off on the host's NIC would change nothing a containerized tcpdump can see. The
container needs CAP_NET_ADMIN for this, which docker-compose.yml already grants for
tcpdump's sake. On the host, it needs root.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

from aipt.core import config

# The full feature set both APIs now agree on (T7 -- previously the
# capture-time API only touched tso/gso/gro while the entrypoint-time API
# touched tso/gso/sg/gro/lro; DESIGN.md's "practically the same, only the env
# var name differs" claim was not true of the feature-set until this was
# unified). `lro` is almost always reported `[fixed]` by veth/most drivers,
# and `_set()`/`Window`/`Toggle` all already skip fixed features rather than
# erroring, so including it here is harmless even where it cannot actually
# be toggled -- and it means a driver that *does* allow disabling it gets
# the same literal one-packet-per-segment posture as everything else. `sg`
# (scatter-gather) is included for the same "most literal view" reason it
# was originally added to the entrypoint-time set.
FEATURES = ("tso", "gso", "sg", "gro", "lro")

# Deprecated alias for FEATURES, kept because external callers/tests may
# still reference the entrypoint-time name. Equal to FEATURES since T7 --
# do not let this drift from FEATURES again.
ENTRYPOINT_FEATURES = list(FEATURES)

# What ethtool calls them when reporting, as opposed to when setting.
_REPORTED = {
    "tso": "tcp-segmentation-offload",
    "gso": "generic-segmentation-offload",
    "sg": "scatter-gather",
    "gro": "generic-receive-offload",
    "lro": "large-receive-offload",
}

_LINE = re.compile(r"^([a-z0-9-]+):\s*(on|off)\b(.*)$")


def enabled() -> bool:
    """Whether a capture should turn offload off for its window.

    Checks the canonical `NIC_OFFLOAD_DISABLE` first, then falls back to the
    deprecated `TRAFFIC_PCAP_NO_OFFLOAD` alias -- either being truthy is
    enough.
    """
    return config.flag_any("NIC_OFFLOAD_DISABLE", "TRAFFIC_PCAP_NO_OFFLOAD")


def ethtool_path() -> str | None:
    return shutil.which("ethtool")


def egress_iface(target: str) -> str:
    """The interface packets to `target` actually leave by.

    `TRAFFIC_PCAP_IFACE` defaults to `any`, which is a pseudo-interface tcpdump
    understands and ethtool does not. Asking the routing table is the only way to turn
    "capture everything" into a device with offload settings on it -- and it gives the
    right answer inside a container (eth0) and on a host (the physical NIC) without
    either having to be special-cased.
    """
    try:
        out = subprocess.run(["ip", "route", "get", target],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return ""
    m = re.search(r"\bdev\s+(\S+)", out)
    return m.group(1) if m else ""


def read(iface: str) -> dict:
    """The current on/off state of each feature, plus whether it can be changed.

    A feature reported `[fixed]` cannot be turned off at all, and saying so is the
    difference between "we left it on" and "we could not turn it off".
    """
    tool = ethtool_path()
    if not tool or not iface:
        return {}
    try:
        out = subprocess.run([tool, "-k", iface],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return {}

    reported = {}
    for line in out.splitlines():
        m = _LINE.match(line.strip())
        if m:
            reported[m.group(1)] = (m.group(2) == "on", "[fixed]" in m.group(3))

    state = {}
    for short, long in _REPORTED.items():
        if long in reported:
            on, fixed = reported[long]
            state[short] = {"on": on, "fixed": fixed}
    return state


def _set(iface: str, values: dict) -> str:
    """Apply `{feature: on?}`. Returns "" on success, else the reason.

    One ethtool call for all of them: the driver reinitialises the ring on each change,
    and doing that three times is three chances to bounce a link that carries the
    traffic being measured.
    """
    tool = ethtool_path()
    if not tool:
        return "ethtool not installed"
    if not iface:
        return "no interface"
    args = []
    for feature, on in values.items():
        args += [feature, "on" if on else "off"]
    if not args:
        return ""
    try:
        proc = subprocess.run([tool, "-K", iface, *args],
                              capture_output=True, text=True, timeout=15)
    except Exception as exc:
        return f"ethtool failed: {exc}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return f"ethtool -K {iface} exited {proc.returncode}: {err[:200]}"
    return ""


class Window:
    """Turn offload off for the duration of a capture, then put it back exactly.

    "Exactly" means restoring each feature to the value it had, not to `on`. A box
    where GRO was already off for its own reasons must not come out of a run with GRO
    enabled -- a measurement tool that leaves the machine different from how it found
    it is a bug, and a silent one.

    Nothing here raises. Missing ethtool, no CAP_NET_ADMIN, a driver that refuses --
    all of them mean the capture proceeds with offload on, saying so, because a pcap
    that is harder to read still beats a run that died configuring the NIC.
    """

    def __init__(self, target: str, iface: str = ""):
        self.iface = iface or egress_iface(target)
        self.before: dict = {}
        self.changed: list[str] = []
        self.error = ""
        self.applied = False

    def __enter__(self) -> "Window":
        self.before = read(self.iface)
        if not enabled():
            return self
        if not self.before:
            # Three different missing pieces, and naming the wrong one sends the
            # operator to install the wrong package. The `ip` case is the easy one to
            # get wrong: ethtool is present and working, but nothing has told it which
            # device to work on, because `any` is not one.
            if not ethtool_path():
                self.error = "ethtool not installed"
            elif not self.iface:
                self.error = ("cannot determine the egress interface "
                              "(is iproute2 installed?)")
            else:
                self.error = f"ethtool -k {self.iface} said nothing usable"
            return self

        want = {f: False for f, s in self.before.items()
                if s["on"] and not s["fixed"]}
        if not want:
            return self                      # already off, or fixed on: nothing to do

        err = _set(self.iface, want)
        if err:
            self.error = err
            return self
        self.changed = sorted(want)
        self.applied = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.restore()

    def restore(self) -> None:
        """Idempotent. Runs even when the capture failed -- especially then."""
        if not self.applied:
            return
        self.applied = False
        err = _set(self.iface, {f: self.before[f]["on"] for f in self.changed})
        if err:
            # Loud, because the machine is now not how we found it and nothing else
            # in the run will notice.
            self.error = (f"{self.error + '; ' if self.error else ''}"
                          f"RESTORE FAILED for {', '.join(self.changed)} on "
                          f"{self.iface}: {err}")

    def result(self) -> dict:
        """What the pcap needs to be self-describing."""
        during = {f: (False if f in self.changed else s["on"])
                  for f, s in self.before.items()}
        return {
            "iface": self.iface,
            "requested": enabled(),
            "disabled": list(self.changed),
            "during_capture": during,
            "before": {f: s["on"] for f, s in self.before.items()},
            "fixed": sorted(f for f, s in self.before.items() if s["fixed"]),
            "error": self.error,
        }


def current(target: str = "1.1.1.1") -> dict:
    """The state right now, in the shape `describe()` reads.

    A preflight wants the same sentence a finished capture reports, but it has no
    Window to ask -- and `read()` alone returns a different shape, which a caller that
    passed it straight to `describe()` would see silently reported as "unknown". Hence
    one function that produces the shape, rather than two callers agreeing to build it.
    """
    iface = egress_iface(target)
    state = read(iface)
    return {
        "iface": iface,
        "during_capture": {f: s["on"] for f, s in state.items()},
        "fixed": sorted(f for f, s in state.items() if s["fixed"]),
    }


def describe(state: dict) -> str:
    """One line for the UI and the log, saying what a reader of this pcap must know."""
    if not state or not state.get("during_capture"):
        return "offload state unknown (ethtool unavailable)"
    on = sorted(f for f, v in state["during_capture"].items() if v)
    if not on:
        return f"offload off on {state['iface']}: packet sizes are real wire frames"
    return (f"offload ON ({', '.join(on)}) on {state['iface'] or '?'}: pcap packet "
            f"sizes are kernel super-packets, not wire frames; segment counts are not "
            f"meaningful")


# --- entrypoint-time bulk toggle (inherited from tcp_congestion) ----------
#
# Before T7, this half disabled every ENTRYPOINT_FEATURES entry once, at
# container start, and left it that way for the container's life -- no
# restore, and a different (5-item) feature set than `Window`'s capture-time
# 3. Both differences were bugs relative to DESIGN.md's claim that the two
# APIs are "practically the same, only the env var name differs": that
# claim is what this module is supposed to make true.
#
# Since T7:
#   - feature-set: unified. `build_commands()` toggles `FEATURES` (the same
#     5 `Window` reads/restores), not a separate ENTRYPOINT_FEATURES list.
#     `ENTRYPOINT_FEATURES` is kept as a `== FEATURES` alias so any code or
#     test that still names it does not need to change.
#   - restore policy: unified. `Toggle` below is the entrypoint-time mirror
#     of `Window` -- it reads before/changed state the same way and offers
#     `.restore()` for callers (or an entrypoint's shutdown/SIGTERM path)
#     that want the container's NIC put back the way `Window` would. The
#     free functions (`build_commands()`/`apply()`/`from_env()`) keep their
#     original signatures and "disable once, no restore" behavior so no
#     existing caller/entrypoint script breaks; `Toggle` is additive, not a
#     breaking replacement.


def build_commands(iface: str, disable: bool) -> list[str]:
    """Return the ethtool command(s) to apply *disable* to every FEATURES
    entry on *iface*. Returns an empty list when disable is False (nothing
    to do -- interface is left on whatever the kernel/driver default is).

    This is the blunt, state-blind form: it does not read current state
    first, so it always emits the same "everything off" command regardless
    of what is already off. `Toggle` (below) is the state-aware, restorable
    alternative -- prefer it when a restore path is wanted.
    """
    if not disable:
        return []
    state = " ".join(f"{feat} off" for feat in FEATURES)
    return [f"ethtool -K {iface} {state}"]


def _run(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True)


def apply(iface: str, disable: bool, dry_run: bool = False) -> list[str]:
    """Install the offload toggle. dry_run=True returns commands without
    executing them.

    Backward-compatible: this remains the blunt "disable once, do not
    restore" entrypoint behavior it always was (now over the unified
    FEATURES set). For a restorable entrypoint-time toggle, use `Toggle`.
    """
    cmds = build_commands(iface=iface, disable=disable)
    if not dry_run:
        for cmd in cmds:
            _run(cmd)
    return cmds


def from_env() -> dict:
    """Read NIC_OFFLOAD_DISABLE (or the deprecated TRAFFIC_PCAP_NO_OFFLOAD
    alias) and NIC_OFFLOAD_IFACE from the environment."""
    return {
        "disable": enabled(),
        "iface": os.environ.get("NIC_OFFLOAD_IFACE", "eth0"),
    }


class Toggle:
    """Entrypoint-time bulk offload toggle, with the same restore contract
    as `Window` -- state read before changing anything, only what was
    actually flipped gets flipped back, and a `[fixed]` feature is left
    alone rather than making the whole `ethtool -K` call fail.

    The difference from `Window` is scope and lifetime, not policy: `Window`
    is a context manager scoped to one capture and is always used with
    `with`; `Toggle` is meant to live for a container's lifetime (`apply()`
    at start, `restore()` at shutdown/SIGTERM if the operator wants the NIC
    left as it was found) and is driven explicitly rather than through
    `with`, since an entrypoint script's "duration" is the whole container,
    not a `with` block's scope.

    Nothing here raises, for the same reason `Window` does not: a container
    that dies configuring its NIC is worse than one that starts with offload
    on and says so.
    """

    def __init__(self, iface: str):
        self.iface = iface
        self.before: dict = {}
        self.changed: list[str] = []
        self.error = ""
        self.applied = False

    def apply(self, disable: bool) -> list[str]:
        """Disable every non-fixed, currently-on FEATURES entry on `iface`.

        Returns the `ethtool -K` command(s) actually issued (empty if
        `disable` is False, everything was already off, or every on feature
        is `[fixed]`). Mirrors `Window.__enter__`'s want/skip-fixed logic.
        """
        self.before = read(self.iface)
        if not disable:
            return []
        if not self.before:
            # No usable `ethtool -k` output (missing ethtool, bad iface, ...).
            # Fall back to the blunt one-shot command so the historical
            # "just turn it off" entrypoint behavior still happens; there is
            # nothing to restore later because there is no `before` to
            # restore from.
            self.error = f"ethtool -k {self.iface} said nothing usable; cannot track state to restore"
            return build_commands(self.iface, disable=True)

        want = {f: False for f, s in self.before.items()
                if s["on"] and not s["fixed"]}
        if not want:
            return []

        err = _set(self.iface, want)
        if err:
            self.error = err
            return []
        self.changed = sorted(want)
        self.applied = True
        return [f"ethtool -K {self.iface} " + " ".join(f"{f} off" for f in self.changed)]

    def restore(self) -> None:
        """Idempotent. Put back exactly what `apply()` changed."""
        if not self.applied:
            return
        self.applied = False
        err = _set(self.iface, {f: self.before[f]["on"] for f in self.changed})
        if err:
            self.error = (f"{self.error + '; ' if self.error else ''}"
                          f"RESTORE FAILED for {', '.join(self.changed)} on "
                          f"{self.iface}: {err}")

    def result(self) -> dict:
        """Same shape as `Window.result()`, so a container's shutdown log
        can describe its NIC state the same way a pcap does."""
        during = {f: (False if f in self.changed else s["on"])
                  for f, s in self.before.items()}
        return {
            "iface": self.iface,
            "disabled": list(self.changed),
            "during": during,
            "before": {f: s["on"] for f, s in self.before.items()},
            "fixed": sorted(f for f, s in self.before.items() if s["fixed"]),
            "error": self.error,
        }
