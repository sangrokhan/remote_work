"""offload: env-driven NIC offload (TSO/GSO/SG/GRO/LRO) toggling at
container start.

Environment variable:
  NIC_OFFLOAD_DISABLE  — "1"/"true"/"yes"/"on" disables every offload
                          feature this project cares about. Anything else
                          (including unset) leaves the interface's offload
                          settings untouched (the container/kernel default,
                          typically all on except LRO which is [fixed] off
                          on veth anyway).
  NIC_OFFLOAD_IFACE     — network interface (default: eth0).

Why this matters for the lab: TSO/GSO let the kernel hand the NIC (or a
software fallback, since veth has no real NIC) one big segment instead of
MTU-sized packets, so what tcpdump captures and what cwnd.Monitor's sample
timing looks like can be coarser than the wire-level packet count would
suggest. Turning all of it off gives the most literal "one packet per
cwnd-worth of data" view for the idle-reset measurements this project is
built around; LRO is already [fixed] off on veth pairs (no real hardware
to do receive-side coalescing), so it is included in the command list for
completeness/documentation but effectively a no-op here.

Usage (entrypoint), mirroring netem.py:
  from tcp_congestion.offload import from_env, apply
  cfg = from_env()
  apply(**cfg)
"""

from __future__ import annotations

import os
import subprocess

# ethtool -K feature names this project toggles, in the order shown by
# `ethtool -k`. LRO is included for documentation even though veth reports
# it `[fixed]` (ethtool -K silently no-ops on a fixed feature rather than
# erroring, so leaving it in the command list is harmless).
FEATURES = ["tso", "gso", "sg", "gro", "lro"]


def _flag(value: str) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def build_commands(iface: str, disable: bool) -> list[str]:
    """Return the ethtool command(s) to apply *disable* to every FEATURES
    entry on *iface*. Returns an empty list when disable is False (nothing
    to do -- interface is left on whatever the kernel/driver default is).
    """
    if not disable:
        return []
    state = " ".join(f"{feat} off" for feat in FEATURES)
    return [f"ethtool -K {iface} {state}"]


def _run(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True)


def apply(iface: str, disable: bool, dry_run: bool = False) -> list[str]:
    """Install the offload toggle. dry_run=True returns commands without
    executing them."""
    cmds = build_commands(iface=iface, disable=disable)
    if not dry_run:
        for cmd in cmds:
            _run(cmd)
    return cmds


def from_env() -> dict:
    """Read NIC_OFFLOAD_DISABLE and NIC_OFFLOAD_IFACE from the environment."""
    return {
        "disable": _flag(os.environ.get("NIC_OFFLOAD_DISABLE", "")),
        "iface": os.environ.get("NIC_OFFLOAD_IFACE", "eth0"),
    }
