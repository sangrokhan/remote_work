"""netem: tc netem delay applied from environment variables at container start.

Environment variables:
  NETEM_DELAY_MS  — one-way delay to inject (ms). 0 = disabled (default).
  NETEM_IFACE     — network interface (default: eth0).

Usage (entrypoint):
  from aipt.core.netem import from_env, apply
  cfg = from_env()
  apply(**cfg)

Why fq is chained as a CHILD of netem, not left as the leftover default:
`tc qdisc add dev eth0 root netem ...` REPLACES the root qdisc outright --
after that, `net.core.default_qdisc=fq` (or anything else) is moot, because
netem now owns root explicitly. Without chaining, injecting RTT with netem
silently strips away the fq pacing this project's BBR arm depends on, and
the fq_codel-style early drops it's meant to avoid never left either --
there is just netem alone. `tc-netem`'s own docs cover exactly this:
attach netem at root and put the real queuing discipline (fq here) as its
child via `parent 1:`, so both the delay injection and fq's precise pacing
apply to the same traffic.

Unchanged from `tcp_congestion/tcp_congestion/netem.py` -- token_traffic has
no equivalent, so this is a straight migration, not a merge.
"""

from __future__ import annotations

import os
import subprocess

# Chained child qdisc for BBR's pacing needs (see module docstring). Handle
# numbers are arbitrary but must not collide with netem's handle 1:.
CHILD_QDISC = "fq"
_ROOT_HANDLE = "1:"
_CHILD_HANDLE = "10:"


def parse_delay(value: str) -> int:
    """Parse a delay string to int ms. Empty/zero → 0. Negative → 0."""
    if not value.strip():
        return 0
    n = int(value)          # raises ValueError for non-numeric
    return max(0, n)


def build_commands(iface: str, delay_ms: int) -> list[str]:
    """Return the tc shell commands to install netem delay on *iface*, with
    fq chained underneath so BBR's pacing model still gets fq semantics
    even while netem injects RTT.

    Returns an empty list when delay_ms == 0 (nothing to do -- the
    interface is left on whatever qdisc it already had, e.g. the kernel's
    net.core.default_qdisc).
    Always emits a `del` before `add` so repeated calls are idempotent.
    """
    if delay_ms == 0:
        return []
    return [
        f"tc qdisc del dev {iface} root 2>/dev/null || true",
        f"tc qdisc add dev {iface} root handle {_ROOT_HANDLE} netem delay {delay_ms}ms",
        f"tc qdisc add dev {iface} parent {_ROOT_HANDLE} handle {_CHILD_HANDLE} {CHILD_QDISC}",
    ]


def _run(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True)


def apply(iface: str, delay_ms: int, dry_run: bool = False) -> list[str]:
    """Install the netem rule. dry_run=True returns commands without executing."""
    cmds = build_commands(iface=iface, delay_ms=delay_ms)
    if not dry_run:
        for cmd in cmds:
            _run(cmd)
    return cmds


def from_env() -> dict:
    """Read NETEM_DELAY_MS and NETEM_IFACE from the environment."""
    raw = os.environ.get("NETEM_DELAY_MS", "0")
    return {
        "delay_ms": parse_delay(raw),
        "iface": os.environ.get("NETEM_IFACE", "eth0"),
    }
