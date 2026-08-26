"""congestion: query which TCP congestion-control algorithms and qdisc are
active on this host/container, for the 4-way (cubic/reno/bbr/vegas) lab.

Read-only. This module never loads a kernel module or changes a sysctl --
it only reports what is currently true so the web UI can tell the operator
exactly what to fix (see README's `modprobe` / `tc qdisc replace` recipes).
Sysctl values are read straight from /proc so this works identically in the
Docker client container (NET_ADMIN, eth0) and a bare `python -m ...` run.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

# The four algorithms the comparison lab drives: cubic/reno are loss-based,
# vegas is delay-based, bbr is model-based (BtlBw x RTprop). See the BBR
# v1/v2/v3 write-up this project follows up on for why the qdisc matters too.
REQUIRED_ALGORITHMS = ["cubic", "reno", "bbr", "vegas"]

_AVAIL_PATH = Path("/proc/sys/net/ipv4/tcp_available_congestion_control")
_CURRENT_PATH = Path("/proc/sys/net/ipv4/tcp_congestion_control")

# BBR's designed pacing model assumes `fq`; `fq_codel`'s CoDel AQM injects
# early drops that read as loss to cubic/reno and as spurious delay signals
# to vegas, so a real 4-way comparison wants the qdisc pinned to fq.
RECOMMENDED_QDISC = "fq"


def available_algorithms(path: Path = _AVAIL_PATH) -> list[str]:
    """Algorithms the kernel currently reports usable (built-in or modprobe'd)."""
    try:
        text = path.read_text().strip()
    except OSError:
        return []
    return text.split()


def current_default_algorithm(path: Path = _CURRENT_PATH) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def qdisc_kind(iface: str = "eth0") -> tuple[str, str]:
    """Return (kind, raw_output) for the *effective* queuing discipline that
    matters for congestion-control pacing on *iface*.

    When RTT injection (tc_congestion.netem) is active, `netem` sits at
    root with RECOMMENDED_QDISC chained as its child (see netem.py's
    docstring for why plain root-netem would silently discard fq). In that
    case this reports the CHILD qdisc's kind, since that is what actually
    paces/queues packets -- "netem" alone at root does not mean fq/fq_codel
    is missing, it means "look one level down". Bare `tc qdisc show`
    already lists every qdisc on the interface across lines, so scanning
    all of them (not just the first "qdisc " match) is what makes this
    correct for the chained case.

    kind is "" (raw carries the reason) when `tc` is missing, the interface
    doesn't exist, or the command otherwise fails -- callers should treat
    that as "unknown", not "confirmed bad".
    """
    tc = shutil.which("tc")
    if tc is None:
        return "", "tc (iproute2) not installed"
    try:
        proc = subprocess.run([tc, "qdisc", "show", "dev", iface],
                              capture_output=True, text=True, timeout=5)
    except Exception as exc:
        return "", f"tc failed: {exc}"
    if proc.returncode != 0:
        return "", (proc.stderr or proc.stdout or "tc exited non-zero").strip()[:200]
    out = proc.stdout.strip()
    kinds = re.findall(r"qdisc (\S+)", out)
    if not kinds:
        return "", out[:200] or "no qdisc reported"
    if len(kinds) == 1:
        return kinds[0], out
    # Multiple qdiscs = chained (e.g. netem at root, fq as its child from
    # netem.build_commands). Prefer the recommended one if it's anywhere in
    # the chain; otherwise report the last (innermost/child) entry, since
    # that is the one actually shaping outgoing packets.
    if RECOMMENDED_QDISC in kinds:
        return RECOMMENDED_QDISC, out
    return kinds[-1], out


def status(iface: str = "eth0", required: list[str] | None = None) -> dict:
    """Everything the web UI's readiness banner needs, in one call.

    `ready` is True only when every algorithm in *required* is loaded on
    this kernel AND the root qdisc on *iface* is RECOMMENDED_QDISC.
    """
    required = required or REQUIRED_ALGORITHMS
    avail = available_algorithms()
    avail_set = set(avail)
    missing = [a for a in required if a not in avail_set]
    kind, raw = qdisc_kind(iface)
    qdisc_ok = kind == RECOMMENDED_QDISC

    guidance: list[str] = []
    if missing:
        mods = " ".join(f"tcp_{a}" for a in missing)
        guidance.append(
            "Missing congestion-control module(s): " + ", ".join(missing) +
            ". Load with: sudo modprobe " + mods)
    if not qdisc_ok:
        shown = kind or "unknown"
        guidance.append(
            f"qdisc on {iface} is '{shown}', not '{RECOMMENDED_QDISC}'. "
            f"BBR needs {RECOMMENDED_QDISC}'s precise pacing, and "
            f"fq_codel's CoDel drops would also perturb the loss-based "
            f"(cubic/reno) and delay-based (vegas) arms. Fix with: "
            f"sudo tc qdisc replace dev {iface} root {RECOMMENDED_QDISC}")

    return {
        "iface": iface,
        "required": required,
        "available": sorted(avail),
        "missing": missing,
        "current_default": current_default_algorithm(),
        "qdisc": kind,
        "qdisc_raw": raw,
        "qdisc_ok": qdisc_ok,
        "ready": not missing and qdisc_ok,
        "guidance": guidance,
        "offload": offload_status(iface),
    }


# ethtool -k feature name -> the short name tcp_congestion.offload.FEATURES
# uses. Order matches `ethtool -k`'s typical output for readability.
_OFFLOAD_FEATURE_LABELS = {
    "tso": "tcp-segmentation-offload",
    "gso": "generic-segmentation-offload",
    "sg": "scatter-gather",
    "gro": "generic-receive-offload",
    "lro": "large-receive-offload",
}


def offload_status(iface: str = "eth0") -> dict:
    """Report which of TSO/GSO/SG/GRO/LRO are currently on for *iface*.

    Informational only -- unlike the algorithm/qdisc checks, offload state
    doesn't gate `ready`: it's an experiment choice (see tcp_congestion.
    offload's docstring for why an operator might turn it off), not a
    correctness requirement for the 4-way comparison.
    """
    ethtool = shutil.which("ethtool")
    if ethtool is None:
        return {"available": False, "reason": "ethtool not installed",
                "features": {}, "all_off": False}
    try:
        proc = subprocess.run([ethtool, "-k", iface],
                              capture_output=True, text=True, timeout=5)
    except Exception as exc:
        return {"available": False, "reason": f"ethtool failed: {exc}",
                "features": {}, "all_off": False}
    if proc.returncode != 0:
        reason = (proc.stderr or proc.stdout or "ethtool exited non-zero").strip()[:200]
        return {"available": False, "reason": reason, "features": {}, "all_off": False}

    out = proc.stdout
    features: dict[str, str] = {}
    for short, label in _OFFLOAD_FEATURE_LABELS.items():
        m = re.search(rf"^{re.escape(label)}:\s*(\S+)", out, re.MULTILINE)
        features[short] = m.group(1) if m else "unknown"

    all_off = all(v.startswith("off") for v in features.values())
    return {"available": True, "reason": "", "features": features, "all_off": all_off}
