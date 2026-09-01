"""aipt.core.idle_reset -- read/write ``net.ipv4.tcp_slow_start_after_idle``
for the idle-reset (slow-start-after-idle) causal experiment.

This is the sysctl the whole `cwnd.py` module docstring is about: when it is
``1`` (the Linux default), the kernel throws away a socket's congestion
window after one idle RTO, so every LLM turn after the first re-enters slow
start. Toggling it to ``0`` (or back to ``1``) on the *responding* side of a
connection is the most direct causal test of "does idle-reset actually cost
TTFT" -- more direct than varying gateway delay/loss or turn spacing, which
only vary how *often*/how *severely* the reset fires, not whether it fires
at all.

Which side to toggle: TCP's congestion window is per-socket, per-direction
send state. A client (``web``) sending a small request and a server
(``mock-server``/``local-llm``) sending a much larger response is dominated
by the *server's* send-side cwnd, so this module is meant to be read/written
inside the responding container's own network namespace -- ``mock-server``
imports it directly (same Python process, see
``aipt.backends.mock.server``'s ``/admin/idle-reset`` endpoint); ``local-llm``
wraps the upstream ``llama-server`` binary (a separate process this project
does not own), so a small sidecar admin server
(``docker/idle_reset_admin.py``) imports this module instead, running
alongside ``llama-server`` in the same container/netns.

Same never-raises, ``(ok, reason)`` / ``{"ok": ..., ...}`` contract as
``aipt.gateway.forwarding`` and ``aipt.gateway.netem_control`` -- a missing
sysctl file, a permission error, or a non-Linux host is an expected,
reportable condition here, not a crash. Requires ``CAP_NET_ADMIN`` (the same
capability every container in this stack that touches network state already
has, see docker-compose.yml) to *write*; *reading* generally only needs the
file to exist and be world-readable, which it is under a normal container
netns.
"""

from __future__ import annotations

#: Standard Linux sysctl path for the per-namespace idle-reset flag. A
#: parameter (not a hardcoded literal) on every function below so tests can
#: point at a scratch file instead of touching the real kernel setting.
IDLE_RESET_PATH = "/proc/sys/net/ipv4/tcp_slow_start_after_idle"

_UNREADABLE = (
    "{path} could not be read -- most likely this process is not running "
    "inside a Linux container with /proc mounted (e.g. local dev/tests "
    "outside docker), or the path itself was overridden incorrectly."
)
_UNWRITABLE = (
    "{path} could not be written -- most likely this process lacks "
    "CAP_NET_ADMIN (docker-compose.yml: cap_add: [NET_ADMIN] on this "
    "service), or /proc/sys is mounted read-only in this container. ({exc})"
)


def read(path: str = IDLE_RESET_PATH) -> tuple[bool | None, str]:
    """(enabled_or_None, reason). ``enabled`` is ``True``/``False`` for a
    readable ``"1"``/``"0"``, ``None`` (with a reason) on any read failure
    or unexpected content -- never raises, never guesses."""
    try:
        with open(path, "r") as f:
            value = f.read().strip()
    except OSError as exc:
        return None, _UNREADABLE.format(path=path) + f" ({exc})"
    if value == "1":
        return True, "ready"
    if value == "0":
        return False, "ready"
    return None, f"{path} contained unexpected value {value!r}"


def write(enabled: bool, path: str = IDLE_RESET_PATH) -> tuple[bool, str]:
    """Set the sysctl to ``1`` (enabled, Linux default) or ``0`` (disabled --
    a socket in this netns keeps its cwnd across an idle gap). Returns
    ``(ok, reason)``; never raises. A write followed immediately by
    :func:`read` is how a caller confirms the value actually took (some
    sandboxes silently no-op a write to a read-only /proc/sys mount)."""
    payload = "1" if enabled else "0"
    try:
        with open(path, "w") as f:
            f.write(payload)
    except OSError as exc:
        return False, _UNWRITABLE.format(path=path, exc=exc)
    return True, "ready"


def status(path: str = IDLE_RESET_PATH) -> dict:
    """``{"ok": bool, "enabled": bool|None, "reason": str}`` -- the shape
    every ``/admin/idle-reset`` (mock-server, local-llm sidecar) and the
    web proxy route return."""
    enabled, reason = read(path)
    return {"ok": enabled is not None, "enabled": enabled, "reason": reason}


__all__ = ["IDLE_RESET_PATH", "read", "write", "status"]
