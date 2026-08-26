"""aipt.gateway.forwarding -- kernel IP forwarding availability check
(DESIGN.md 4.7 "미해결 세부사항" 1, 확정 L3 라우팅 설계).

The Gateway container is supposed to be a pure L3 IP-forwarding hop:
``net.ipv4.ip_forward=1`` (set via docker-compose.yml's
``sysctls: [net.ipv4.ip_forward=1]``) plus kernel routing between the two
bridge networks (``net-client``, ``net-backend``) it straddles -- no
application-level proxy code, no TCP payload/header inspection.

That sysctl can silently fail to take effect for reasons that have nothing
to do with this codebase (host kernel doesn't support per-netns sysctl
namespacing, the container runtime dropped the setting, someone forgot the
``sysctls:`` block after editing docker-compose.yml, ...). Rather than
assume it worked, this module actually reads
``/proc/sys/net/ipv4/ip_forward`` at runtime so ``GET /health`` can report
the truth instead of the compose file's stated intent.

Same never-raises, ``(ok, reason)``/``{"ok": bool, "reason": ...}`` contract
as ``aipt.gateway.netem_control`` -- a missing/wrong value here is an
expected, reportable condition (e.g. running this app outside its intended
container for local dev/tests), not a crash.
"""

from __future__ import annotations

#: Standard Linux sysctl path for the IPv4 forwarding flag. A parameter on
#: the functions below (rather than a hardcoded literal) so tests can point
#: at a scratch file instead of mocking `open` globally.
IP_FORWARD_PATH = "/proc/sys/net/ipv4/ip_forward"

_NOT_ENABLED = (
    "net.ipv4.ip_forward is not 1 -- the gateway container's sysctl "
    "(docker-compose.yml: sysctls: [net.ipv4.ip_forward=1] on the `gateway` "
    "service) did not take effect, or this process lacks the privilege to "
    "see/set it. Without this, the kernel will not route packets between "
    "net-client and net-backend even though tc netem profiles apply fine."
)
_UNREADABLE = (
    "{path} could not be read -- most likely this process is not running "
    "inside a Linux container with /proc mounted (e.g. local dev/tests "
    "outside docker), or the path itself was overridden incorrectly."
)


def read_ip_forward(path: str = IP_FORWARD_PATH) -> tuple[bool, str]:
    """Read the raw sysctl value at *path*. Never raises.

    Returns ``(True, "ready")`` only when the file exists, is readable, and
    its contents are exactly ``"1"``. Any other outcome (missing file,
    permission error, ``"0"``) returns ``(False, reason)``.
    """
    try:
        with open(path, "r") as f:
            value = f.read().strip()
    except OSError as exc:
        return False, f"{_UNREADABLE.format(path=path)} ({exc})"
    if value != "1":
        return False, _NOT_ENABLED
    return True, "ready"


def available(path: str = IP_FORWARD_PATH) -> tuple[bool, str]:
    """Whether this Gateway container is actually forwarding IP packets at
    the kernel level right now. Same ``(ok, reason)`` shape as
    ``aipt.gateway.netem_control.available()`` -- intended to be surfaced
    directly in ``GET /health`` alongside the netem availability check."""
    return read_ip_forward(path)


def status(path: str = IP_FORWARD_PATH) -> dict:
    """``{"ok": bool, "reason": str}`` variant of :func:`available`, for
    callers (like ``aipt.gateway.app``) that want the same dict shape
    ``netem_control.apply_profile`` uses rather than a bare tuple."""
    ok, reason = available(path)
    return {"ok": ok, "reason": reason}


__all__ = [
    "IP_FORWARD_PATH",
    "read_ip_forward",
    "available",
    "status",
]
