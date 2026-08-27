"""aipt.core.congestion -- what TCP congestion-control algorithms this
kernel can actually run right now, shared by every backend's connection
(Mock's raw socket in ``aipt.backends.mock.conversation``, and public_ai/
local_llm's pooled ``aipt.core.wire.session()`` connections).

Why this exists as its own module rather than a hardcoded list: a fixed
list (``cubic``, ``reno``, ``bbr``, ``vegas``, ``bic``, ``htcp``) cannot
promise the *running* kernel has all of them loaded -- a CI box or a slim
container image commonly ships only ``cubic``/``reno``. Offering a name the
kernel does not have loaded lets an operator pick it, watch the run fail
with an opaque ``algorithm_error`` after the fact, and have no way to tell
from the dropdown alone that it was never going to work. Reading
``/proc/sys/net/ipv4/tcp_available_congestion_control`` -- the same file
``sysctl net.ipv4.tcp_available_congestion_control`` reports -- answers the
question the dropdown is actually asking: "what can this box run *right
now*", not "what does Linux support in general".
"""

from __future__ import annotations

import socket

#: Linux's IPPROTO_TCP sockopt for pinning a socket's congestion-control
#: algorithm. Older CPython builds this ships against may not have the
#: constant defined even though the kernel supports the sockopt -- 13 is
#: TCP_CONGESTION's fixed value across every architecture Linux runs on.
TCP_CONGESTION = getattr(socket, "TCP_CONGESTION", 13)

#: What both `sysctl net.ipv4.tcp_available_congestion_control` and this
#: module read: the congestion-control modules the running kernel has
#: loaded, space-separated. Not `tcp_allowed_congestion_control` -- that
#: one is the (usually narrower) admin-imposed allowlist for what an
#: unprivileged process may *set*, and this dropdown is offered to
#: whichever user is running this box's web UI, not necessarily root.
_PROC_AVAILABLE = "/proc/sys/net/ipv4/tcp_available_congestion_control"


def available_algorithms() -> tuple[list[str], str]:
    """(names, reason) -- the congestion-control algorithms actually loaded
    in the running kernel, freshly read on every call (a module loaded or
    unloaded after this process started must be reflected without a
    restart, the same "read the environment on every call, never freeze it
    at import" posture as ``aipt.core.capture.pcap_dir()``).

    Empty list + a reason on any read failure (missing /proc -- not Linux,
    or a sandbox with no /proc/sys visibility -- or permission, or an empty
    file). Never fabricates a fallback list: a caller that got an empty
    list must say so to the operator, not quietly hand back a `cubic`
    guess for a kernel that might not have TCP loaded that way at all.
    """
    try:
        raw = open(_PROC_AVAILABLE).read()
    except OSError as exc:
        return [], f"could not read {_PROC_AVAILABLE}: {exc}"
    names = raw.split()
    if not names:
        return [], f"{_PROC_AVAILABLE} was empty"
    return names, "ok"


__all__ = ["TCP_CONGESTION", "available_algorithms"]
