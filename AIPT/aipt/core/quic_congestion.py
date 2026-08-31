"""aipt.core.quic_congestion -- what QUIC congestion-control algorithms
this process can actually run right now, mirrors aipt.core.congestion's
"read what's actually available, never fabricate a fallback list" posture
but for QUIC (aioquic.quic.congestion's registry) instead of the kernel's
/proc/sys/net/ipv4/tcp_available_congestion_control.

Why a separate module rather than folding into aipt.core.congestion:
QUIC congestion control lives entirely in userspace (aioquic), not the
kernel -- there is no /proc file to read, and "available" here means "is
the aioquic package importable and does it have factories registered",
not "does this kernel have a module loaded". Conflating the two would
make aipt.core.congestion's docstring (which is specifically about
kernel modules) misleading.

Importing this module also has the side effect of importing
aipt.backends.quic_mock.congestion, which registers the "idle_probe"
algorithm (see that module) -- so the dropdown this feeds always includes
it alongside aioquic's own built-in "reno"/"cubic", without the caller
having to know that registration needs to happen somewhere.
"""

from __future__ import annotations


def available() -> tuple[bool, str]:
    """(ok, reason) -- whether aioquic is importable at all. Checked
    separately from available_algorithms() so a caller (e.g. the web UI
    deciding whether to show the QUIC transport option) can distinguish
    "not installed" from "installed but somehow has zero algorithms
    registered" (the latter would indicate an aioquic API change this
    module hasn't caught up with, not a missing optional dependency)."""
    try:
        import aioquic  # noqa: F401
    except ImportError as exc:
        return False, f"aioquic not installed (optional [quic] extra): {exc}"
    return True, "ok"


def available_algorithms() -> tuple[list[str], str]:
    """(names, reason) -- QUIC congestion-control algorithm names this
    process can actually select right now.

    Empty list + a reason when aioquic isn't installed at all, or (should
    never happen in practice, but never fabricated) when it's installed
    yet has nothing registered. Never returns a hardcoded guess: the
    dropdown this feeds must only ever offer a name
    ``aioquic.quic.congestion.base.create_congestion_control()`` can
    actually construct.
    """
    ok, reason = available()
    if not ok:
        return [], reason
    try:
        from aioquic.quic.congestion import base as cc_base
        # Import side effect: registers aioquic's own "reno"/"cubic".
        import aioquic.quic.congestion.reno  # noqa: F401
        import aioquic.quic.congestion.cubic  # noqa: F401
    except ImportError as exc:
        return [], f"aioquic installed but congestion submodule import failed: {exc}"

    try:
        # Import side effect: registers this project's "idle_probe"
        # (aipt/backends/quic_mock/congestion.py). Best-effort -- a
        # failure here still leaves aioquic's own reno/cubic usable.
        from aipt.backends.quic_mock import congestion as _idle_probe_cc  # noqa: F401
    except Exception:
        pass

    names = sorted(cc_base._factories.keys())
    if not names:
        return [], "aioquic installed but no congestion algorithms registered"
    return names, "ok"


__all__ = ["available", "available_algorithms"]
