"""aipt.gateway.netem_control -- the Gateway container's tc netem control
loop (DESIGN.md 4.7 B9: "aipt/core/netem.py 로직을 재사용/승격").

``aipt.core.netem`` was a thin, delay-only entrypoint helper (one env var,
one shell templated command). This module is that logic *promoted* into
the Gateway's actual control surface: full ``Profile`` objects (delay +
jitter + loss + reorder), applied/queried/cleared at runtime rather than
only once at container start, with the fq-child-qdisc trick
(``aipt.core.netem``'s module docstring explains why) kept for the delay
case.

Nothing here raises for the expected failure mode -- the process not
having ``CAP_NET_ADMIN`` (true of this sandbox, true of any container that
forgot ``cap_add: [NET_ADMIN]``). That mirrors ``aipt.core.offload``'s
``_set()``/``Window`` and ``aipt.core.capture``'s ``available()``: every
public function here returns ``{"ok": bool, ...}`` (or ``(bool, reason)``
for the plain probe), the "no" branch always names the missing knob, and a
caller (the FastAPI app) can surface that honestly instead of 500ing.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from aipt.gateway.profiles import Profile, PRESETS

# Same handles as aipt.core.netem -- kept aligned in case a future change
# wants netem_control to literally call into aipt.core.netem for the
# delay-only path instead of duplicating the constants.
CHILD_QDISC = "fq"
_ROOT_HANDLE = "1:"
_CHILD_HANDLE = "10:"

DEFAULT_IFACE = os.environ.get("GATEWAY_IFACE", "eth0")

_NO_TC = (
    "tc (iproute2) not installed -- install the iproute2 package "
    "(the Gateway container image does this in docker/Dockerfile.gateway)."
)
_NO_CAP_ADMIN = (
    "tc qdisc command failed -- most likely this process lacks "
    "CAP_NET_ADMIN. Run as root, or start the container with "
    "--cap-add=NET_ADMIN (docker-compose: cap_add: [NET_ADMIN])."
)


def tc_path() -> str | None:
    return shutil.which("tc")


def available() -> tuple[bool, str]:
    """Whether netem control can actually run right now. Returns
    ``(ok, reason_if_not)`` -- same shape as ``aipt.core.capture.available``.

    This only checks the *tool* is present; whether the process actually
    has NET_ADMIN is only knowable by trying a command (checked lazily by
    :func:`_run`, since there is no cheap capability probe short of
    attempting the syscall).
    """
    if tc_path() is None:
        return False, _NO_TC
    return True, "ready"


def build_commands(iface: str, profile: Profile) -> list[list[str]]:
    """Return the ``tc`` argv lists to install *profile* on *iface*.

    Always emits a ``qdisc del`` first (idempotent re-apply, matching
    ``aipt.core.netem.build_commands``'s "always del before add"). When the
    profile has no impairment at all (the ``clean`` preset), the delete is
    still returned so calling :func:`apply_profile` with ``clean`` actively
    clears any previously-applied netem rule rather than being a no-op --
    this is how :func:`clear` is implemented (delegates to
    ``apply_profile(iface, PRESETS["clean"])``).
    """
    cmds: list[list[str]] = [["tc", "qdisc", "del", "dev", iface, "root"]]

    netem_args = _netem_args(profile)
    if not netem_args:
        return cmds

    cmds.append(["tc", "qdisc", "add", "dev", iface, "root", "handle", _ROOT_HANDLE, "netem", *netem_args])
    # Chain fq underneath, same rationale as aipt.core.netem: netem replaces
    # the root qdisc outright, so without this BBR's pacing model silently
    # loses fq semantics the moment any delay is injected.
    cmds.append(["tc", "qdisc", "add", "dev", iface, "parent", _ROOT_HANDLE, "handle", _CHILD_HANDLE, CHILD_QDISC])
    return cmds


def _netem_args(profile: Profile) -> list[str]:
    """The ``netem <args...>`` portion for one profile. Empty when the
    profile injects nothing (delay=jitter=loss=reorder=0)."""
    args: list[str] = []
    if profile.delay_ms > 0:
        args += ["delay", f"{profile.delay_ms}ms"]
        if profile.jitter_ms > 0:
            args += [f"{profile.jitter_ms}ms"]
    if profile.loss_pct > 0:
        args += ["loss", f"{profile.loss_pct}%"]
    if profile.reorder_pct > 0:
        # netem requires a delay for `reorder` to have an effect (packets
        # sent without delay have nothing to reorder against) -- if the
        # profile asks for reorder but no delay, fall back to a minimal
        # 1ms delay so the reorder percentage is not silently dropped.
        if profile.delay_ms == 0:
            args += ["delay", "1ms"]
        args += ["reorder", f"{profile.reorder_pct}%"]
    return args


def _run(argv: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        return False, _NO_TC
    except Exception as exc:
        return False, f"tc invocation failed: {exc}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        # `tc qdisc del ... root` on an interface with no netem rule yet
        # exits non-zero ("Cannot delete qdisc with handle of zero") --
        # that is not a real failure, it just means there was nothing to
        # clear, so it is swallowed rather than surfaced as an error.
        if argv[:3] == ["tc", "qdisc", "del"]:
            return True, ""
        return False, f"{' '.join(argv)} exited {proc.returncode}: {err[:200]} -- {_NO_CAP_ADMIN}"
    return True, ""


def apply_profile(iface: str, profile: Profile, *, dry_run: bool = False) -> dict:
    """Install *profile* on *iface*. Never raises.

    Returns ``{"ok": True, "profile": ..., "commands": [...]}`` on success
    (or when ``dry_run=True``, in which case commands are built but not
    executed -- same contract as ``aipt.core.netem.apply``), or
    ``{"ok": False, "reason": "...", "commands": [...]}`` when the tool is
    missing or a command failed (e.g. no CAP_NET_ADMIN).
    """
    ok, reason = available()
    commands = build_commands(iface, profile)
    rendered = [" ".join(c) for c in commands]
    if not ok:
        return {"ok": False, "reason": reason, "profile": profile.as_dict(), "commands": rendered}
    if dry_run:
        return {"ok": True, "profile": profile.as_dict(), "commands": rendered, "dry_run": True}

    for argv in commands:
        cmd_ok, cmd_reason = _run(argv)
        if not cmd_ok:
            return {"ok": False, "reason": cmd_reason, "profile": profile.as_dict(), "commands": rendered}

    _STATE[iface] = profile
    return {"ok": True, "profile": profile.as_dict(), "commands": rendered}


# In-process record of "what did we last successfully apply" per iface, so
# GET /gateway/profile can answer without re-parsing `tc qdisc show`
# output. This is intentionally best-effort/in-memory (mirrors
# aipt/web/store.py's own "in-memory only for this phase" stance) -- a
# process restart forgets it, same as the actual kernel qdisc state would
# reset on interface/container restart anyway.
_STATE: dict[str, Profile] = {}


def current_profile(iface: str) -> Profile:
    """The profile this process last successfully applied to *iface*, or
    the ``clean`` preset if nothing has been applied yet (matching the
    "no impairment until told otherwise" default)."""
    return _STATE.get(iface, PRESETS["clean"])


def clear(iface: str, *, dry_run: bool = False) -> dict:
    """Remove any netem rule from *iface* (apply the ``clean`` profile)."""
    return apply_profile(iface, PRESETS["clean"], dry_run=dry_run)


__all__ = [
    "DEFAULT_IFACE",
    "tc_path",
    "available",
    "build_commands",
    "apply_profile",
    "current_profile",
    "clear",
]
