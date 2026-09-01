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

2026-09 재설계 (client-link-only shaping, 근거: 주인님 피드백): 이전 설계는
Gateway의 두 인터페이스(client_iface/backend_iface) **양쪽 egress에 사용자가
고른 프로파일을 동일하게** 걸었다. 이건 "client↔backend 전체를 하나의
논리적 링크로 보고, 왕복 지연을 요청/응답 두 egress에 절반씩 나눠 걸어서
RTT=2×delay를 재현"하는 근사였지만, 실제 토폴로지와 맞지 않는다는 지적을
받았다: Gateway↔backend 구간은 Docker 브리지 위의 사실상 Ethernet
링크(같은 데이터센터/호스트 내부)이므로 손상이 없어야 하고, 손상은
client↔Gateway 구간(실제 인터넷 access network를 흉내내는 대상)에만 걸려야
한다.

이제 두 구간을 분리한다:
  - **client_iface**: 사용자가 고른 프로파일(clean/wired/wireless/custom)을
    **양방향** 모두에 적용한다. tc netem은 egress에만 걸리므로, 요청 방향
    (client→Gateway, client_iface의 ingress)은 IFB(Intermediate Functional
    Block) 디바이스로 리다이렉트해서 egress qdisc를 태워야 한다
    (``ensure_ifb`` + ``build_ingress_redirect_commands``). 응답 방향
    (Gateway→client)은 기존처럼 client_iface egress에 직접 건다.
  - **backend_iface**: 사용자가 고른 프로파일과 무관하게, 항상
    ``profiles.ETHERNET_BASELINE``(사실상 무손상)만 적용한다 — "Gateway↔
    backend는 Ethernet 수준"이라는 원칙을 코드로 강제한다. ingress는 건드리지
    않는다(baseline이 무손상이므로 shaping할 이유가 없다).
"""

from __future__ import annotations

import os
import shutil
import subprocess

from aipt.gateway.profiles import ETHERNET_BASELINE, Profile, PRESETS

# Same handles as aipt.core.netem -- kept aligned in case a future change
# wants netem_control to literally call into aipt.core.netem for the
# delay-only path instead of duplicating the constants.
CHILD_QDISC = "fq"
_ROOT_HANDLE = "1:"
_CHILD_HANDLE = "10:"
_INGRESS_HANDLE = "ffff:"

DEFAULT_IFACE = os.environ.get("GATEWAY_IFACE", "eth0")

# DESIGN.md 4.7 확정 설계 (2026-08-26, 2026-09 client-link-only 재설계):
# Gateway는 net-client/net-backend 두 브리지 네트워크 모두에 속한다.
# client_iface는 양방향(egress 직접 + ingress는 IFB 경유)으로 shaping하고,
# backend_iface는 고정 baseline만 적용한다 -- 자세한 근거는 모듈 독스트링.
#
# Docker는 컨테이너에 여러 네트워크를 붙일 때 어떤 인터페이스가 eth0/eth1이
# 될지 순서를 보장하지 않으므로, 하드코딩 대신 명시적인 env var로 받는다.
# docker-compose.yml에서 gateway 서비스가 net-client에 먼저 연결되면 보통
# eth0=client, eth1=backend가 되지만 그 가정에 의존하지 않기 위한 override.
DEFAULT_CLIENT_IFACE = os.environ.get("GATEWAY_CLIENT_IFACE", os.environ.get("GATEWAY_IFACE", "eth0"))
DEFAULT_BACKEND_IFACE = os.environ.get("GATEWAY_BACKEND_IFACE", "eth1")
# IFB device that client_iface's ingress (client -> Gateway request leg) is
# redirected to so its egress qdisc (netem) can shape it -- tc netem only
# ever shapes egress, this is the standard workaround for ingress shaping.
DEFAULT_IFB_DEV = os.environ.get("GATEWAY_IFB_DEV", "ifb0")

_NO_TC = (
    "tc (iproute2) not installed -- install the iproute2 package "
    "(the Gateway container image does this in docker/Dockerfile.gateway)."
)
_NO_CAP_ADMIN = (
    "tc qdisc command failed -- most likely this process lacks "
    "CAP_NET_ADMIN. Run as root, or start the container with "
    "--cap-add=NET_ADMIN (docker-compose: cap_add: [NET_ADMIN])."
)
_NO_IFB = (
    "ip link add <ifb> type ifb failed -- most likely the ifb kernel "
    "module isn't loaded on the host (modprobe ifb) or this process "
    "lacks CAP_NET_ADMIN. Ingress (client-> Gateway request leg) shaping "
    "requires IFB; egress-only shaping still applies to the response leg."
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
    """Return the ``tc`` argv lists to install *profile* on *iface*'s
    **egress**.

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


def build_ingress_redirect_commands(iface: str, ifb_dev: str) -> list[list[str]]:
    """Return the ``tc`` argv lists that redirect *iface*'s **ingress**
    traffic to *ifb_dev* so it can be shaped by an egress qdisc there (tc
    netem has no native ingress mode -- IFB is the standard workaround:
    <https://man7.org/linux/man-pages/man8/tc-mirred.8.html>).

    Idempotent re-apply: always deletes any existing ingress qdisc on
    *iface* first (mirrors :func:`build_commands`'s "del before add").
    """
    return [
        ["tc", "qdisc", "del", "dev", iface, "ingress"],
        ["tc", "qdisc", "add", "dev", iface, "handle", _INGRESS_HANDLE, "ingress"],
        [
            "tc", "filter", "add", "dev", iface, "parent", _INGRESS_HANDLE,
            "protocol", "ip", "u32", "match", "u32", "0", "0",
            "action", "mirred", "egress", "redirect", "dev", ifb_dev,
        ],
    ]


def build_ifb_setup_commands(ifb_dev: str) -> list[list[str]]:
    """Return the argv lists that ensure *ifb_dev* exists and is up.

    ``ip link add ... type ifb`` fails with a non-zero exit if the device
    already exists -- that failure is swallowed by :func:`_run` (same
    "del/add churn is fine" posture as netem qdisc re-application), not
    treated as a real error.
    """
    return [
        ["modprobe", "ifb"],
        ["ip", "link", "add", ifb_dev, "type", "ifb"],
        ["ip", "link", "set", "dev", ifb_dev, "up"],
    ]


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
        # (or `... ingress` with no ingress qdisc, or `ip link add` on an
        # already-existing device, or `modprobe` for an already-loaded
        # module) exits non-zero for "nothing to do" reasons that are not
        # real failures -- swallowed rather than surfaced as an error, same
        # "idempotent re-apply churn is fine" posture as the original
        # qdisc-del case.
        if argv[:3] in (["tc", "qdisc", "del"], ["ip", "link", "add"]) or argv[0] == "modprobe":
            return True, ""
        return False, f"{' '.join(argv)} exited {proc.returncode}: {err[:200]} -- {_NO_CAP_ADMIN}"
    return True, ""


def apply_profile(iface: str, profile: Profile, *, dry_run: bool = False) -> dict:
    """Install *profile* on *iface*'s **egress**. Never raises.

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


def apply_ingress_profile(
    iface: str, ifb_dev: str, profile: Profile, *, dry_run: bool = False
) -> dict:
    """Shape *iface*'s **ingress** traffic to *profile* by redirecting it
    to *ifb_dev* and applying netem there (:func:`apply_profile` only ever
    touches egress -- this is the ingress counterpart). Never raises.

    Steps: ensure *ifb_dev* exists and is up, redirect *iface* ingress to
    it, then install *profile* as netem on *ifb_dev*'s egress (which is
    where the redirected packets now flow out). Same
    ``{"ok": bool, "reason": ..., "commands": [...]}`` contract as
    :func:`apply_profile`; a failure at any step short-circuits and reports
    which command failed.
    """
    ok, reason = available()
    setup_cmds = build_ifb_setup_commands(ifb_dev)
    redirect_cmds = build_ingress_redirect_commands(iface, ifb_dev)
    netem_cmds = build_commands(ifb_dev, profile)
    rendered = [" ".join(c) for c in (*setup_cmds, *redirect_cmds, *netem_cmds)]
    if not ok:
        return {"ok": False, "reason": reason, "profile": profile.as_dict(), "commands": rendered}
    if dry_run:
        return {"ok": True, "profile": profile.as_dict(), "commands": rendered, "dry_run": True}

    for argv in (*setup_cmds, *redirect_cmds, *netem_cmds):
        cmd_ok, cmd_reason = _run(argv)
        if not cmd_ok:
            failure_reason = cmd_reason if argv[0] != "ip" else f"{cmd_reason} ({_NO_IFB})"
            return {"ok": False, "reason": failure_reason, "profile": profile.as_dict(), "commands": rendered}

    _STATE[f"{iface}:ingress"] = profile
    return {"ok": True, "profile": profile.as_dict(), "commands": rendered}


# In-process record of "what did we last successfully apply" per key
# (iface for egress, "<iface>:ingress" for ingress-via-IFB), so
# GET /gateway/profile can answer without re-parsing `tc qdisc show`
# output. This is intentionally best-effort/in-memory (mirrors
# aipt/web/store.py's own "in-memory only for this phase" stance) -- a
# process restart forgets it, same as the actual kernel qdisc state would
# reset on interface/container restart anyway.
_STATE: dict[str, Profile] = {}


def current_profile(iface: str) -> Profile:
    """The profile this process last successfully applied to *iface*'s
    egress, or the ``clean`` preset if nothing has been applied yet
    (matching the "no impairment until told otherwise" default)."""
    return _STATE.get(iface, PRESETS["clean"])


def current_ingress_profile(iface: str) -> Profile:
    """The profile this process last successfully applied to *iface*'s
    ingress (via IFB redirect), or ``clean`` if nothing has been applied
    yet."""
    return _STATE.get(f"{iface}:ingress", PRESETS["clean"])


def clear(iface: str, *, dry_run: bool = False) -> dict:
    """Remove any netem rule from *iface*'s egress (apply the ``clean``
    profile)."""
    return apply_profile(iface, PRESETS["clean"], dry_run=dry_run)


def apply_client_link_profile(
    client_iface: str, ifb_dev: str, profile: Profile, *, dry_run: bool = False
) -> dict:
    """Install *profile* on **both directions** of the client-facing link:
    egress (Gateway -> client, response leg, direct netem on
    *client_iface*) and ingress (client -> Gateway, request leg, via
    :func:`apply_ingress_profile`'s IFB redirect). This is the "shape the
    access-network leg the way the user actually configured it" half of
    the 2026-09 client-link-only design (module docstring).

    Both directions are attempted even if one fails, so the caller learns
    about both failures. Returns::

        {
          "ok": bool,                # True only if BOTH directions succeeded
          "profile": profile.as_dict(),
          "egress": {...apply_profile(client_iface, ...) result...},
          "ingress": {...apply_ingress_profile(client_iface, ifb_dev, ...) result...},
          "reason": "..." (present only when ok is False),
        }
    """
    egress_result = apply_profile(client_iface, profile, dry_run=dry_run)
    ingress_result = apply_ingress_profile(client_iface, ifb_dev, profile, dry_run=dry_run)
    ok = egress_result["ok"] and ingress_result["ok"]
    result = {
        "ok": ok,
        "profile": profile.as_dict(),
        "egress": egress_result,
        "ingress": ingress_result,
    }
    if not ok:
        failures = []
        if not egress_result["ok"]:
            failures.append(f"egress: {egress_result['reason']}")
        if not ingress_result["ok"]:
            failures.append(f"ingress: {ingress_result['reason']}")
        result["reason"] = "; ".join(failures)
    return result


def apply_backend_link_baseline(backend_iface: str, *, dry_run: bool = False) -> dict:
    """Install the fixed :data:`aipt.gateway.profiles.ETHERNET_BASELINE`
    on *backend_iface*'s egress -- **not** the user-selected profile.
    Gateway<->backend is modeled as an intra-datacenter Ethernet hop, so it
    always gets this baseline regardless of what the client-facing link is
    configured to (module docstring). Ingress is left untouched: the
    baseline carries no impairment, so there is nothing to shape.
    """
    return apply_profile(backend_iface, ETHERNET_BASELINE, dry_run=dry_run)


def apply_gateway_profile(
    client_iface: str,
    backend_iface: str,
    ifb_dev: str,
    profile: Profile,
    *,
    dry_run: bool = False,
) -> dict:
    """Top-level entry point: install *profile* on the client-facing link
    (both directions) and the fixed Ethernet baseline on the backend-facing
    link (module docstring, 2026-09 client-link-only design). Never raises.

    Returns::

        {
          "ok": bool,                  # True only if client link succeeded
                                        # (backend baseline failures are
                                        # reported but don't gate "ok" --
                                        # see note below)
          "profile": profile.as_dict(),           # what was requested for the client link
          "client_iface": client_iface,
          "backend_iface": backend_iface,
          "ifb_dev": ifb_dev,
          "client": {...apply_client_link_profile(...) result...},
          "backend": {...apply_backend_link_baseline(...) result, "profile" is ETHERNET_BASELINE...},
          "reason": "..." (present only when ok is False),
        }

    Note on ``ok``: the client link (the part the caller actually asked to
    change) determines top-level ``ok``. The backend baseline is a fixed,
    always-the-same operation independent of the request -- its own
    ``ok``/``reason`` is still reported under ``backend`` so a
    misconfigured backend leg (e.g. missing CAP_NET_ADMIN) is never
    silently hidden, but it does not mask whether the requested profile
    change itself succeeded.
    """
    client_result = apply_client_link_profile(client_iface, ifb_dev, profile, dry_run=dry_run)
    backend_result = apply_backend_link_baseline(backend_iface, dry_run=dry_run)

    result = {
        "ok": client_result["ok"],
        "profile": profile.as_dict(),
        "client_iface": client_iface,
        "backend_iface": backend_iface,
        "ifb_dev": ifb_dev,
        "client": client_result,
        "backend": backend_result,
    }
    if not client_result["ok"]:
        result["reason"] = f"client_iface={client_iface}: {client_result['reason']}"
    return result


def current_gateway_profile(client_iface: str, backend_iface: str, ifb_dev: str) -> dict:
    """The profile currently applied to the client-facing link (egress +
    ingress, both should match after a successful :func:`apply_gateway_profile`
    call) and the fixed baseline on the backend-facing link."""
    return {
        "client_iface": client_iface,
        "backend_iface": backend_iface,
        "ifb_dev": ifb_dev,
        "client": {
            "egress": current_profile(client_iface).as_dict(),
            "ingress": current_ingress_profile(client_iface).as_dict(),
        },
        "backend": current_profile(backend_iface).as_dict(),
    }


def clear_gateway(client_iface: str, backend_iface: str, ifb_dev: str, *, dry_run: bool = False) -> dict:
    """Remove impairment from the client link (apply ``clean`` to both its
    directions) and re-assert the backend baseline (which is already
    ``clean``-equivalent, but re-applied for consistency)."""
    return apply_gateway_profile(client_iface, backend_iface, ifb_dev, PRESETS["clean"], dry_run=dry_run)


__all__ = [
    "DEFAULT_IFACE",
    "DEFAULT_CLIENT_IFACE",
    "DEFAULT_BACKEND_IFACE",
    "DEFAULT_IFB_DEV",
    "tc_path",
    "available",
    "build_commands",
    "build_ingress_redirect_commands",
    "build_ifb_setup_commands",
    "apply_profile",
    "apply_ingress_profile",
    "apply_client_link_profile",
    "apply_backend_link_baseline",
    "apply_gateway_profile",
    "current_profile",
    "current_ingress_profile",
    "current_gateway_profile",
    "clear",
    "clear_gateway",
]

