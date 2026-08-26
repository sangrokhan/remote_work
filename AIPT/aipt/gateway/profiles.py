"""aipt.gateway.profiles -- named netem parameter presets (DESIGN.md 4.7,
"구성" table, 웹 UI dropdown values: clean/broadband/3g/satellite/lossy/
custom).

Each :class:`Profile` is the full set of ``tc netem`` knobs this package
controls: ``delay_ms``/``jitter_ms`` (variable one-way delay),
``loss_pct`` (packet loss %), ``reorder_pct`` (packet reordering %). Values
are intentionally simple (no distribution/correlation parameters) --
DESIGN.md's own "미해결 세부사항" leaves those for later phases; this is
the first, concrete cut.

``custom`` is not itself a fixed preset -- it is the escape hatch that lets
a caller (env vars, the POST /gateway/profile body) supply arbitrary
values, still expressed as the same :class:`Profile` shape so
``netem_control`` never has to special-case it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: The names the web UI dropdown (DESIGN.md B11) and the runtime API are
#: expected to offer, in the order DESIGN.md 4.7 lists them.
PRESET_NAMES = ("clean", "broadband", "3g", "satellite", "lossy", "custom")


@dataclass(frozen=True)
class Profile:
    """One netem parameter set. All fields are non-negative; percentages
    are 0-100. ``name`` is the preset label this came from ("custom" for
    an ad-hoc combination)."""

    name: str
    delay_ms: int = 0
    jitter_ms: int = 0
    loss_pct: float = 0.0
    reorder_pct: float = 0.0

    def as_dict(self) -> dict:
        return {
            "profile": self.name,
            "delay_ms": self.delay_ms,
            "jitter_ms": self.jitter_ms,
            "loss_pct": self.loss_pct,
            "reorder_pct": self.reorder_pct,
        }


# Presets, DESIGN.md 4.7's dropdown list. Numbers are illustrative
# real-world-ish figures (not measured against a real 3G/satellite link --
# a later phase can recalibrate), chosen so the six presets are clearly
# distinguishable from each other in an experiment:
#   clean      -- no injected impairment at all (the "perfect network" the
#                 project used to implicitly assume, now explicit and
#                 opt-in rather than the only option).
#   broadband  -- typical wired/cable home connection: small delay, no
#                 meaningful loss.
#   3g         -- mobile network: noticeable delay + jitter + a little loss.
#   satellite  -- GEO satellite link: large one-way delay dominates, plus
#                 some jitter and loss.
#   lossy      -- deliberately loss/reorder-heavy, low delay -- for
#                 exercising retransmission/reordering handling rather than
#                 RTT itself.
PRESETS: dict[str, Profile] = {
    "clean": Profile(name="clean", delay_ms=0, jitter_ms=0, loss_pct=0.0, reorder_pct=0.0),
    "broadband": Profile(name="broadband", delay_ms=15, jitter_ms=3, loss_pct=0.01, reorder_pct=0.0),
    "3g": Profile(name="3g", delay_ms=150, jitter_ms=40, loss_pct=1.0, reorder_pct=0.5),
    "satellite": Profile(name="satellite", delay_ms=600, jitter_ms=20, loss_pct=0.5, reorder_pct=0.1),
    "lossy": Profile(name="lossy", delay_ms=20, jitter_ms=5, loss_pct=5.0, reorder_pct=3.0),
}


def get_preset(name: str) -> Profile:
    """Look up a named preset. Raises ``KeyError`` for unknown names (and
    for "custom", which has no fixed values -- build it with
    :func:`custom_profile` instead)."""
    if name not in PRESETS:
        raise KeyError(
            f"unknown gateway profile {name!r}; known presets: "
            f"{', '.join(sorted(PRESETS))} (or 'custom' with explicit values)"
        )
    return PRESETS[name]


def custom_profile(
    *, delay_ms: int = 0, jitter_ms: int = 0, loss_pct: float = 0.0, reorder_pct: float = 0.0
) -> Profile:
    """Build a ``custom``-named profile from arbitrary values (POST body,
    or env fallback). Negative inputs are clamped to 0, matching
    ``aipt.core.netem.parse_delay``'s "never install a negative delay"
    stance."""
    return Profile(
        name="custom",
        delay_ms=max(0, int(delay_ms)),
        jitter_ms=max(0, int(jitter_ms)),
        loss_pct=max(0.0, float(loss_pct)),
        reorder_pct=max(0.0, float(reorder_pct)),
    )


def resolve(name: str, **overrides) -> Profile:
    """``clean``..``lossy`` -> the fixed preset (overrides ignored,
    matching "선택하면 그 프리셋" semantics); ``custom`` -> built from
    *overrides*."""
    if name == "custom":
        return custom_profile(**overrides)
    return get_preset(name)


def _env_int(*names: str, default: int = 0) -> int:
    for n in names:
        raw = os.environ.get(n)
        if raw is not None and raw.strip():
            try:
                return max(0, int(raw))
            except ValueError:
                continue
    return default


def _env_float(*names: str, default: float = 0.0) -> float:
    for n in names:
        raw = os.environ.get(n)
        if raw is not None and raw.strip():
            try:
                return max(0.0, float(raw))
            except ValueError:
                continue
    return default


def from_env() -> Profile:
    """Read the container's startup preset from the environment
    (DESIGN.md 4.7 설정 방식 (a)).

    ``GATEWAY_PROFILE`` selects a named preset directly (default
    ``"clean"``). If unset/``"custom"``, individual knobs are read from
    ``GATEWAY_DELAY_MS``/``GATEWAY_JITTER_MS``/``GATEWAY_LOSS_PCT``/
    ``GATEWAY_REORDER_PCT`` -- with ``CLIENT_NETEM_DELAY_MS`` and
    ``SERVER_NETEM_DELAY_MS`` (tcp_congestion's original ad-hoc knobs,
    DESIGN.md 4.7 배경) honoured as deprecated delay-only aliases when the
    canonical var is unset, so existing docker-compose.yml/.env files from
    the pre-Gateway setup keep working unmodified.
    """
    preset = os.environ.get("GATEWAY_PROFILE", "").strip().lower()
    if preset and preset != "custom":
        return get_preset(preset)

    delay_ms = _env_int("GATEWAY_DELAY_MS", "CLIENT_NETEM_DELAY_MS", "SERVER_NETEM_DELAY_MS", default=0)
    jitter_ms = _env_int("GATEWAY_JITTER_MS", default=0)
    loss_pct = _env_float("GATEWAY_LOSS_PCT", default=0.0)
    reorder_pct = _env_float("GATEWAY_REORDER_PCT", default=0.0)

    if delay_ms == 0 and jitter_ms == 0 and loss_pct == 0.0 and reorder_pct == 0.0:
        return PRESETS["clean"]
    return custom_profile(
        delay_ms=delay_ms, jitter_ms=jitter_ms, loss_pct=loss_pct, reorder_pct=reorder_pct
    )


__all__ = [
    "Profile",
    "PRESETS",
    "PRESET_NAMES",
    "get_preset",
    "custom_profile",
    "resolve",
    "from_env",
]
