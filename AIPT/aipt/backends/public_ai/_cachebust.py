"""A cold prefix for every arm, so no arm inherits another one's cache.

Ported from ``token_traffic/core/cachebust.py`` (DESIGN.md 5, A2). Kept
private to ``aipt.backends.public_ai`` -- this is a public_ai-specific
concern (the two vendors' implicit prefix caching), not a general core
utility.

Both vendors cache on an exact token prefix, and both do it implicitly -- there is no
"don't cache" flag on either API. Whatever prefix a run puts on the wire is left warm
on some node afterwards, and the next run to send the same prefix reads its neighbour's
cache. The identical system prompt that makes the arms comparable is exactly what makes
them contaminate each other.

`prompt_cache_key` does not fix this. It biases which node a request lands on; it does
not namespace the cache. Two arms with different keys still hit each other's prefix
whenever routing puts them together.

So the prefix itself is made distinct: a per-(run, backend, arm) marker on the front of
the system prompt. In front, because a prefix cache matches from the first token -- a
marker anywhere else leaves everything before it shared and cacheable.

Off with TRAFFIC_CACHE_BUST=0, which restores the old warm behaviour on purpose: the
difference between a cold arm and one running on a neighbour's cache is itself worth a
measurement, and it is only available if the flag can be turned off.
"""

from __future__ import annotations

import hashlib
import os

# Bracketed and on its own paragraph: a model reads it as a tag rather than as an
# instruction, and every arm's prompt is offset by exactly the same tokens.
_MARKER = "[run {tag}]\n\n"
_TAG_LEN = 16                       # 64 bits of digest; collisions are not a concern

_FALSE = {"0", "false", "no", "off"}

# The drifting marker: the same idea as the run marker, moved one level down so that it
# changes every *turn* instead of every run. Fixed width (a zero-padded counter), so the
# system prompt is the same number of tokens on every turn and the only thing that moves
# is which tokens they are.
_TURN_MARKER = "[turn {turn:03d}]\n\n"

# Set once per run, read by whoever needs the tag -- the adapter to build the system
# prompt, the openai adapter to build its prompt_cache_key. Run-level, never per-arm:
# there is no "current arm" hiding here, so an arm's tag is the same value whoever
# asks for it and whenever.
_RUN = {"timestamp": "", "enabled": False, "drift": False}


def env_default() -> bool:
    """On unless the operator turned it off. The safe default is the reproducible one:
    a warm run that nobody asked for looks like a fast arm, not like a bug."""
    return (os.environ.get("TRAFFIC_CACHE_BUST") or "").strip().lower() not in _FALSE


def begin(timestamp: str, enabled: bool | None = None,
          drift: bool = False) -> None:
    """Open a run. `enabled=None` defers to TRAFFIC_CACHE_BUST.

    `drift` turns on the per-turn marker -- the failure this module's whole design exists
    to avoid, run deliberately as its own measurement. See `per_turn`.
    """
    _RUN["timestamp"] = timestamp or ""
    _RUN["enabled"] = env_default() if enabled is None else bool(enabled)
    _RUN["drift"] = bool(drift)


def enabled() -> bool:
    return bool(_RUN["enabled"])


def drift_enabled() -> bool:
    return bool(_RUN["drift"])


def per_turn(system: str, turn: int) -> str:
    """The system prompt for turn `turn`, with a counter in front when drift is on."""
    if not _RUN["drift"] or not system:
        return system
    return _TURN_MARKER.format(turn=turn) + system


def tag(backend: str, arm: str) -> str:
    """This arm's marker in this run, or "" when cache-busting is off.

    Derived, not random: the same run replayed from the same timestamp rebuilds the same
    prefixes, and a tag in an old record can be matched back to the arm that sent it.
    """
    if not _RUN["enabled"]:
        return ""
    seed = f"{_RUN['timestamp']}|{backend}|{arm}".encode()
    return hashlib.sha256(seed).hexdigest()[:_TAG_LEN]


def apply(system: str, backend: str, arm: str) -> str:
    """The system prompt this arm should send: the marker, then the prompt."""
    t = tag(backend, arm)
    if not t or not system:
        return system
    return _MARKER.format(tag=t) + system


def reset() -> None:
    """Reset run state -- for tests that need a clean slate between cases."""
    _RUN.update(timestamp="", enabled=False, drift=False)
