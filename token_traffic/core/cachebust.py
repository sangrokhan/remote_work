"""A cold prefix for every arm, so no arm inherits another one's cache.

Both vendors cache on an exact token prefix, and both do it implicitly -- there is no
"don't cache" flag on either API. Whatever prefix a run puts on the wire is left warm
on some node afterwards, and the next run to send the same prefix reads its neighbour's
cache. The identical system prompt that makes the arms comparable is exactly what makes
them contaminate each other.

Measured, before this module existed:

    openai:responses_inline  turn 1  input=4445  cached=4224

Turn 1 of an arm that had sent nothing before it, billed as 95% cached. The prefix came
from `responses_stateless`, three arms earlier in the same run. And across runs, the
same cold/warm split moved chat_stateless's first-turn TTFT from 1801 ms to 662 ms --
a 3x swing that belongs to cache state and not to anything the experiment varies.

`prompt_cache_key` does not fix this. It biases which node a request lands on; it does
not namespace the cache. Two arms with different keys still hit each other's prefix
whenever routing puts them together, which is what the run above caught happening.

So the prefix itself is made distinct: a per-(run, provider, arm) marker on the front of
the system prompt. In front, because a prefix cache matches from the first token -- a
marker anywhere else leaves everything before it shared and cacheable.

Two properties the marker has to have, and one it must not:

  Constant within an arm. Turn 2 must still hit the prefix turn 1 left warm; that is
  what a real client gets, and it is the thing being measured. A marker that varied per
  turn would miss on every turn and turn cached_tokens into noise -- the failure the
  openai adapter's docstring warned about, and a different failure from this one.

  Fixed width, whatever the arm is called. The marker is a 16-hex digest, not the arm's
  name: `interaction_stateless` and `cached` would otherwise carry system prompts of
  different token counts, and the input_tokens gap between two arms would have the
  length of their names in it.

  Not secret, and not skipped. Every tag a run used is recorded in params, so a run's
  prefixes can be reconstructed and a warm hit can be explained rather than guessed at.

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

# Set once per run by core.runner, read by whoever needs the tag -- the runner to build
# the system prompt, the openai adapter to build its prompt_cache_key. Run-level, never
# per-arm: there is no "current arm" hiding here, so an arm's tag is the same value
# whoever asks for it and whenever.
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
    """The system prompt for turn `turn`, with a counter in front when drift is on.

    This is the negative control, and it is the one thing the rest of this module is built
    to prevent. A prefix cache matches from the first token, so a marker that moves every
    turn misses on every turn: the server cannot reuse the KV it computed for turn k-1 and
    re-prefills the entire prompt -- and, on the arms that keep a history, the entire
    conversation behind it -- from scratch. Expect `cached_tokens` to collapse to ~0 and
    TTFT to rise with the length of the prompt rather than sit flat.

    That is the argument this makes runnable: a system prompt must be byte-identical for
    the whole of a multi-turn or agentic task. Anything genuinely per-turn -- a timestamp,
    a turn counter, a request id -- belongs *after* the stable prefix, never in front of
    it. Turn it on to measure what putting it in front costs.

    A fixed-width counter, so the prompt is the same token count on every turn: the
    difference between a drift run and a still one is then the cache, not the payload.
    """
    if not _RUN["drift"] or not system:
        return system
    return _TURN_MARKER.format(turn=turn) + system


def tag(provider: str, arm: str) -> str:
    """This arm's marker in this run, or "" when cache-busting is off.

    Derived, not random: the same run replayed from the same timestamp rebuilds the same
    prefixes, and a tag in an old record can be matched back to the arm that sent it.
    """
    if not _RUN["enabled"]:
        return ""
    seed = f"{_RUN['timestamp']}|{provider}|{arm}".encode()
    return hashlib.sha256(seed).hexdigest()[:_TAG_LEN]


def apply(system: str, provider: str, arm: str) -> str:
    """The system prompt this arm should send: the marker, then the prompt.

    An empty system prompt stays empty. The marker exists to make a shared prefix
    distinct, and an arm that sends no system prompt has no shared prefix to break --
    inventing one would put a system prompt on the wire that the scenario never asked
    for, and the arm would stop being the arm.
    """
    t = tag(provider, arm)
    if not t or not system:
        return system
    return _MARKER.format(tag=t) + system


def tags(pairs) -> dict:
    """{"provider:arm": tag} for every pair in the run. What params records."""
    return {f"{p}:{a}": tag(p, a) for p, a in pairs if tag(p, a)}
