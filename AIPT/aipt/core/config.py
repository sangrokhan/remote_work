"""config: the single place all AIPT env-var parsing lives.

This module absorbed two independent lineages:

  - `token_traffic/core/config.py`'s `flag()`/`is_mock()` -- the mock-vs-live
    switch. There used to be two readings of "is this call synthetic":
    `TRAFFIC_MOCK=true` satisfied one provider's parser and not the other's,
    so half a run was synthetic and half of it was billed, and the run was
    then filed in the *live* bucket because the flag that decides the bucket
    had its own third reading. Mock data indistinguishable from measured
    data is the one failure this package exists to make impossible, so the
    parse lives here and nowhere else.

  - the `_flag()` helpers that were copy-pasted, slightly differently each
    time, into `tcp_congestion/tcp_congestion/{offload,cwnd}.py`. Same idea
    (generous truthy parsing of an env var), same `{"1","true","yes","on"}`
    set, three separate definitions. `flag()` here is the one definition;
    `offload.py`/`cwnd.py` import it instead of redefining it.

  - `env_int()` generalizes the "parse an int env var, fall back to a
    default on anything invalid" pattern that used to be inlined (and
    silently duplicated with subtly different edge-case handling) in
    `cwnd.py`'s `interval_ms()`/`max_samples()`.

The parsing is deliberately generous: anything a person would plausibly type
to mean "yes" means yes. The dangerous direction is not accepting `on`; it is
one module accepting it while another does not.
"""

from __future__ import annotations

import os

_TRUE = {"1", "true", "yes", "on"}


def flag(name: str) -> bool:
    """Generous truthy parse of a single environment variable."""
    return (os.environ.get(name) or "").strip().lower() in _TRUE


def flag_any(*names: str) -> bool:
    """True if ANY of the given env vars is truthy.

    For a canonical env var with deprecated aliases: pass the canonical name
    first and the aliases after (order does not affect the result -- this is
    a plain OR -- but it documents which name is which for readers).
    """
    return any(flag(n) for n in names)


def env_int(name: str, default: int) -> int:
    """Parse an int-valued env var. Blank, missing, or non-numeric -> default.

    Unlike `parse_delay()`-style parsers elsewhere (e.g. `netem.parse_delay`,
    which deliberately *raises* on garbage input because a mistyped delay
    should fail loudly), this one is for the "tuning knob with a sane
    default" case: `TRAFFIC_CWND_INTERVAL_MS`, `TRAFFIC_CWND_MAX_SAMPLES`,
    and friends, where a bad value should fall back rather than crash the
    experiment.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        n = int(raw)
    except ValueError:
        return default
    return n


def env_str(name: str, default: str = "") -> str:
    """Plain string env var read with a default, for consistency with the
    numeric/bool helpers above (and so call sites read `config.env_str(...)`
    instead of reaching for `os.environ.get` directly)."""
    return os.environ.get(name, default)


def is_mock(provider: str = "") -> bool:
    """Whether calls are synthetic. `TRAFFIC_MOCK` covers the suite; `<PROVIDER>_MOCK`
    covers one provider, so a Gemini-only key can still exercise the OpenAI arms."""
    if flag("TRAFFIC_MOCK"):
        return True
    return bool(provider) and flag(f"{provider.upper()}_MOCK")
