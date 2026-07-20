"""The arms must not be able to read each other's prefix cache.

Every assertion here stands for a number that was already wrong once. The stateful arm
came back billed 4224 cached tokens on its own turn 1, off a prefix an earlier arm in
the same run had left warm, and chat_stateless's first-turn TTFT moved 1801 ms -> 662 ms
between two runs for no reason but cache state. Neither is visible in the run document;
both look like a fast arm.
"""

from __future__ import annotations

import pytest

from core import cachebust

SYSTEM = "You are a terse assistant."


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("TRAFFIC_CACHE_BUST", raising=False)


def test_two_arms_of_one_run_do_not_share_a_prefix():
    cachebust.begin("2026-07-14T09:52:30+00:00")
    a = cachebust.apply(SYSTEM, "openai", "responses_stateless")
    b = cachebust.apply(SYSTEM, "openai", "responses_inline")
    assert a != b
    # Distinct from the first token, not merely somewhere: a prefix cache matches from
    # the front, and a marker anywhere else leaves the shared part cacheable.
    assert a[0] != SYSTEM[0] and a.split("\n")[0] != b.split("\n")[0]


def test_two_providers_do_not_share_a_prefix():
    cachebust.begin("2026-07-14T09:52:30+00:00")
    assert (cachebust.apply(SYSTEM, "gemini", "stateless")
            != cachebust.apply(SYSTEM, "openai", "chat_stateless"))


def test_the_same_arm_run_twice_does_not_share_a_prefix():
    cachebust.begin("2026-07-14T09:52:30+00:00")
    first = cachebust.apply(SYSTEM, "openai", "chat_stateless")
    cachebust.begin("2026-07-14T10:11:00+00:00")
    assert cachebust.apply(SYSTEM, "openai", "chat_stateless") != first


def test_the_prefix_holds_still_inside_one_arm():
    """The marker is per-run, never per-turn. Turn 2 has to hit what turn 1 left warm --
    that is what a real client gets, and it is the thing being measured. A marker that
    moved between turns would miss on every turn and make cached_tokens noise."""
    cachebust.begin("2026-07-14T09:52:30+00:00")
    assert (cachebust.apply(SYSTEM, "gemini", "cached")
            == cachebust.apply(SYSTEM, "gemini", "cached"))


def test_every_arm_pays_the_same_number_of_marker_bytes():
    """A marker built from the arm's name would put the length of `interaction_stateless`
    into the input_tokens gap between it and `cached`. It is a fixed-width digest."""
    cachebust.begin("2026-07-14T09:52:30+00:00")
    lengths = {len(cachebust.apply(SYSTEM, "gemini", arm))
               for arm in ("cached", "stateless", "interaction_stateless")}
    assert len(lengths) == 1


def test_a_tag_is_derived_not_random():
    """Same run, same prefixes -- so a tag in an old record can be matched back to the
    arm that sent it, and a run can be replayed."""
    cachebust.begin("2026-07-14T09:52:30+00:00")
    first = cachebust.tag("openai", "chat_stateless")
    cachebust.begin("2026-07-14T09:52:30+00:00")
    assert cachebust.tag("openai", "chat_stateless") == first


def test_off_means_the_prompt_goes_out_untouched():
    cachebust.begin("2026-07-14T09:52:30+00:00", enabled=False)
    assert cachebust.apply(SYSTEM, "openai", "chat_stateless") == SYSTEM
    assert cachebust.tag("openai", "chat_stateless") == ""
    assert cachebust.tags([("openai", "chat_stateless")]) == {}


def test_the_env_flag_only_turns_it_off(monkeypatch):
    monkeypatch.setenv("TRAFFIC_CACHE_BUST", "0")
    assert cachebust.env_default() is False
    monkeypatch.setenv("TRAFFIC_CACHE_BUST", "1")
    assert cachebust.env_default() is True
    monkeypatch.delenv("TRAFFIC_CACHE_BUST")
    # Unset is on. The safe default is the reproducible one: a run that quietly inherited
    # a warm cache reads as a fast arm, not as a bug.
    assert cachebust.env_default() is True


class TestPrefixDrift:
    """The failure this module prevents, run on purpose as its own measurement.

    A marker that moves every turn misses the prefix cache every turn: the server cannot
    reuse the KV from turn k-1 and re-prefills the whole prompt. That is the argument --
    a system prompt must be byte-identical for the whole of a multi-turn task -- made
    runnable rather than asserted.
    """

    def test_off_by_default_so_a_run_is_not_quietly_sabotaged(self):
        cachebust.begin("2026-07-15T09:00:00+00:00")
        assert cachebust.drift_enabled() is False
        assert cachebust.per_turn(SYSTEM, 1) == SYSTEM
        assert cachebust.per_turn(SYSTEM, 7) == SYSTEM

    def test_every_turn_gets_a_different_prefix(self):
        cachebust.begin("2026-07-15T09:00:00+00:00", drift=True)
        seen = {cachebust.per_turn(SYSTEM, k) for k in range(1, 6)}
        assert len(seen) == 5, "five turns, five prefixes: the cache misses every turn"
        # Distinct from the first token. A prefix cache matches from the front, so a
        # marker anywhere else would leave the head of the prompt cacheable.
        assert cachebust.per_turn(SYSTEM, 1).startswith("[turn 001]")
        assert cachebust.per_turn(SYSTEM, 1).endswith(SYSTEM)

    def test_the_prompt_is_the_same_size_on_every_turn(self):
        """Fixed-width counter: the difference between a drift run and a still one has to
        be the cache, not the payload. A marker that grew with the turn number would put
        its own bytes into the input_tokens curve."""
        cachebust.begin("2026-07-15T09:00:00+00:00", drift=True)
        assert len({len(cachebust.per_turn(SYSTEM, k))
                    for k in (1, 9, 10, 99, 100)}) == 1

    def test_it_is_reproducible(self):
        cachebust.begin("2026-07-15T09:00:00+00:00", drift=True)
        first = cachebust.per_turn(SYSTEM, 3)
        cachebust.begin("2026-07-15T10:00:00+00:00", drift=True)
        assert cachebust.per_turn(SYSTEM, 3) == first, "a counter, not a random token"

    def test_an_empty_prompt_stays_empty(self):
        cachebust.begin("2026-07-15T09:00:00+00:00", drift=True)
        assert cachebust.per_turn("", 1) == ""


def test_an_empty_system_prompt_stays_empty():
    """An arm that sends no system prompt has no shared prefix to break. Inventing one
    would put a prompt on the wire the scenario never asked for."""
    cachebust.begin("2026-07-14T09:52:30+00:00")
    assert cachebust.apply("", "gemini", "nocontext") == ""
