"""Cache hits must not be a coin flip.

Measured against the live API: sending an identical 4k-token prefix five times
with no prompt_cache_key hit the cache 2/5 times, in no particular order. With a
stable prompt_cache_key it hit 4/5 — a cold write on the first call, then every
call after it. The hit depends on which inference node the request lands on, and
prompt_cache_key is what pins the routing.

Without this, cached_tokens is noise, and every cost number derived from it is
noise too. So: each arm carries a cache key that is stable across the turns of
one run, and distinct from the other arms' — otherwise one arm silently warms
another's cache and the comparison is contaminated.
"""

from __future__ import annotations

import pytest

import experiment
import openai_client as oc
import wire
from fake_openai import FakeOpenAI


@pytest.fixture
def fake(monkeypatch):
    srv = FakeOpenAI()
    monkeypatch.setenv("OPENAI_BASE_URL", srv.start())
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    wire.reset_session()
    yield srv
    srv.stop()
    wire.reset_session()


def _keys(bodies) -> list[str]:
    return [b.get("prompt_cache_key") for b in bodies]


def test_every_arm_sends_a_cache_key(fake):
    for arm in oc.ARMS:
        fake.requests.clear()
        experiment.run_arm(arm, experiment.fixture_mod.load("perf"),
                           model="test-model", turns=3, repeat=1)
        bodies = [r["body"] for r in fake.requests
                  if not r["path"].endswith("/conversations")]
        keys = _keys(bodies)
        assert all(keys), f"{arm} sent a call with no prompt_cache_key: {keys}"


def test_cache_key_is_stable_across_the_turns_of_one_run(fake):
    """Routing only sticks if the key does not change turn to turn."""
    experiment.run_arm("chat_stateless", experiment.fixture_mod.load("perf"),
                       model="test-model", turns=4, repeat=1)
    keys = _keys(fake.bodies_for("/chat/completions"))
    assert len(set(keys)) == 1, f"key changed mid-run: {keys}"


def test_arms_do_not_share_a_cache_key(fake):
    """A shared key would let one arm warm the next arm's cache. The arm that
    ran second would then look cheaper purely because it ran second."""
    seen = {}
    for arm in oc.ARMS:
        fake.requests.clear()
        experiment.run_arm(arm, experiment.fixture_mod.load("perf"),
                           model="test-model", turns=2, repeat=1)
        bodies = [r["body"] for r in fake.requests
                  if not r["path"].endswith("/conversations")]
        seen[arm] = _keys(bodies)[0]

    assert len(set(seen.values())) == len(oc.ARMS), f"arms share a key: {seen}"


def test_repeats_do_not_share_a_cache_key(fake):
    """Each repeat is meant to be an independent conversation, starting cold.
    Reusing the key across repeats would hand repeats 2..N a warm cache and make
    the averaged first-turn cost a fiction."""
    keys = []
    for repeat in (1, 2, 3):
        fake.requests.clear()
        experiment.run_arm("chat_stateless", experiment.fixture_mod.load("perf"),
                           model="test-model", turns=2, repeat=repeat)
        keys.append(_keys(fake.bodies_for("/chat/completions"))[0])

    assert len(set(keys)) == 3, f"repeats share a key: {keys}"
