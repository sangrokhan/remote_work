"""Each arm must put on the wire exactly what the experiment claims it does.

If responses_stateful quietly resent the history, the whole result would be a
tautology. These tests pin the request bodies.
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


def _run(fake, arm, turns=4):
    return experiment.run_arm(arm, experiment.fixture_mod.load("perf"),
                              model="test-model", turns=turns, repeat=1)


def test_chat_stateless_resends_everything(fake):
    _run(fake, "chat_stateless", turns=4)
    bodies = fake.bodies_for("/chat/completions")
    assert len(bodies) == 4

    # turn k carries system + 2*(k-1) prior messages + the new question
    for k, body in enumerate(bodies, start=1):
        msgs = body["messages"]
        assert msgs[0]["role"] == "system"
        assert len(msgs) == 1 + 2 * (k - 1) + 1
        assert msgs[-1]["role"] == "user"
    assert body["stream"] is False


def test_responses_stateless_resends_everything(fake):
    _run(fake, "responses_stateless", turns=4)
    bodies = fake.bodies_for("/responses")
    assert len(bodies) == 4
    for k, body in enumerate(bodies, start=1):
        items = body["input"]
        assert items[0]["role"] == "system"
        assert len(items) == 1 + 2 * (k - 1) + 1
        assert body["store"] is False
        assert "conversation" not in body


def test_responses_stateful_sends_only_the_new_question(fake):
    run = _run(fake, "responses_stateful", turns=4)

    conv_bodies = fake.bodies_for("/conversations")
    assert len(conv_bodies) == 1, "system prompt is uploaded once, at setup"

    bodies = fake.bodies_for("/responses")
    assert len(bodies) == 4
    for body in bodies:
        assert len(body["input"]) == 1
        assert body["input"][0]["role"] == "user"
        assert body["conversation"] == "conv_1"
        # the system prompt must NOT ride along on any turn
        assert "system" not in str(body["input"])

    assert run.setup["req_payload_bytes"] > 0, "setup bytes are counted, not hidden"


def test_stateful_upload_stays_flat_while_stateless_grows(fake):
    """The shape of the whole experiment, in one assertion."""
    stateless = _run(fake, "chat_stateless", turns=6)
    fake.requests.clear()
    stateful = _run(fake, "responses_stateful", turns=6)

    sl = [t["req_payload_bytes"] for t in stateless.turns]
    sf = [t["req_payload_bytes"] for t in stateful.turns]

    assert sl[-1] > sl[0], "stateless upload grows every turn"
    # stateful turns carry only a question, so they track question length, not history
    assert max(sf) < min(sl), "every stateful turn uploads less than the smallest stateless turn"
    assert sum(sf) * 5 < sum(sl), "the cumulative gap is not marginal"
