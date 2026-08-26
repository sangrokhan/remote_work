"""Each arm must put on the wire exactly what the experiment claims it does.

Ported from ``token_traffic/tests/test_provider_openai.py`` (DESIGN.md 5, A2)
onto ``aipt.backends.public_ai.openai``. Offline: mock mode, no key, no socket.
"""

from __future__ import annotations

import json

import pytest

from aipt.backends.public_ai import openai as p


@pytest.fixture(autouse=True)
def mock(monkeypatch):
    monkeypatch.setenv("TRAFFIC_MOCK", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p.reset_mock()
    yield
    p.reset_mock()


SYSTEM = "You are a terse assistant. " * 40
STEPS = [f"Question number {k}, and some words to give it a body." for k in range(1, 7)]


def _run(arm, turns=6, measure="bytes"):
    return p.run_arm(arm, "gpt-4.1-nano", SYSTEM, STEPS[:turns], measure)


def _steady(records):
    return [r for r in records if r["phase"] == "steady"]


def _bodies(records):
    return [json.loads(r["request_raw"]) for r in _steady(records)]


def test_ready_without_a_key_in_mock_mode():
    ok, why = p.ready()
    assert ok and "mock" in why


def test_every_arm_produces_a_record_per_turn():
    for arm in p.HEADLINE_ARMS:
        p.reset_mock()
        recs = _steady(_run(arm))
        assert len(recs) == len(STEPS)
        assert all(r["backend"] == "public_ai" and r.get("engine") == "openai"
                   for r in recs)


def test_chat_stateless_resends_everything():
    bodies = _bodies(_run("chat_stateless"))
    for k, body in enumerate(bodies, start=1):
        msgs = body["messages"]
        assert msgs[0]["role"] == "system"
        assert len(msgs) == 1 + 2 * (k - 1) + 1
        assert msgs[-1]["role"] == "user"
        assert body["stream"] is False


def test_responses_stateless_is_the_control_arm():
    bodies = _bodies(_run("responses_stateless"))
    for k, body in enumerate(bodies, start=1):
        assert body["input"][0]["role"] == "system"
        assert body["store"] is False
        assert "conversation" not in body
        assert len(body["input"]) == 1 + 2 * (k - 1) + 1


def test_the_inline_arm_creates_an_empty_conversation():
    records = _run("responses_inline", turns=3)
    setup = records[0]
    assert setup["phase"] == "setup"
    assert setup["turn"] == 0
    assert SYSTEM not in setup["request_raw"]
    assert json.loads(setup["request_raw"]) == {}
    assert setup["kind"] == "conversation_create"
    assert setup["billed"] is False
    assert setup["input_tokens"] == 0


def test_the_inline_arm_uploads_the_system_prompt_once_on_turn_one():
    bodies = _bodies(_run("responses_inline", turns=3))
    first, rest = bodies[0], bodies[1:]

    roles = [it["role"] for it in first["input"]]
    assert roles == ["system", "user"]
    assert "instructions" not in first

    for body in rest:
        assert [it["role"] for it in body["input"]] == ["user"]
        assert SYSTEM not in json.dumps(body)


def test_the_chained_arm_resends_the_system_prompt_every_turn():
    bodies = _bodies(_run("responses", turns=3))
    for body in bodies:
        assert body["instructions"] == SYSTEM
        assert [it["role"] for it in body["input"]] == ["user"]
        assert body["store"] is True
        assert "conversation" not in body


def test_the_chain_is_linked_turn_to_turn():
    records = _steady(_run("responses", turns=3))
    bodies = [json.loads(r["request_raw"]) for r in records]
    ids = [json.loads(r["response_raw"])["id"] for r in records]

    assert "previous_response_id" not in bodies[0]
    assert bodies[1]["previous_response_id"] == ids[0]
    assert bodies[2]["previous_response_id"] == ids[1]


def test_a_reasoning_item_is_echoed_back_verbatim(monkeypatch):
    monkeypatch.setattr(p, "REASONING_EFFORT", "low")
    bodies = _bodies(_run("responses_stateless", turns=3))

    echoed = [it for it in bodies[-1]["input"] if it.get("type") == "reasoning"]
    assert len(echoed) == 2
    assert all(it["encrypted_content"] for it in echoed)


def test_a_reasoning_summary_never_becomes_the_answer():
    assert p._responses_text_of(
        {"type": "response.reasoning_summary_text.delta", "delta": "hmm..."}) == ""
    assert p._responses_text_of(
        {"type": "response.output_text.delta", "delta": "the answer"}) == "the answer"


def test_latency_marks_only_exist_on_a_timed_pass():
    byte_pass = _steady(_run("chat_stateless", turns=2, measure="bytes"))
    assert all(r["ttft_ms"] == 0 for r in byte_pass)

    p.reset_mock()
    timed = _steady(_run("chat_stateless", turns=2, measure="latency"))
    for r in timed:
        assert 0 < r["ttfb_ms"] <= r["ttft_ms"] <= r["ttlt_ms"] <= r["turn_end_ms"]


def test_the_prompt_cache_key_rotates_with_the_run():
    from aipt.backends.public_ai import _cachebust as cachebust

    cachebust.begin("2026-07-14T09:52:30+00:00")
    first = p._cache_key("chat_stateless")
    assert first != p._cache_key("responses_stateless")

    cachebust.begin("2026-07-14T10:11:00+00:00")
    assert p._cache_key("chat_stateless") != first

    cachebust.begin("2026-07-14T10:11:00+00:00", enabled=False)
    assert p._cache_key("chat_stateless") == "tt-openai-chat_stateless"
    cachebust.reset()


# --- the OpenAIBackend connect/send_turn/close lifecycle -----------------------

def test_backend_chat_stateless_resends_everything():
    backend = p.OpenAIBackend()
    ok, why = backend.ready()
    assert ok and "mock" in why

    backend.connect(arm="chat_stateless", model="gpt-4.1-nano", system=SYSTEM)
    x1 = backend.send_turn(1, STEPS[0], "bytes")
    x2 = backend.send_turn(2, STEPS[1], "bytes")
    backend.close()

    b1 = json.loads(x1.request_json)["messages"]
    b2 = json.loads(x2.request_json)["messages"]
    assert len(b1) == 2  # system + q1
    assert len(b2) == 4  # system + q1 + a1 + q2


def test_backend_responses_inline_creates_conversation_on_connect():
    backend = p.OpenAIBackend()
    backend.connect(arm="responses_inline", model="gpt-4.1-nano", system=SYSTEM)
    assert len(backend.pending_setup_records) == 1
    setup = backend.pending_setup_records[0]
    assert setup["phase"] == "setup"
    assert setup["kind"] == "conversation_create"

    x1 = backend.send_turn(1, STEPS[0], "bytes")
    b1 = json.loads(x1.request_json)
    assert [it["role"] for it in b1["input"]] == ["system", "user"]
    x2 = backend.send_turn(2, STEPS[1], "bytes")
    b2 = json.loads(x2.request_json)
    assert [it["role"] for it in b2["input"]] == ["user"]
    backend.close()


def test_backend_responses_chains_previous_response_id():
    backend = p.OpenAIBackend()
    backend.connect(arm="responses", model="gpt-4.1-nano", system=SYSTEM)
    x1 = backend.send_turn(1, STEPS[0], "bytes")
    x2 = backend.send_turn(2, STEPS[1], "bytes")
    backend.close()

    b1 = json.loads(x1.request_json)
    b2 = json.loads(x2.request_json)
    assert "previous_response_id" not in b1
    assert b2["previous_response_id"] == json.loads(x1.response_json)["id"]


def test_backend_send_turn_before_connect_raises():
    backend = p.OpenAIBackend()
    with pytest.raises(RuntimeError):
        backend.send_turn(1, "hi", "bytes")


def test_backend_unknown_arm_raises_on_connect():
    backend = p.OpenAIBackend()
    with pytest.raises(ValueError):
        backend.connect(arm="telepathy", model="gpt-4.1-nano", system="")
