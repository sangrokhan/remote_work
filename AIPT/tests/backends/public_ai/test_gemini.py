"""What the Gemini arms must put on the wire, checked offline against the mock.

Ported from ``token_traffic/tests/test_provider_gemini.py`` (DESIGN.md 5, A2)
onto ``aipt.backends.public_ai.gemini``. Two call surfaces are exercised:

  * ``run_arm`` -- the legacy full-conversation replay, kept for exact parity
    with the original token_traffic behaviour; every test below runs
    unmodified against it (only the import path and the record's
    ``provider`` -> ``backend`` column rename changed).
  * ``GeminiBackend`` (connect/send_turn/close) -- the new Backend-protocol
    lifecycle; a handful of tests drive it directly to confirm each arm's
    state machine produces the same wire shapes turn by turn.

Every test runs in mock mode. Nothing here touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aipt.backends.public_ai import gemini

RECORD_FILE = Path(__file__).resolve().parents[3] / "records" / "perf.json"
MODEL = gemini.DEFAULT_MODEL


@pytest.fixture(autouse=True)
def mock_mode(monkeypatch):
    monkeypatch.setenv("TRAFFIC_MOCK", "1")


def scenario(turns=3) -> tuple[str, list[str]]:
    data = json.loads(RECORD_FILE.read_text())
    system = data.get("system", "")
    if isinstance(system, list):
        system = "\n\n".join(system)
    steps = [s["text"] for s in data.get("steps", []) if s.get("text")]
    return system, steps[:turns]


def run(arm, turns=3, measure="bytes") -> list[dict]:
    system, steps = scenario(turns)
    return gemini.run_arm(arm, MODEL, system, steps, measure)


def bodies(recs) -> list[dict]:
    return [json.loads(r["request_raw"]) for r in recs]


def steady(recs) -> list[dict]:
    return [r for r in recs if r["phase"] == "steady"]


# --- the backend protocol / registry -----------------------------------------

def test_the_six_arms_are_declared():
    assert gemini.ARMS == ("stateless", "nocontext", "cached", "interaction",
                            "interaction_inline", "interaction_stateless")


def test_nocontext_is_a_diagnostic_not_a_headline():
    assert "nocontext" not in gemini.HEADLINE_ARMS
    assert set(gemini.HEADLINE_ARMS) == set(gemini.ARMS) - {"nocontext"}


def test_mock_mode_is_ready_without_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert gemini.ready() == (True, "")


def test_a_live_run_without_a_key_is_not_ready(monkeypatch):
    monkeypatch.delenv("TRAFFIC_MOCK", raising=False)
    monkeypatch.delenv("GEMINI_MOCK", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    ok, reason = gemini.ready()
    assert not ok and "GEMINI_API_KEY" in reason


@pytest.mark.parametrize("arm", gemini.ARMS)
def test_every_arm_produces_one_steady_record_per_turn(arm):
    recs = run(arm, turns=2)
    turns = steady(recs)
    assert [r["turn"] for r in turns] == [1, 2]
    assert all(r["backend"] == "public_ai" and r["arm"] == arm for r in recs)
    assert all(r.get("engine") == "gemini" for r in recs)
    assert all("wire_sent" in r and "input_tokens" in r for r in recs)


def test_an_unknown_arm_is_an_error_not_an_empty_run():
    with pytest.raises(ValueError):
        gemini.run_arm("telepathy", MODEL, "sys", ["q"], "bytes")


# --- stateless: the client keeps the history, signature and all ---------------

def test_the_stateless_history_grows_turn_over_turn():
    sizes = [len(b["contents"]) for b in bodies(run("stateless", turns=3))]
    assert sizes == [2, 4, 6]


def test_the_stateless_arm_echoes_the_thought_signature():
    contents = bodies(run("stateless", turns=3))[-1]["contents"]
    model_turns = [c for c in contents if c["role"] == "model"]
    assert len(model_turns) == 2
    assert all(p.get("thoughtSignature") for c in model_turns for p in c["parts"])


def test_the_stateless_input_tokens_grow_turn_over_turn():
    tokens = [r["input_tokens"] for r in run("stateless", turns=4)]
    assert all(b > a for a, b in zip(tokens, tokens[1:])), tokens


# --- nocontext: nobody keeps anything ----------------------------------------

def test_the_nocontext_arm_sends_no_history_at_all():
    contents = bodies(run("nocontext", turns=3))[-1]["contents"]
    assert [c["role"] for c in contents] == ["user"]


# --- cached: the prefix lives server-side -------------------------------------

def test_the_cache_builds_are_prep_and_the_turns_are_steady():
    recs = run("cached", turns=2)
    assert {r["phase"] for r in recs} == {"cachegen", "steady"}
    assert [r["turn"] for r in steady(recs)] == [1, 2]


def test_the_steady_turns_reference_the_cache_from_turn_two():
    turns = steady(run("cached", turns=3))
    sent = bodies(turns)
    assert "cachedContent" not in sent[0]
    assert all(b["cachedContent"].startswith("cachedContents/") for b in sent[1:])
    assert all(len(b["contents"]) == 1 for b in sent[1:])


def test_the_cached_turns_are_billed_for_the_cached_prefix():
    turns = steady(run("cached", turns=3))
    assert turns[0]["cached_tokens"] == 0
    assert all(r["cached_tokens"] > 0 for r in turns[1:])


# --- interaction: the server keeps the history --------------------------------

def test_the_chained_arm_sends_only_the_new_question():
    for b in bodies(run("interaction", turns=3)):
        assert len(b["input"]) == 1
        assert b["input"][0]["type"] == "user_input"


def test_the_chained_arm_carries_previous_interaction_id_from_turn_two():
    for k, b in enumerate(bodies(run("interaction", turns=3)), start=1):
        assert b["store"] is True
        assert bool(b.get("previous_interaction_id")) is (k > 1)


def test_the_inline_arm_puts_the_prompt_in_the_first_user_turn():
    system, steps = scenario(3)
    sent = bodies(run("interaction_inline", turns=3))
    assert "system_instruction" not in sent[0]
    assert sent[0]["input"][0]["content"][0]["text"].startswith(system[:60])
    assert sent[-1]["input"][0]["content"][0]["text"] == steps[2]


# --- interaction_stateless: the endpoint held fixed, the state taken away ------

def test_it_never_stores_the_interaction():
    for b in bodies(run("interaction_stateless", turns=3)):
        assert b["store"] is False


def test_it_never_chains_on_a_previous_interaction():
    for b in bodies(run("interaction_stateless", turns=3)):
        assert "previous_interaction_id" not in b


def test_the_signature_survives_the_round_trip():
    sent = bodies(run("interaction_stateless", turns=3))[-1]["input"]
    signatures = [s["signature"] for s in sent if s.get("type") == "thought"]
    assert len(signatures) == 2 and all(signatures)


# --- reasoning is not the answer, and a stored turn is not free ---------------

def test_the_answer_never_carries_the_models_reasoning():
    steps = [{"type": "thought", "signature": "SIG",
              "content": [{"type": "text", "text": "let me think"}]},
             {"type": "model_output", "content": [{"type": "text", "text": "Paris."}]}]
    assert gemini.answer_text(steps) == "Paris."


def test_a_thought_part_does_not_start_the_ttft_clock():
    event = {"candidates": [{"content": {"parts": [
        {"text": "reasoning...", "thought": True}, {"text": "Paris."}]}}]}
    assert gemini.gen_text(event) == "Paris."


def test_a_stored_interaction_pays_a_write_tail_after_its_last_token():
    stored = run("interaction", turns=1, measure="latency")[0]
    client = run("interaction_stateless", turns=1, measure="latency")[0]
    assert stored["turn_end_ms"] - stored["ttlt_ms"] >= gemini.MOCK_STORE_TAIL_MS
    assert client["turn_end_ms"] == client["ttlt_ms"]


# --- the GeminiBackend connect/send_turn/close lifecycle -----------------------

def test_backend_stateless_matches_run_arm_shape():
    system, steps = scenario(3)
    backend = gemini.GeminiBackend()
    ok, _ = backend.ready()
    assert ok

    backend.connect(arm="stateless", model=MODEL, system=system)
    exchanges = []
    for k, q in enumerate(steps, start=1):
        exchanges.append(backend.send_turn(k, q, "bytes"))
    backend.close()

    assert len(exchanges) == 3
    bodies_ = [json.loads(x.request_json)["contents"] for x in exchanges]
    assert [len(b) for b in bodies_] == [2, 4, 6]


def test_backend_nocontext_sends_only_the_question_after_turn_one():
    system, steps = scenario(3)
    backend = gemini.GeminiBackend()
    backend.connect(arm="nocontext", model=MODEL, system=system)
    x1 = backend.send_turn(1, steps[0], "bytes")
    x2 = backend.send_turn(2, steps[1], "bytes")
    backend.close()

    b1 = json.loads(x1.request_json)["contents"]
    b2 = json.loads(x2.request_json)["contents"]
    assert [c["role"] for c in b1] == ["user", "user"]
    assert [c["role"] for c in b2] == ["user"]


def test_backend_interaction_chains_from_turn_two():
    system, steps = scenario(3)
    backend = gemini.GeminiBackend()
    backend.connect(arm="interaction", model=MODEL, system=system)
    x1 = backend.send_turn(1, steps[0], "bytes")
    x2 = backend.send_turn(2, steps[1], "bytes")
    backend.close()

    b1 = json.loads(x1.request_json)
    b2 = json.loads(x2.request_json)
    assert "previous_interaction_id" not in b1
    assert b2["previous_interaction_id"] == json.loads(x1.response_json)["id"]


def test_backend_cached_arm_uses_a_server_side_cache_from_turn_two():
    """Online-caching adaptation (module docstring): turn 1 sends the system
    prompt plainly; by turn 2 a cache built from turn 1's transcript exists
    and is referenced instead."""
    system, steps = scenario(3)
    backend = gemini.GeminiBackend()
    backend.connect(arm="cached", model=MODEL, system=system)
    x1 = backend.send_turn(1, steps[0], "bytes")
    x2 = backend.send_turn(2, steps[1], "bytes")
    backend.close()

    b1 = json.loads(x1.request_json)
    b2 = json.loads(x2.request_json)
    assert "cachedContent" not in b1
    assert b2.get("cachedContent", "").startswith("cachedContents/")
    assert len(b2["contents"]) == 1


def test_backend_send_turn_before_connect_raises():
    backend = gemini.GeminiBackend()
    with pytest.raises(RuntimeError):
        backend.send_turn(1, "hi", "bytes")


def test_backend_unknown_arm_raises_on_connect():
    backend = gemini.GeminiBackend()
    with pytest.raises(ValueError):
        backend.connect(arm="telepathy", model=MODEL, system="")
