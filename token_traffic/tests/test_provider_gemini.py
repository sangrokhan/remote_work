"""What the Gemini arms must put on the wire, checked offline against the mock.

These are not shape tests for their own sake. Each one pins a fact that was paid for
once already, in a live run, and that a plausible-looking refactor silently breaks:

  - a client-side history that echoes the model's turn from the answer text drops the
    thought step and its signature, and under-reports its own upload by ~1 KB a turn;
  - a chained interaction that resends the history double-counts it;
  - a cached arm whose steady turns forget to reference the cache measures the
    stateless arm with extra steps;
  - a mock whose input tokens do not grow with the payload makes the arm whose entire
    point is a superlinear curve look flat.

Every test runs in mock mode. Nothing here touches the network.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from providers import base, gemini                              # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "perf.json"
MODEL = gemini.DEFAULT_MODEL


@pytest.fixture(autouse=True)
def mock_mode(monkeypatch):
    monkeypatch.setenv("TRAFFIC_MOCK", "1")


def scenario(turns=3) -> tuple[str, list[str]]:
    data = json.loads(FIXTURE.read_text())
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


# --- the provider protocol ---------------------------------------------------

def test_the_registry_hands_back_the_module():
    assert base.get("gemini") is gemini


def test_the_six_arms_are_declared():
    assert gemini.ARMS == ("stateless", "nocontext", "cached", "interaction",
                           "interaction_inline", "interaction_stateless")


def test_nocontext_is_a_diagnostic_not_a_headline():
    """It answers with no history at all. It is the floor the other arms are measured
    against, not a way anyone would run a chat."""
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
    assert all(r["provider"] == "gemini" and r["arm"] == arm for r in recs)
    assert all("wire_sent" in r and "input_tokens" in r for r in recs)


def test_an_unknown_arm_is_an_error_not_an_empty_run():
    """An arm that silently returns no records is a hole in a chart nobody notices."""
    with pytest.raises(ValueError):
        gemini.run_arm("telepathy", MODEL, "sys", ["q"], "bytes")


# --- stateless: the client keeps the history, signature and all ---------------

def test_the_stateless_history_grows_turn_over_turn():
    sizes = [len(b["contents"]) for b in bodies(run("stateless", turns=3))]
    assert sizes == [2, 4, 6]          # system + q1, +a1 +q2, +a2 +q3


def test_the_stateless_arm_echoes_the_thought_signature():
    """The model's turn goes back exactly as it came off the wire. Rebuilt from the
    answer text it would lose the signature -- and ~1 KB of upload a turn with it."""
    contents = bodies(run("stateless", turns=3))[-1]["contents"]
    model_turns = [c for c in contents if c["role"] == "model"]
    assert len(model_turns) == 2
    assert all(p.get("thoughtSignature") for c in model_turns for p in c["parts"])


def test_the_stateless_history_is_the_models_own_answer():
    recs = run("stateless", turns=2)
    said = [c for c in bodies(recs)[-1]["contents"]
            if c["role"] == "model"][0]["parts"][0]["text"]
    assert said == recs[0]["response_text"]


def test_the_stateless_input_tokens_grow_turn_over_turn():
    tokens = [r["input_tokens"] for r in run("stateless", turns=4)]
    assert all(b > a for a, b in zip(tokens, tokens[1:])), tokens


# --- nocontext: nobody keeps anything ----------------------------------------

def test_the_nocontext_arm_sends_no_history_at_all():
    contents = bodies(run("nocontext", turns=3))[-1]["contents"]
    assert [c["role"] for c in contents] == ["user"]


def test_the_nocontext_upload_stays_flat_after_the_first_turn():
    """The system prompt rides turn 1 alone; every later turn is one question."""
    sent = [r["req_payload_bytes"] for r in run("nocontext", turns=3)]
    assert sent[0] > sent[1] and abs(sent[1] - sent[2]) < sent[0] // 2


# --- cached: the prefix lives server-side -------------------------------------

def test_the_cache_builds_are_prep_and_the_turns_are_steady():
    recs = run("cached", turns=2)
    assert {r["phase"] for r in recs} == {"cachegen", "steady"}
    assert [r["turn"] for r in steady(recs)] == [1, 2]


def test_the_steady_turns_reference_the_cache_from_turn_two():
    """Turn 1 has no prior cache and sends the prompt itself; every later turn points
    at the cache holding the prefix and sends only its question."""
    turns = steady(run("cached", turns=3))
    sent = bodies(turns)
    assert "cachedContent" not in sent[0]
    assert all(b["cachedContent"].startswith("cachedContents/") for b in sent[1:])
    assert all(len(b["contents"]) == 1 for b in sent[1:])


def test_the_cached_turns_are_billed_for_the_cached_prefix():
    turns = steady(run("cached", turns=3))
    assert turns[0]["cached_tokens"] == 0
    assert all(r["cached_tokens"] > 0 for r in turns[1:])


def test_the_caches_are_built_from_the_conversation_that_actually_happened():
    """A cache of answers the model never gave measures nothing, so prep replays the
    conversation first and caches what came back."""
    recs = run("cached", turns=2)
    prep = [r for r in recs if r["phase"] == "cachegen"]
    answered = [r["response_text"] for r in prep if r["response_text"]]
    cache_bodies = [b for b in bodies(prep) if "ttl" in b]
    cached_text = json.dumps(cache_bodies[-1]["contents"])
    assert answered and all(a in cached_text for a in answered)


# --- interaction: the server keeps the history --------------------------------

def test_the_chained_arm_sends_only_the_new_question():
    for b in bodies(run("interaction", turns=3)):
        assert len(b["input"]) == 1
        assert b["input"][0]["type"] == "user_input"


def test_the_chained_arm_carries_previous_interaction_id_from_turn_two():
    for k, b in enumerate(bodies(run("interaction", turns=3)), start=1):
        assert b["store"] is True
        assert bool(b.get("previous_interaction_id")) is (k > 1)


def test_the_chained_arm_resends_the_system_prompt_every_turn():
    """system_instruction is interaction-scoped: the server keeps the conversation but
    not the instruction, so the prompt is re-uploaded on every single turn."""
    system, _ = scenario()
    for b in bodies(run("interaction", turns=3)):
        assert b["system_instruction"] == system


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


def test_it_sends_the_system_prompt_every_turn():
    system, _ = scenario()
    for b in bodies(run("interaction_stateless", turns=3)):
        assert b["system_instruction"] == system


def test_the_history_grows_by_a_question_and_a_model_turn_each_round():
    """A model turn is what the server sent back: a thought step *and* a model_output
    step. So each round adds three steps, not two."""
    sizes = [len(b["input"]) for b in bodies(run("interaction_stateless", turns=4))]
    assert sizes == [1, 4, 7, 10]


def test_the_steps_alternate_user_then_model():
    sent = bodies(run("interaction_stateless", turns=3))[-1]["input"]
    assert [s["type"] for s in sent] == [
        "user_input", "thought", "model_output",
        "user_input", "thought", "model_output", "user_input"]


def test_the_signature_survives_the_round_trip():
    sent = bodies(run("interaction_stateless", turns=3))[-1]["input"]
    signatures = [s["signature"] for s in sent if s.get("type") == "thought"]
    assert len(signatures) == 2 and all(signatures)


def test_the_history_carries_this_arms_own_answers():
    recs = run("interaction_stateless", turns=2)
    sent = bodies(recs)[1]["input"]
    model_out = [s for s in sent if s["type"] == "model_output"]
    assert model_out[0]["content"][0]["text"] == recs[0]["response_text"]


def test_the_questions_go_in_in_order():
    _, steps = scenario(3)
    sent = bodies(run("interaction_stateless", turns=3))[-1]["input"]
    asked = [s["content"][0]["text"] for s in sent if s["type"] == "user_input"]
    assert asked == steps


def test_its_input_tokens_grow_turn_over_turn():
    """The arm's whole point is a superlinear input-token curve, like stateless. A
    flat curve means the mock is ignoring the history the payload actually carries --
    the exact bug this test exists to catch."""
    tokens = [r["input_tokens"] for r in run("interaction_stateless", turns=4)]
    assert all(b > a for a, b in zip(tokens, tokens[1:])), tokens


def test_the_chained_arms_upload_does_not_grow_with_the_conversation():
    sent = [r["req_payload_bytes"] for r in run("interaction", turns=4)]
    grown = [r["req_payload_bytes"] for r in run("interaction_stateless", turns=4)]
    assert max(sent) - min(sent) < grown[-1] - grown[0]


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
    """Measured at ~1.8 s on the real endpoint. If the mock let the stored arms end on
    their last token, a mock run would report them as free."""
    stored = run("interaction", turns=1, measure="latency")[0]
    client = run("interaction_stateless", turns=1, measure="latency")[0]
    assert stored["turn_end_ms"] - stored["ttlt_ms"] >= gemini.MOCK_STORE_TAIL_MS
    assert client["turn_end_ms"] == client["ttlt_ms"]


def test_a_bigger_upload_takes_longer_to_put_on_the_wire():
    """req_sent is the mark the history-resending arms pay in, so the mock has to
    scale it with the payload or the arms all look equally fast."""
    marks = [r["req_sent_ms"] for r in run("interaction_stateless", turns=4,
                                           measure="latency")]
    assert marks[-1] > marks[0]
