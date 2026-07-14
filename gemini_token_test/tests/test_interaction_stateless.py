"""A fifth arm: the Interactions API with server-side state switched off.

The four existing arms vary two things at once -- which endpoint they call, and who
keeps the conversation history -- so `interaction` vs `stateless` never said which
of the two moved the numbers. This arm holds /interactions fixed and takes
previous_interaction_id away: store:false, and the client resends the whole
conversation as a Step[] every turn. What is left between it and `interaction` is
exactly what previous_interaction_id buys.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiment
import interaction_client as ic


def _bodies(monkeypatch, turns=3, **kw):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = ic.run_interaction("gemini-3.1-flash-lite", turns=turns, **kw)
    return [json.loads(r["request_raw"]) for r in out["interaction_records"]]


# --- the wire shape -------------------------------------------------------

def test_it_never_stores_the_interaction(monkeypatch):
    for b in _bodies(monkeypatch, client_history=True):
        assert b["store"] is False


def test_it_never_chains_on_a_previous_interaction(monkeypatch):
    for b in _bodies(monkeypatch, client_history=True):
        assert "previous_interaction_id" not in b


def test_it_sends_the_system_prompt_every_turn(monkeypatch):
    system, _, _ = experiment.load_request("perf")
    for b in _bodies(monkeypatch, client_history=True):
        assert b["system_instruction"] == system


def test_the_history_grows_by_a_question_and_an_answer_each_turn(monkeypatch):
    """A model turn is what the server sent back: a thought step *and* a
    model_output step. So each round adds three steps, not two."""
    bodies = _bodies(monkeypatch, client_history=True, turns=4)
    assert [len(b["input"]) for b in bodies] == [1, 4, 7, 10]


def test_the_steps_alternate_user_then_model(monkeypatch):
    steps = _bodies(monkeypatch, client_history=True, turns=3)[-1]["input"]
    kinds = [s["type"] for s in steps]
    assert kinds == ["user_input", "thought", "model_output",
                     "user_input", "thought", "model_output", "user_input"]


def test_the_history_carries_this_arms_own_answers(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = ic.run_interaction("gemini-3.1-flash-lite", turns=2, client_history=True)
    recs = out["interaction_records"]
    sent = json.loads(recs[1]["request_raw"])["input"]
    answered = recs[0]["response_text"]
    model_out = [s for s in sent if s["type"] == "model_output"]
    assert model_out[0]["content"][0]["text"] == answered


def test_the_questions_go_in_in_order(monkeypatch):
    _, steps, _ = experiment.load_request("perf")
    sent = _bodies(monkeypatch, client_history=True, turns=3)[-1]["input"]
    asked = [s["content"][0]["text"] for s in sent if s["type"] == "user_input"]
    assert asked == steps[:3]


# --- the existing arms must not move --------------------------------------

def test_the_interaction_arm_still_chains_and_stores(monkeypatch):
    for k, b in enumerate(_bodies(monkeypatch), start=1):
        assert b["store"] is True
        assert isinstance(b["input"], list) and len(b["input"]) == 1
        if k > 1:
            assert b["previous_interaction_id"]


def test_the_inline_arm_still_chains_and_stores(monkeypatch):
    for k, b in enumerate(_bodies(monkeypatch, inline_system=True), start=1):
        assert b["store"] is True
        assert len(b["input"]) == 1
        if k > 1:
            assert b["previous_interaction_id"]


def test_client_history_is_off_by_default(monkeypatch):
    b = _bodies(monkeypatch, turns=1)[0]
    assert b["store"] is True


# --- wired into the comparison --------------------------------------------

def test_the_arm_exists_and_is_a_headline_arm():
    assert "interaction_stateless" in experiment.COMPARE_ARMS
    assert "interaction_stateless" in experiment.DEFAULT_ARMS


def test_the_arm_produces_the_shared_record(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=2,
                                    arms=["interaction_stateless"])
    recs = [r for r in out["records"] if r["arm"] == "interaction_stateless"]
    assert [r["turn"] for r in recs] == [1, 2]
    assert all(r["phase"] == "steady" for r in recs)
    assert all("wire_sent" in r and "input_tokens" in r for r in recs)


def test_all_three_interaction_arms_run_side_by_side(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = experiment.run_comparison(
        "gemini-3.1-flash-lite", turns=1,
        arms=["interaction", "interaction_inline", "interaction_stateless"])
    assert {r["arm"] for r in out["records"]} == {
        "interaction", "interaction_inline", "interaction_stateless"}


def test_the_arm_resends_a_growing_history_through_the_comparison(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=3,
                                    arms=["interaction_stateless"])
    recs = sorted((r for r in out["records"] if r["arm"] == "interaction_stateless"),
                  key=lambda r: r["turn"])
    sizes = [len(json.loads(r["request_raw"])["input"]) for r in recs]
    assert sizes == [1, 4, 7]      # +user, +thought, +model_output each round


# --- mock input_tokens must reflect what the payload actually carries ------

def test_mock_input_tokens_grow_turn_over_turn(monkeypatch):
    """The arm's whole point is a superlinear input-token curve, like `stateless`.
    A flat curve in mock mode means the estimate is ignoring `history` -- the
    exact bug this test exists to catch."""
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = ic.run_interaction("gemini-3.1-flash-lite", turns=4, client_history=True)
    in_tok = [r["input_tokens"] for r in out["interaction_records"]]
    assert all(b > a for a, b in zip(in_tok, in_tok[1:])), in_tok


def test_mock_input_tokens_unchanged_for_the_chained_arm(monkeypatch):
    """This fix must only touch the client_history path. The chained arm's mock
    input_tokens are pinned to their pre-existing values."""
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = ic.run_interaction("gemini-3.1-flash-lite", turns=4)
    in_tok = [r["input_tokens"] for r in out["interaction_records"]]
    assert in_tok == [5292, 116, 107, 98]
