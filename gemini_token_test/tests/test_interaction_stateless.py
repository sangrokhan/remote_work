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
    bodies = _bodies(monkeypatch, client_history=True, turns=4)
    assert [len(b["input"]) for b in bodies] == [1, 3, 5, 7]


def test_the_steps_alternate_user_then_model(monkeypatch):
    steps = _bodies(monkeypatch, client_history=True, turns=3)[-1]["input"]
    kinds = [s["type"] for s in steps]
    assert kinds == ["user_input", "model_output", "user_input",
                     "model_output", "user_input"]


def test_the_history_carries_this_arms_own_answers(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = ic.run_interaction("gemini-3.1-flash-lite", turns=2, client_history=True)
    recs = out["interaction_records"]
    sent = json.loads(recs[1]["request_raw"])["input"]
    answered = recs[0]["response_text"]
    assert sent[1]["content"][0]["text"] == answered


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
