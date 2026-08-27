"""aipt.backends.mock.records: Q&A record loading + byte-size sweep mode.

New tests (DESIGN.md 5, B1) -- no direct predecessor in tcp_congestion.
"""

import json

import pytest

from aipt.backends.mock import records


def test_names_lists_registered_records():
    assert "smoke" in records.names()


def test_load_reads_smoke_record():
    rec = records.load("smoke")
    assert rec.name == "smoke"
    assert rec.system_prompt == "You are a terse test assistant."
    assert len(rec) == 3
    assert rec.turns[0].question == "What is 2+2?"
    assert rec.turns[0].answer == "4"


def test_load_unknown_record_raises_keyerror():
    with pytest.raises(KeyError):
        records.load("does-not-exist")


def test_turn_byte_properties():
    rec = records.load("smoke")
    t0 = rec.turns[0]
    assert t0.question_bytes == len("What is 2+2?".encode())
    assert t0.answer_bytes == len("4".encode())


def test_load_scenario_record_from_arbitrary_path(tmp_path):
    doc = {
        "name": "adhoc",
        "system_prompt": "sys",
        "turns": [{"question": "q", "answer": "a"}],
    }
    p = tmp_path / "adhoc.json"
    p.write_text(json.dumps(doc))
    rec = records.load_scenario_record(p)
    assert rec.name == "adhoc"
    assert len(rec) == 1
    assert rec.turns[0].question == "q"
    assert rec.turns[0].answer == "a"


def test_load_scenario_record_rejects_missing_question(tmp_path):
    doc = {"name": "bad", "turns": [{"answer": "a"}]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        records.load_scenario_record(p)


def test_load_scenario_record_rejects_missing_answer(tmp_path):
    doc = {"name": "bad", "turns": [{"question": "q"}]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        records.load_scenario_record(p)


def test_load_scenario_record_rejects_non_object_turn(tmp_path):
    doc = {"name": "bad", "turns": ["not-an-object"]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        records.load_scenario_record(p)


def test_load_scenario_record_accepts_steps_shape_with_answers(tmp_path):
    doc = {
        "name": "steps-doc",
        "system": ["part one", "part two"],
        "steps": [{"text": "q1", "answer": "a1"}, {"text": "q2", "answer": "a2"}],
    }
    p = tmp_path / "steps-doc.json"
    p.write_text(json.dumps(doc))
    rec = records.load_scenario_record(p)
    assert rec.name == "steps-doc"
    assert rec.system_prompt == "part one\n\npart two"
    assert len(rec) == 2
    assert rec.turns[0].question == "q1"
    assert rec.turns[0].answer == "a1"


def test_load_scenario_record_rejects_steps_missing_answer(tmp_path):
    doc = {"name": "bad-steps", "steps": [{"text": "q1"}]}
    p = tmp_path / "bad-steps.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        records.load_scenario_record(p)


def test_load_scenario_record_prefers_turns_over_steps_when_both_present(tmp_path):
    doc = {
        "name": "both",
        "turns": [{"question": "q", "answer": "a"}],
        "steps": [{"text": "ignored", "answer": "ignored"}],
    }
    p = tmp_path / "both.json"
    p.write_text(json.dumps(doc))
    rec = records.load_scenario_record(p)
    assert len(rec) == 1
    assert rec.turns[0].question == "q"


# --- byte-size sweep mode ---------------------------------------------------


def test_byte_size_scenario_produces_requested_turn_count():
    rec = records.byte_size_scenario(
        num_turns=3, turn_user_msg_bytes=100, mock_response_bytes=50)
    assert len(rec) == 3
    for turn in rec.turns:
        assert turn.answer_bytes == 50


def test_byte_size_scenario_folds_system_prompt_into_turn_zero_only():
    rec = records.byte_size_scenario(
        num_turns=2, turn_user_msg_bytes=100, mock_response_bytes=50,
        system_prompt_bytes=1000)
    assert rec.turns[0].question_bytes == 1000 + 100
    assert rec.turns[1].question_bytes == 100


def test_byte_size_scenario_rejects_zero_turns():
    with pytest.raises(ValueError):
        records.byte_size_scenario(
            num_turns=0, turn_user_msg_bytes=100, mock_response_bytes=50)
