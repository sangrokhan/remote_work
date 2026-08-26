"""aipt.backends.mock.fixtures: Q&A fixture loading + byte-size sweep mode.

New tests (DESIGN.md 5, B1) -- no direct predecessor in tcp_congestion.
"""

import json

import pytest

from aipt.backends.mock import fixtures


def test_names_lists_registered_fixtures():
    assert "smoke" in fixtures.names()


def test_load_reads_smoke_fixture():
    fx = fixtures.load("smoke")
    assert fx.name == "smoke"
    assert fx.system_prompt == "You are a terse test assistant."
    assert len(fx) == 3
    assert fx.turns[0].question == "What is 2+2?"
    assert fx.turns[0].answer == "4"


def test_load_unknown_fixture_raises_keyerror():
    with pytest.raises(KeyError):
        fixtures.load("does-not-exist")


def test_turn_byte_properties():
    fx = fixtures.load("smoke")
    t0 = fx.turns[0]
    assert t0.question_bytes == len("What is 2+2?".encode())
    assert t0.answer_bytes == len("4".encode())


def test_load_qa_fixture_from_arbitrary_path(tmp_path):
    doc = {
        "name": "adhoc",
        "system_prompt": "sys",
        "turns": [{"question": "q", "answer": "a"}],
    }
    p = tmp_path / "adhoc.json"
    p.write_text(json.dumps(doc))
    fx = fixtures.load_qa_fixture(p)
    assert fx.name == "adhoc"
    assert len(fx) == 1
    assert fx.turns[0].question == "q"
    assert fx.turns[0].answer == "a"


def test_load_qa_fixture_rejects_missing_question(tmp_path):
    doc = {"name": "bad", "turns": [{"answer": "a"}]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        fixtures.load_qa_fixture(p)


def test_load_qa_fixture_rejects_missing_answer(tmp_path):
    doc = {"name": "bad", "turns": [{"question": "q"}]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        fixtures.load_qa_fixture(p)


def test_load_qa_fixture_rejects_non_object_turn(tmp_path):
    doc = {"name": "bad", "turns": ["not-an-object"]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        fixtures.load_qa_fixture(p)


# --- byte-size sweep mode ---------------------------------------------------


def test_byte_size_fixture_produces_requested_turn_count():
    fx = fixtures.byte_size_fixture(
        num_turns=3, turn_user_msg_bytes=100, mock_response_bytes=50)
    assert len(fx) == 3
    for turn in fx.turns:
        assert turn.answer_bytes == 50


def test_byte_size_fixture_folds_system_prompt_into_turn_zero_only():
    fx = fixtures.byte_size_fixture(
        num_turns=2, turn_user_msg_bytes=100, mock_response_bytes=50,
        system_prompt_bytes=1000)
    assert fx.turns[0].question_bytes == 1000 + 100
    assert fx.turns[1].question_bytes == 100


def test_byte_size_fixture_rejects_zero_turns():
    with pytest.raises(ValueError):
        fixtures.byte_size_fixture(
            num_turns=0, turn_user_msg_bytes=100, mock_response_bytes=50)
