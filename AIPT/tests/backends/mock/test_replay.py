"""aipt.backends.mock.replay: capture-to-fixture byte-pattern replay.

New tests (DESIGN.md 5, B3) -- no direct predecessor in tcp_congestion.
"""

import json

import pytest

from aipt.backends.mock import replay


def test_from_capture_doc_preserves_question_text():
    doc = {
        "name": "captured-run",
        "turns": [{"question": "what happened?", "answer": "the pool exhausted"}],
    }
    fx = replay.from_capture_doc(doc)
    assert fx.turns[0].question == "what happened?"


def test_from_capture_doc_replaces_answer_with_same_length_placeholder():
    original_answer = "the pool exhausted at 14:22 UTC after a slow query"
    doc = {"name": "captured-run",
           "turns": [{"question": "what happened?", "answer": original_answer}]}
    fx = replay.from_capture_doc(doc)
    replayed = fx.turns[0].answer
    assert replayed != original_answer          # content is not replayed
    assert len(replayed.encode()) == len(original_answer.encode())  # size is
    assert set(replayed) == {"x"}                # pure filler, no leaked content


def test_from_capture_doc_replaces_system_prompt_bytes_only():
    doc = {
        "name": "captured-run",
        "system_prompt": "You are ATLAS, a senior SRE agent.",
        "turns": [],
    }
    fx = replay.from_capture_doc(doc)
    assert fx.system_prompt != doc["system_prompt"]
    assert len(fx.system_prompt.encode()) == len(doc["system_prompt"].encode())


def test_from_capture_doc_empty_system_prompt_stays_empty():
    doc = {"name": "captured-run", "turns": []}
    fx = replay.from_capture_doc(doc)
    assert fx.system_prompt == ""


def test_from_capture_doc_description_notes_bytes_only():
    doc = {"name": "captured-run", "turns": []}
    fx = replay.from_capture_doc(doc)
    assert "bytes only" in fx.description
    assert "no timing" in fx.description


def test_from_capture_file_round_trips(tmp_path):
    doc = {
        "name": "file-capture",
        "turns": [{"question": "q1", "answer": "an answer with some length"}],
    }
    p = tmp_path / "file-capture.json"
    p.write_text(json.dumps(doc))
    fx = replay.from_capture_file(p)
    assert fx.turns[0].question == "q1"
    assert len(fx.turns[0].answer.encode()) == len("an answer with some length".encode())


def test_from_capture_doc_rejects_malformed_turn():
    doc = {"name": "bad", "turns": [{"question": "q"}]}  # missing answer
    with pytest.raises(ValueError):
        replay.from_capture_doc(doc)
