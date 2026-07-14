"""A client-side history must echo what the server sent, not a paraphrase of it.

Every real response carries the model's turn as *two* steps -- a `thought` step
holding an encrypted signature, then the `model_output` step holding the text:

    {"type": "thought", "signature": "EjQKMg..."}
    {"type": "model_output", "content": [{"type": "text", "text": "..."}]}

generateContent says the same thing in its own vocabulary: the candidate's parts
come back as {"text": ..., "thoughtSignature": ...}.

The arms that keep the history client-side used to rebuild the model turn from the
answer text alone, which silently dropped the thought step on every turn. The
chained arm does not drop it -- the server stores the steps it produced -- so the
two arms were replaying different conversations, and the client-history arm was
under-reporting its own upload by the size of the signatures it never sent.

Measured (probe.probe_signature_echo, gemini-3.1-flash-lite, 2026-07-14): echoing
the thought step is accepted (200) and costs 0 extra input tokens; dropping it is
also accepted. So this is not about the token bill. It is about sending what a real
client sends -- signatures are mandatory once tools enter the picture -- and about
the ~1 KB per turn of upload that the honest arm has to carry.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiment
import interaction_client as ic
import payloads


# --- payloads: the shapes come straight out of the response ----------------

RESPONSE = {
    "id": "int_1",
    "steps": [
        {"signature": "SIG-ABC", "type": "thought"},
        {"content": [{"text": "Paris.", "type": "text"}], "type": "model_output"},
    ],
}

CANDIDATE_RESPONSE = {
    "candidates": [{
        "content": {"role": "model",
                    "parts": [{"text": "Paris.", "thoughtSignature": "SIG-ABC"}]},
    }],
}


def test_model_steps_come_back_verbatim():
    assert payloads.model_steps_from_response(RESPONSE) == RESPONSE["steps"]


def test_the_thought_step_keeps_its_signature():
    steps = payloads.model_steps_from_response(RESPONSE)
    assert any(s.get("signature") == "SIG-ABC" for s in steps)


def test_a_response_without_steps_falls_back_to_the_text():
    steps = payloads.model_steps_from_response({}, fallback_text="Paris.")
    assert steps == [payloads.model_step("Paris.")]


def test_the_candidate_content_keeps_its_thought_signature():
    content = payloads.model_content_from_response(CANDIDATE_RESPONSE)
    assert content["role"] == "model"
    assert content["parts"][0]["thoughtSignature"] == "SIG-ABC"
    assert content["parts"][0]["text"] == "Paris."


def test_a_candidateless_response_falls_back_to_the_text():
    content = payloads.model_content_from_response({}, fallback_text="Paris.")
    assert content == payloads.model_content("Paris.")


def test_the_answer_text_reads_model_output_only():
    """A thought step with a summary in it is reasoning, not the answer. Collecting
    every text leaf in the response would staple the two together."""
    resp = {"steps": [
        {"type": "thought", "signature": "SIG",
         "content": [{"type": "text", "text": "let me think"}]},
        {"type": "model_output", "content": [{"type": "text", "text": "Paris."}]},
    ]}
    assert payloads.answer_text(resp["steps"]) == "Paris."


# --- interactions: the client-history arm echoes the response --------------

def _records(monkeypatch, **kw):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    return ic.run_interaction("gemini-3.1-flash-lite", turns=3,
                              **kw)["interaction_records"]


def test_the_record_keeps_the_response_steps(monkeypatch):
    """Without the steps on the record there is nothing to echo, and nothing to
    audit afterwards."""
    for r in _records(monkeypatch, client_history=True):
        assert r["response_steps"], r
        assert any(s.get("type") == "thought" for s in r["response_steps"])


def test_the_history_carries_the_thought_step_back(monkeypatch):
    recs = _records(monkeypatch, client_history=True)
    sent = json.loads(recs[-1]["request_raw"])["input"]
    kinds = [s["type"] for s in sent]
    assert kinds == ["user_input", "thought", "model_output",
                     "user_input", "thought", "model_output", "user_input"]


def test_the_echoed_steps_are_the_ones_the_server_sent(monkeypatch):
    recs = _records(monkeypatch, client_history=True)
    sent = json.loads(recs[1]["request_raw"])["input"]
    assert sent[1:3] == recs[0]["response_steps"]


def test_the_signature_survives_the_round_trip(monkeypatch):
    recs = _records(monkeypatch, client_history=True)
    sent = json.loads(recs[-1]["request_raw"])["input"]
    signatures = [s["signature"] for s in sent if s.get("type") == "thought"]
    assert len(signatures) == 2 and all(signatures)


def test_the_chained_arm_still_sends_only_the_new_question(monkeypatch):
    """The server holds the steps for this arm. Echoing them would double them."""
    for b in (json.loads(r["request_raw"]) for r in _records(monkeypatch)):
        assert len(b["input"]) == 1
        assert b["input"][0]["type"] == "user_input"


# --- generateContent: the stateless arm echoes the response ----------------

def test_the_stateless_arm_echoes_the_thought_signature(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    recs = experiment._arm_stateless("gemini-3.1-flash-lite", "sys",
                                     ["q1", "q2", "q3"])
    contents = json.loads(recs[-1]["request_raw"])["contents"]
    model_turns = [c for c in contents if c["role"] == "model"]
    assert len(model_turns) == 2
    assert all(p.get("thoughtSignature") for c in model_turns for p in c["parts"])


def test_the_stateless_history_is_the_models_own_answer(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    recs = experiment._arm_stateless("gemini-3.1-flash-lite", "sys", ["q1", "q2"])
    contents = json.loads(recs[-1]["request_raw"])["contents"]
    said = [c for c in contents if c["role"] == "model"][0]["parts"][0]["text"]
    assert said == recs[0]["response_text"]


def test_the_nocontext_arm_sends_no_history_at_all(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    recs = experiment._arm_nocontext("gemini-3.1-flash-lite", "sys", ["q1", "q2"])
    contents = json.loads(recs[-1]["request_raw"])["contents"]
    assert [c["role"] for c in contents] == ["user"]
