"""A fourth arm: the system prompt rides in the first user turn.

The Interactions API keeps the conversation server-side but not the
system_instruction -- that is interaction-scoped and must be re-sent every turn. Our
system prompt is 20 KB, so it is essentially the whole request, every turn, and the
interaction arm saves almost nothing on the wire despite the server holding the
history.

Put the system prompt in the first user message instead and it becomes part of that
server-side history: turn 1 costs the same, and every turn after it sends only the
question. Whether the model still obeys it as well is the open question -- that is
what the responses CSV is for -- and whether the stable server-side prefix now earns
an implicit cache hit is the other.

The existing interaction arm is left exactly as it was; this is a second one.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiment
import interaction_client as ic


def _bodies(monkeypatch, inline, turns=3):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = ic.run_interaction("gemini-3.1-flash-lite", turns=turns,
                             inline_system=inline)
    return [json.loads(r["request_raw"]) for r in out["interaction_records"]]


# --- the new mode ---------------------------------------------------------

def test_inline_never_sends_system_instruction(monkeypatch):
    for b in _bodies(monkeypatch, inline=True):
        assert "system_instruction" not in b


def test_inline_puts_the_system_prompt_in_the_first_user_turn(monkeypatch):
    system, steps, _ = experiment.load_request("perf")
    first = json.dumps(_bodies(monkeypatch, inline=True)[0])
    assert system[:200] in first
    assert steps[0][:80] in first


def test_inline_later_turns_send_only_the_question(monkeypatch):
    system, steps, _ = experiment.load_request("perf")
    bodies = _bodies(monkeypatch, inline=True)
    for k, b in enumerate(bodies[1:], start=2):
        blob = json.dumps(b)
        assert system[:200] not in blob, f"turn {k} re-sent the system prompt"
        assert b["previous_interaction_id"], f"turn {k} lost the history"


# --- the existing arm must not change -------------------------------------

def test_the_original_interaction_arm_is_untouched(monkeypatch):
    for b in _bodies(monkeypatch, inline=False):
        assert b["system_instruction"], "system_instruction is still sent every turn"


def test_inline_is_off_by_default(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = ic.run_interaction("gemini-3.1-flash-lite", turns=1)
    assert "system_instruction" in json.loads(out["interaction_records"][0]["request_raw"])


# --- wired into the comparison --------------------------------------------

def test_the_arm_exists_and_is_a_headline_arm():
    assert "interaction_inline" in experiment.COMPARE_ARMS
    assert "interaction_inline" in experiment.DEFAULT_ARMS


def test_the_arm_produces_the_shared_record(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=2,
                                    arms=["interaction_inline"])
    recs = [r for r in out["records"] if r["arm"] == "interaction_inline"]
    assert [r["turn"] for r in recs] == [1, 2]
    assert all(r["phase"] == "steady" for r in recs)
    assert all("wire_sent" in r and "input_tokens" in r for r in recs)


def test_both_interaction_arms_can_run_side_by_side(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=1,
                                    arms=["interaction", "interaction_inline"])
    arms = {r["arm"] for r in out["records"]}
    assert arms == {"interaction", "interaction_inline"}
