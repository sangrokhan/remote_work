"""conversation: multi-turn scenario with cumulative context growth.

Pure logic tests here (prompt-size growth, turn-result shape). The full
socket + cwnd-monitor integration is tested in test_conversation_live.py,
since it needs an actual TCP connection.

Two independent size knobs:
  - system_prompt_bytes: sent ONCE, folded into turn 0's request only.
    After that it lives inside the accumulated history like everything else,
    so it is never sent again as "new" bytes -- exactly the way a real chat
    client sends the system prompt once and then just keeps resending the
    growing transcript.
  - turn_user_msg_bytes: sent fresh EVERY turn (the new thing the user typed
    this turn), on top of whatever history has accumulated so far.

Growth rule: turn N's request = accumulated history (all previous turns'
sent bytes + mock responses) + this turn's new user message. Turn 0's
request additionally includes the system prompt, once.
"""

import pytest

from tcp_congestion import conversation


def test_turn_zero_prompt_is_system_prompt_plus_user_message():
    size = conversation.turn_prompt_size(
        turn_index=0, system_prompt_bytes=1000, turn_user_msg_bytes=200,
        history_bytes=0)
    assert size == 1200


def test_turn_one_prompt_excludes_system_prompt_but_includes_history():
    """System prompt is not re-sent as 'new' bytes on turn 1 -- it is already
    baked into history_bytes, which the caller carries forward."""
    size = conversation.turn_prompt_size(
        turn_index=1, system_prompt_bytes=1000, turn_user_msg_bytes=200,
        history_bytes=1350)  # e.g. turn0's 1200-byte prompt + 150-byte response
    assert size == 1550


def test_history_accumulates_prompt_plus_response_each_turn():
    """history_after = this turn's full prompt (as sent) + the mock response."""
    history = 0
    system_prompt = 1000
    turn_msg = 200
    response = 150

    sizes = []
    for i in range(4):
        prompt = conversation.turn_prompt_size(i, system_prompt, turn_msg, history)
        sizes.append(prompt)
        history += prompt + response

    assert sizes[0] == 1200                      # system + msg only
    assert sizes[1] == 1200 + 150 + 200           # turn0 total + response + new msg
    assert sizes == sorted(sizes)                 # strictly increasing


def test_build_turns_applies_system_prompt_only_on_first_turn():
    specs = conversation.build_turns(
        num_turns=3, system_prompt_bytes=1000, turn_user_msg_bytes=200,
        mock_response_bytes=150, inference_delay_ms=500, idle_duration_ms=1000)
    assert len(specs) == 3
    assert specs[0]["prompt_bytes"] == 1200                  # 1000 + 200
    assert specs[1]["prompt_bytes"] == 1200 + 150 + 200       # no system prompt again
    assert specs[2]["prompt_bytes"] > specs[1]["prompt_bytes"]
    for s in specs:
        assert s["inference_delay_ms"] == 500
        assert s["idle_duration_ms"] == 1000


def test_build_turns_with_zero_system_prompt_behaves_like_plain_growth():
    specs = conversation.build_turns(
        num_turns=2, system_prompt_bytes=0, turn_user_msg_bytes=200,
        mock_response_bytes=150, inference_delay_ms=500, idle_duration_ms=1000)
    assert specs[0]["prompt_bytes"] == 200
    assert specs[1]["prompt_bytes"] == 200 + 150 + 200


def test_build_turns_rejects_zero_turns():
    with pytest.raises(ValueError):
        conversation.build_turns(num_turns=0, system_prompt_bytes=1000,
                                  turn_user_msg_bytes=200, mock_response_bytes=150,
                                  inference_delay_ms=500, idle_duration_ms=1000)
