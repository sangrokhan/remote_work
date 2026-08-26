"""aipt.backends.mock.conversation: pure prompt-size-growth logic.

Migrated from tcp_congestion/tests/test_conversation.py (DESIGN.md 5, A3).
Socket + cwnd-monitor integration is in test_conversation_live.py
(@pytest.mark.live).
"""

import pytest

from aipt.backends.mock import conversation


def test_turn_zero_prompt_is_system_prompt_plus_user_message():
    size = conversation.turn_prompt_size(
        turn_index=0, system_prompt_bytes=1000, turn_user_msg_bytes=200,
        history_bytes=0)
    assert size == 1200


def test_turn_one_prompt_excludes_system_prompt_but_includes_history():
    size = conversation.turn_prompt_size(
        turn_index=1, system_prompt_bytes=1000, turn_user_msg_bytes=200,
        history_bytes=1350)
    assert size == 1550


def test_history_accumulates_prompt_plus_response_each_turn():
    history = 0
    system_prompt = 1000
    turn_msg = 200
    response = 150

    sizes = []
    for i in range(4):
        prompt = conversation.turn_prompt_size(i, system_prompt, turn_msg, history)
        sizes.append(prompt)
        history += prompt + response

    assert sizes[0] == 1200
    assert sizes[1] == 1200 + 150 + 200
    assert sizes == sorted(sizes)


def test_build_turns_applies_system_prompt_only_on_first_turn():
    specs = conversation.build_turns(
        num_turns=3, system_prompt_bytes=1000, turn_user_msg_bytes=200,
        mock_response_bytes=150, inference_delay_ms=500, idle_duration_ms=1000)
    assert len(specs) == 3
    assert specs[0]["prompt_bytes"] == 1200
    assert specs[1]["prompt_bytes"] == 1200 + 150 + 200
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
