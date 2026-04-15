from python_backend.app.run_state_machine import can_transition, assert_transition, RUN_STATES, TERMINAL_STATES


def test_run_states_include_checklist_states():
    assert "queued" in RUN_STATES
    assert "running" in RUN_STATES
    assert "awaiting_input" in RUN_STATES
    assert "completed" in RUN_STATES
    assert "failed" in RUN_STATES
    assert "cancelled" in RUN_STATES
    assert "completed" in TERMINAL_STATES
    assert "failed" in TERMINAL_STATES
    assert "cancelled" in TERMINAL_STATES


def test_valid_transitions_for_queued():
    assert can_transition("queued", "running") is True
    assert can_transition("queued", "cancelled") is True


def test_invalid_transitions_for_queued():
    assert can_transition("queued", "awaiting_input") is False
    assert can_transition("queued", "running") is True


def test_valid_transitions_for_running():
    assert can_transition("running", "awaiting_input") is True
    assert can_transition("running", "completed") is True
    assert can_transition("running", "failed") is True
    assert can_transition("running", "cancelled") is True


def test_terminal_states_cannot_transition():
    assert can_transition("completed", "running") is False
    assert can_transition("failed", "cancelled") is False
    assert can_transition("cancelled", "running") is False


def test_invalid_state_transition_raises():
    try:
        assert_transition("running", "queued")
    except Exception as error:
        assert type(error).__name__ == "InvalidRunTransitionError"
        assert getattr(error, "code", None) == "INVALID_RUN_TRANSITION"
    else:
        raise AssertionError("expected InvalidRunTransitionError")
