"""Run state machine primitives for Python backend parity with JS implementation."""

from __future__ import annotations

RUN_STATES = [
    "queued",
    "running",
    "awaiting_input",
    "completed",
    "failed",
    "cancelled",
]

TERMINAL_STATES = {"completed", "failed", "cancelled"}

RUN_TRANSITIONS = {
    "queued": ["running", "cancelled"],
    "running": ["awaiting_input", "completed", "failed", "cancelled"],
    "awaiting_input": ["running", "failed", "cancelled"],
    "completed": [],
    "failed": [],
    "cancelled": [],
}

RUN_TRANSITION_TABLE = {from_state: set(to_states) for from_state, to_states in RUN_TRANSITIONS.items()}


class InvalidRunTransitionError(Exception):
    def __init__(self, from_state: str, to_state: str):
        super().__init__(f"invalid run transition: {from_state} -> {to_state}")
        self.name = "InvalidRunTransitionError"
        self.code = "INVALID_RUN_TRANSITION"
        self.from_state = from_state
        self.to_state = to_state


def is_valid_run_state(value: str | None) -> bool:
    return value in RUN_STATES


def is_terminal_state(value: str | None) -> bool:
    return value in TERMINAL_STATES


def can_transition(from_state: str, to_state: str) -> bool:
    if not is_valid_run_state(from_state) or not is_valid_run_state(to_state):
        return False
    return to_state in RUN_TRANSITION_TABLE[from_state]


def assert_transition(from_state: str, to_state: str) -> None:
    if not can_transition(from_state, to_state):
        raise InvalidRunTransitionError(from_state, to_state)
