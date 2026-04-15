from datetime import datetime

from python_backend.app.run_error_contract import RunNotFoundError
from python_backend.app.run_state_machine import InvalidRunTransitionError
from python_backend.app.run_state_store import RunStateStore


def test_create_run_sets_started_event_and_cursor():
    store = RunStateStore()
    started_event = store.create_run(
        run_id="run_abc",
        thread_id="thread_01",
        workflow_id="support_ticket_classifier_v1",
    )

    state = store.get_run_state("run_abc")

    assert state["state"] == "running"
    assert state["runId"] == "run_abc"
    assert state["threadId"] == "thread_01"
    assert state["eventSeq"] == 1
    assert state["cursor"]["eventSeq"] == 1
    assert state["cursor"]["state"] == "running"
    assert started_event["eventSeq"] == 1
    assert started_event["eventType"] == "run_started"


def test_event_sequence_increments_monotonic():
    store = RunStateStore()
    store.create_run(run_id="run_seq", thread_id="thread_seq", workflow_id="support_ticket_classifier_v1")
    node_started = store.append_event("run_seq", "node_started", {"nodeId": "intent_parser"})
    completed = store.append_event("run_seq", "run_completed", {"output": "ok"})
    state = store.get_run_state("run_seq")

    assert node_started["eventSeq"] == 2
    assert completed["eventSeq"] == 3
    assert state["state"] == "completed"
    assert state["eventSeq"] == 3
    assert state["cursor"]["eventSeq"] == 3


def test_terminal_state_ignores_late_events():
    store = RunStateStore()
    store.create_run(run_id="run_done", thread_id="thread_done")
    completed = store.append_event("run_done", "run_completed", {"status": "ok"})
    ignored = store.append_event("run_done", "node_started", {"nodeId": "late_node"})
    state = store.get_run_state("run_done")

    assert completed["eventSeq"] == 2
    assert ignored is None
    assert state["eventSeq"] == 2


def test_invalid_transition_raises():
    store = RunStateStore()
    store.create_run(run_id="run_invalid", thread_id="thread_invalid")
    run = store._runs["run_invalid"]
    run["state"] = "queued"

    try:
        store.append_event("run_invalid", "run_completed")
    except Exception as error:
        assert type(error).__name__ == "InvalidRunTransitionError"
    else:
        raise AssertionError("expected InvalidRunTransitionError")


def test_replay_from_seq_filtering():
    store = RunStateStore()
    store.create_run(run_id="run_replay", thread_id="thread_replay")
    store.append_event("run_replay", "node_started", {"nodeId": "intent_parser"})
    store.append_event("run_replay", "node_completed", {"nodeId": "intent_parser"})
    events = store.list_events("run_replay", from_seq=2)

    assert len(events) == 1
    assert events[0]["eventSeq"] == 3


def test_last_event_id_replay_filtering():
    store = RunStateStore()
    store.create_run(run_id="run_replay2", thread_id="thread_replay2")
    node_started = store.append_event("run_replay2", "node_started", {"nodeId": "intent_parser"})
    store.append_event("run_replay2", "node_completed", {"nodeId": "intent_parser"})
    events = store.list_events("run_replay2", last_event_id=node_started["eventId"])

    assert len(events) == 1
    assert events[0]["eventSeq"] == 3


def test_idempotent_event_id_reuses_existing_event():
    store = RunStateStore()
    store.create_run(run_id="run_dup", thread_id="thread_dup")
    first = store.append_event("run_dup", "node_started", {"nodeId": "intent_parser"}, {"eventId": "custom-id-1"})
    second = store.append_event("run_dup", "node_started", {"nodeId": "intent_parser"}, {"eventId": "custom-id-1"})
    state = store.get_run_state("run_dup")

    assert first["eventId"] == second["eventId"]
    assert first["eventSeq"] == second["eventSeq"]
    assert state["eventSeq"] == 2


def test_missing_run_raises_run_not_found():
    store = RunStateStore()
    try:
        store.get_run_state("missing")
    except Exception as error:
        assert type(error).__name__ == "RunNotFoundError"
        assert str(error) == "requested run was not found"
    else:
        raise AssertionError("expected RunNotFoundError")


def test_custom_checkpoint_is_preserved():
    store = RunStateStore()
    checkpoint = {"state": "custom", "eventSeq": 999}
    store.create_run(run_id="run_cp", thread_id="thread_cp")
    event = store.append_event(
        "run_cp",
        "node_started",
        {"nodeId": "intent_parser"},
        {"checkpoint": checkpoint},
    )
    assert event["checkpoint"] == checkpoint


def test_non_object_checkpoint_raises():
    store = RunStateStore()
    store.create_run(run_id="run_cp_invalid", thread_id="thread_cp_invalid")
    try:
        store.append_event("run_cp_invalid", "node_started", {"nodeId": "intent_parser"}, {"checkpoint": None})
    except Exception as error:
        assert type(error).__name__ == "TypeError"
    else:
        raise AssertionError("expected TypeError")
