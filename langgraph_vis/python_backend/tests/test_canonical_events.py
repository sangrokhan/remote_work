from python_backend.app.canonical_events import (
    CANONICAL_SCHEMA_VERSION,
    validate_canonical_event,
    to_canonical_events,
    to_canonical_event,
)


def _base_raw_event(event_id: str, run_id: str, thread_id: str, event_seq: int, event_type: str, payload=None):
    return {
        "eventId": event_id,
        "runId": run_id,
        "threadId": thread_id,
        "eventSeq": event_seq,
        "eventType": event_type,
        "issuedAt": "2026-04-14T00:00:00Z",
        "payload": payload or {},
        "checkpoint": {"state": "running", "eventSeq": event_seq},
    }


def test_to_canonical_event_adds_meta_and_preserves_payload():
    raw = _base_raw_event(
        "evt-001",
        "run_1",
        "thread_1",
        1,
        "run_started",
        {"workflowId": "support_ticket_classifier_v1"},
    )
    canonical = to_canonical_event(raw)

    assert canonical["eventType"] == "run_started"
    assert canonical["canonicalMeta"]["isTerminal"] is False
    assert canonical["canonicalMeta"]["source"] == "stream"
    assert canonical["canonicalMeta"]["replayable"] is True
    assert canonical["canonicalMeta"]["schemaVersion"] == CANONICAL_SCHEMA_VERSION
    assert canonical["payload"]["workflowId"] == "support_ticket_classifier_v1"


def test_awaiting_input_maps_to_run_awaiting_input():
    raw = _base_raw_event("evt-002", "run_1", "thread_1", 2, "awaiting_input")
    canonical = to_canonical_event(raw)

    assert canonical["eventType"] == "run_awaiting_input"
    assert canonical["nodeId"] is None


def test_node_token_is_supported():
    raw = _base_raw_event(
        "evt-003",
        "run_1",
        "thread_1",
        3,
        "node_token",
        {"nodeId": "intent_parser", "token": "x"},
    )
    canonical = to_canonical_event(raw)

    assert canonical["eventType"] == "node_token"
    assert canonical["nodeId"] == "intent_parser"


def test_to_canonical_events_deduplicates_and_sorts_by_event_seq():
    raw = [
        _base_raw_event("evt-004", "run_1", "thread_1", 2, "node_started", {"nodeId": "n1"}),
        _base_raw_event("evt-003", "run_1", "thread_1", 3, "node_completed", {"nodeId": "n1"}),
        _base_raw_event("evt-004", "run_1", "thread_1", 2, "node_started", {"nodeId": "n1"}),
    ]
    canonical = to_canonical_events(raw)

    assert [event["eventSeq"] for event in canonical] == [2, 3]


def test_unsupported_event_type_fails():
    raw = [_base_raw_event("evt-005", "run_1", "thread_1", 1, "unknown_event")]
    try:
        to_canonical_events(raw)
        raise AssertionError("expected TypeError")
    except TypeError:
        pass


def test_terminal_events_marked_terminal():
    terminal_event = to_canonical_event(_base_raw_event("evt-006", "run_1", "thread_1", 6, "run_failed"))
    completed_event = to_canonical_event(_base_raw_event("evt-007", "run_1", "thread_1", 7, "run_completed"))
    cancelled_event = to_canonical_event(_base_raw_event("evt-008", "run_1", "thread_1", 8, "run_cancelled"))

    assert terminal_event["canonicalMeta"]["isTerminal"] is True
    assert completed_event["canonicalMeta"]["isTerminal"] is True
    assert cancelled_event["canonicalMeta"]["isTerminal"] is True


def test_event_recovered_is_not_replayable():
    recovered_event = to_canonical_event(_base_raw_event("evt-009", "run_1", "thread_1", 9, "event_recovered"))

    assert recovered_event["canonicalMeta"]["replayable"] is False


def test_invalid_canonical_source_is_rejected():
    raw = _base_raw_event("evt-010", "run_1", "thread_1", 10, "run_started")
    try:
        to_canonical_event(raw, source="invalid")
        raise AssertionError("expected TypeError")
    except TypeError:
        pass


def test_validate_canonical_event_rejects_invalid_source():
    event = to_canonical_event(
        _base_raw_event(
            "evt-011",
            "run_1",
            "thread_1",
            6,
            "run_started",
        )
    )
    event["canonicalMeta"]["source"] = "invalid"
    try:
        validate_canonical_event(event)
        raise AssertionError("expected TypeError")
    except TypeError:
        pass


def test_validate_canonical_event_rejects_non_object_checkpoint():
    event = to_canonical_event(
        _base_raw_event(
            "evt-012",
            "run_1",
            "thread_1",
            7,
            "run_started",
        )
    )
    event["checkpoint"] = None
    try:
        validate_canonical_event(event)
        raise AssertionError("expected TypeError")
    except TypeError:
        pass
