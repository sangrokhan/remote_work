from python_backend.app.history_store import RunHistoryStore
from python_backend.app.run_state_store import RunStateStore


def _build_history_store_with_sample():
    store = RunStateStore()
    store.create_run(run_id="run_hist", thread_id="thread_hist", workflow_id="support_ticket_classifier_v1")
    store.append_event("run_hist", "node_started", {"nodeId": "intent_parser"})
    store.append_event("run_hist", "node_token", {"nodeId": "intent_parser", "token": "안녕"})
    store.append_event("run_hist", "node_token", {"nodeId": "intent_parser", "token": "세요"})
    store.append_event("run_hist", "node_completed", {"nodeId": "intent_parser"})
    store.append_event("run_hist", "run_completed", {"status": "ok"})
    return RunHistoryStore(run_store=store)


def _build_history_store_with_failure():
    store = RunStateStore()
    store.create_run(run_id="run_hist_failed", thread_id="thread_hist_failed", workflow_id="support_ticket_classifier_v1")
    store.append_event("run_hist_failed", "node_started", {"nodeId": "intent_parser"})
    store.append_event("run_hist_failed", "node_token", {"nodeId": "intent_parser", "token": "안녕"})
    store.append_event("run_hist_failed", "run_failed", {
        "failureCode": "MODEL_TIMEOUT",
        "failureCategory": "llm",
        "reason": "upstream timeout",
        "retryable": True,
        "rootCause": "upstream model deadline exceeded",
        "resolutionHints": ["retry once", "check tokens"],
        "evidenceRefs": ["trace-123"],
        "retryInfo": {"attempt": 1},
    })
    return RunHistoryStore(run_store=store)


def _build_history_store_with_minimal_failure():
    store = RunStateStore()
    store.create_run(run_id="run_hist_failed_min", thread_id="thread_hist_failed_min", workflow_id="support_ticket_classifier_v1")
    store.append_event("run_hist_failed_min", "run_failed", {"reason": "validation rejected"})
    return RunHistoryStore(run_store=store)


def test_history_builds_nodes_and_final_state():
    history_store = _build_history_store_with_sample()
    history = history_store.get_history("run_hist")

    node = history["nodes"]["intent_parser"]
    assert node["state"] == "completed"
    assert node["startSeq"] == 2
    assert node["endSeq"] == 5
    assert node["lastSeq"] == 5
    assert node["tokenCount"] == 2
    assert history["finalState"]["state"] == "completed"
    assert history["cursor"]["schemaVersion"] == "1.0.0"
    assert history["failureContext"] is None


def test_history_failure_run_populates_failure_context():
    history_store = _build_history_store_with_failure()
    history = history_store.get_history("run_hist_failed")

    assert history["finalState"]["state"] == "failed"
    assert history["failureContext"] is not None
    assert history["failureContext"]["schemaVersion"] == "1.0.0"
    assert history["failureContext"]["eventType"] == "run_failed"
    assert history["failureContext"]["failureCode"] == "MODEL_TIMEOUT"
    assert history["failureContext"]["failureCategory"] == "llm"
    assert history["failureContext"]["rootCause"] == "upstream model deadline exceeded"
    assert history["failureContext"]["resolutionHints"] == ["retry once", "check tokens"]
    assert history["failureContext"]["retryable"] is True
    assert history["failureContext"]["errorAt"]
    assert history["failureContext"]["evidenceRefs"] == ["trace-123"]
    assert history["failureContext"]["retryInfo"] == {"attempt": 1}


def test_failure_context_default_resolution_hints_when_not_set():
    history_store = _build_history_store_with_minimal_failure()
    history = history_store.get_history("run_hist_failed_min")

    assert history["failureContext"]["failureCode"] == "UNKNOWN"
    assert history["failureContext"]["failureCategory"] == "unknown"
    assert history["failureContext"]["resolutionHints"] == [
        "Collect request/response trace and retry once",
        "Escalate to on-call with runId and eventId",
    ]
    assert history["failureContext"]["retryable"] is None


def test_pagination_limit_next_cursor():
    history_store = _build_history_store_with_sample()
    history = history_store.get_history("run_hist", limit=3)

    assert history["count"] == 3
    assert history["pagination"]["hasMore"] is True
    assert history["pagination"]["nextCursor"]["eventSeq"] == 3
    assert history["pagination"]["nextCursor"]["state"] == "running"

    next_page = history_store.get_history(
        "run_hist",
        from_seq=history["pagination"]["nextCursor"]["eventSeq"],
        limit=10,
    )
    assert next_page["count"] == 3


def test_last_event_id_filter_starts_after_last_event():
    history_store = _build_history_store_with_sample()
    full = history_store.get_history("run_hist", limit=10)
    last_seq_4 = full["events"][3]["eventId"]
    filtered = history_store.get_history("run_hist", last_event_id=last_seq_4)

    assert filtered["count"] == 2
    assert filtered["events"][0]["eventSeq"] == 5


def test_node_id_filter_returns_only_node_events():
    history_store = _build_history_store_with_sample()
    node_events = history_store.get_history("run_hist", node_id="intent_parser", limit=10)

    assert all(event["nodeId"] == "intent_parser" for event in node_events["events"])
    assert len(node_events["events"]) == 4


def test_node_id_filter_returns_only_matching_nodes_map():
    history_store = _build_history_store_with_sample()
    node_events = history_store.get_history("run_hist", node_id="intent_parser")

    assert list(node_events["nodes"].keys()) == ["intent_parser"]
    assert len(node_events["nodes"]) == 1


def test_limit_must_be_positive_integer():
    history_store = _build_history_store_with_sample()
    try:
        history_store.get_history("run_hist", limit="abc")
        raise AssertionError("expected TypeError")
    except TypeError:
        pass


def test_last_event_id_must_be_non_empty_string():
    history_store = _build_history_store_with_sample()
    try:
        history_store.get_history("run_hist", last_event_id=" ")
        raise AssertionError("expected TypeError")
    except TypeError:
        pass


def test_from_seq_and_last_event_id_are_mutually_exclusive():
    history_store = _build_history_store_with_sample()
    run_state = history_store.get_history("run_hist")
    some_id = run_state["events"][0]["eventId"]

    try:
        history_store.get_history("run_hist", from_seq=1, last_event_id=some_id)
        raise AssertionError("expected TypeError")
    except TypeError:
        pass
