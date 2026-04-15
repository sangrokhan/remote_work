from fastapi import FastAPI
from fastapi.testclient import TestClient

from python_backend.app.history_api import create_run_history_router
from python_backend.app.run_state_store import RunStateStore
from python_backend.app.run_error_contract import RUN_ERROR_CODES


def _create_client_with_store(store: RunStateStore):
    app = FastAPI()
    app.include_router(create_run_history_router(store=store))
    return TestClient(app)


def _sample_store():
    store = RunStateStore()
    store.create_run(run_id="run_api_hist", thread_id="thread_api_hist", workflow_id="support_ticket_classifier_v1")
    store.append_event("run_api_hist", "node_started", {"nodeId": "intent_parser"})
    store.append_event("run_api_hist", "node_completed", {"nodeId": "intent_parser"})
    store.append_event("run_api_hist", "run_completed", {"status": "ok"})
    return store


def _sample_failed_store():
    store = RunStateStore()
    store.create_run(run_id="run_api_hist_failed", thread_id="thread_api_hist_failed", workflow_id="support_ticket_classifier_v1")
    store.append_event("run_api_hist_failed", "node_started", {"nodeId": "intent_parser"})
    store.append_event("run_api_hist_failed", "run_failed", {
        "failureCode": "MODEL_TIMEOUT",
        "failureCategory": "llm",
        "reason": "upstream timeout",
        "retryable": True,
        "retryInfo": {"attempt": 1},
        "evidenceRefs": ["trace-1"],
    })
    return store


def test_get_history_returns_nodes_and_failure_context():
    store = _sample_store()
    with _create_client_with_store(store) as client:
        response = client.get("/api/runs/run_api_hist/history")

    assert response.status_code == 200
    body = response.json()
    assert body["runId"] == "run_api_hist"
    assert body["threadId"] == "thread_api_hist"
    assert body["nodes"]["intent_parser"]["state"] == "completed"
    assert body["finalState"]["state"] == "completed"
    assert body["failureContext"] is None


def test_history_supports_limit_and_from_seq():
    store = _sample_store()
    with _create_client_with_store(store) as client:
        response = client.get("/api/runs/run_api_hist/history?limit=2")
        page1 = response.json()

    assert response.status_code == 200
    assert len(page1["events"]) == 2
    assert page1["pagination"]["hasMore"] is True
    next_cursor = page1["pagination"]["nextCursor"]
    assert next_cursor is not None

    with _create_client_with_store(store) as client:
        response = client.get(f"/api/runs/run_api_hist/history?fromSeq={next_cursor['eventSeq']}&limit=2")
        page2 = response.json()

    assert response.status_code == 200
    assert all(event["eventSeq"] > next_cursor["eventSeq"] for event in page2["events"])


def test_history_node_filter_filters_events():
    store = _sample_store()
    with _create_client_with_store(store) as client:
        response = client.get("/api/runs/run_api_hist/history?nodeId=intent_parser")
        body = response.json()

    assert response.status_code == 200
    assert body["count"] == 2
    assert all(event["nodeId"] == "intent_parser" for event in body["events"])
    assert list(body["nodes"].keys()) == ["intent_parser"]


def test_history_invalid_run_id_returns_404():
    store = _sample_store()
    with _create_client_with_store(store) as client:
        response = client.get("/api/runs/not-found/history")
    assert response.status_code == 404


def test_history_supports_last_event_id_header_cursor():
    store = _sample_store()
    state = store.get_run_state("run_api_hist")
    all_events = store.list_events("run_api_hist")
    node_started_event_id = all_events[1]["eventId"]
    with _create_client_with_store(store) as client:
        response = client.get(
            "/api/runs/run_api_hist/history?limit=10",
            headers={"last-event-id": node_started_event_id},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["events"][0]["eventSeq"] == 3
    assert body["events"][1]["eventSeq"] == state["eventSeq"]


def test_history_invalid_limit_returns_invalid_run_payload():
    store = _sample_store()
    with _create_client_with_store(store) as client:
        response = client.get("/api/runs/run_api_hist/history?limit=bad")

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"]


def test_history_invalid_reconnect_query_returns_invalid_reconnect_query():
    store = _sample_store()
    with _create_client_with_store(store) as client:
        response = client.get("/api/runs/run_api_hist/history?fromSeq=bad")

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == RUN_ERROR_CODES["INVALID_RECONNECT_QUERY"]


def test_history_from_seq_and_header_last_event_id_is_invalid():
    store = _sample_store()
    with _create_client_with_store(store) as client:
        response = client.get(
            "/api/runs/run_api_hist/history?fromSeq=1",
            headers={"last-event-id": "some-id"},
        )
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"]


def test_history_failed_run_exposes_failure_context():
    store = _sample_failed_store()
    with _create_client_with_store(store) as client:
        response = client.get("/api/runs/run_api_hist_failed/history")

    assert response.status_code == 200
    body = response.json()
    assert body["finalState"]["state"] == "failed"
    assert body["failureContext"] is not None
    assert body["failureContext"]["schemaVersion"] == "1.0.0"
    assert body["failureContext"]["eventType"] == "run_failed"
    assert body["failureContext"]["failureCode"] == "MODEL_TIMEOUT"
    assert body["failureContext"]["failureCategory"] == "llm"
    assert body["failureContext"]["errorAt"] is not None
    assert isinstance(body["failureContext"]["resolutionHints"], list)
    assert body["failureContext"]["retryable"] is True
    assert body["failureContext"]["retryInfo"] == {"attempt": 1}


def test_history_last_event_id_query_is_invalid_reconnect_query():
    store = _sample_store()
    with _create_client_with_store(store) as client:
        response = client.get("/api/runs/run_api_hist/history?lastEventId= ")

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == RUN_ERROR_CODES["INVALID_RECONNECT_QUERY"]
