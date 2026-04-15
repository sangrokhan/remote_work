from contextlib import contextmanager

from fastapi.testclient import TestClient

from python_backend.app.run_state_api import create_run_state_router
from python_backend.app.run_state_store import RunStateStore
from python_backend.app.run_error_contract import RUN_ERROR_CODES, RUN_ERROR_MESSAGES
from fastapi import FastAPI


def _create_client_with_store(store: RunStateStore):
    app = FastAPI()
    app.include_router(create_run_state_router(store=store))
    return TestClient(app)


def _create_sample_store():
    store = RunStateStore()
    store.create_run(
        run_id="run_api",
        thread_id="thread_api",
        workflow_id="support_ticket_classifier_v1",
    )
    store.append_event("run_api", "node_started", {"nodeId": "intent_parser"})
    store.append_event("run_api", "node_completed", {"nodeId": "intent_parser"})
    return store


def test_create_run_api_generates_run():
    store = RunStateStore()
    with _create_client_with_store(store) as client:
        response = client.post(
            "/api/runs",
            json={
                "runId": "run_new",
                "threadId": "thread_new",
                "workflowId": "support_ticket_classifier_v1",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["runId"] == "run_new"
    assert body["threadId"] == "thread_new"
    assert body["state"]["runId"] == "run_new"
    assert body["state"]["threadId"] == "thread_new"
    assert body["state"]["state"] == "running"


def test_create_run_api_returns_bad_request_for_invalid_payload():
    store = RunStateStore()
    with _create_client_with_store(store) as client:
        response = client.post("/api/runs", data="not-json", headers={"Content-Type": "application/json"})

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"]


def test_append_event_api_accepts_event():
    store = _create_sample_store()
    with _create_client_with_store(store) as client:
        response = client.post(
            "/api/runs/run_api/events",
            json={"eventType": "node_token", "payload": {"nodeId": "intent_parser", "token": "test"}},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["event"]["eventType"] == "node_token"
    assert body["state"]["state"] == "running"


def test_append_event_api_rejects_missing_event_type():
    store = _create_sample_store()
    with _create_client_with_store(store) as client:
        response = client.post("/api/runs/run_api/events", json={"payload": {}})

    assert response.status_code == 400
    assert response.json()["code"] == RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"]


def test_append_event_api_returns_405_for_non_post():
    store = _create_sample_store()
    with _create_client_with_store(store) as client:
        response = client.put("/api/runs/run_api/events", json={"eventType": "node_started"})

    assert response.status_code == 405
    assert response.json()["code"] == RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"]


def test_get_state_returns_cursor_and_state():
    store = _create_sample_store()
    with _create_client_with_store(store) as client:
        response = client.get("/api/runs/run_api/state")

    assert response.status_code == 200
    body = response.json()
    assert body["runId"] == "run_api"
    assert body["threadId"] == "thread_api"
    assert body["state"] == "running"
    assert body["cursor"]["eventSeq"] >= 1
    assert body["cursor"]["state"] == "running"


def test_get_state_unknown_run_returns_not_found():
    store = _create_sample_store()
    with _create_client_with_store(store) as client:
        response = client.get("/api/runs/run_missing/state")

    assert response.status_code == 404
    body = response.json()
    assert body["code"] == RUN_ERROR_CODES["RUN_NOT_FOUND"]


def test_events_support_from_seq():
    store = _create_sample_store()
    with _create_client_with_store(store) as client:
        response = client.get("/api/runs/run_api/events?fromSeq=2")
    assert response.status_code == 200
    body = response.json()
    assert body["runId"] == "run_api"
    assert len(body["events"]) == 1
    assert body["events"][0]["eventSeq"] == 3


def test_events_support_last_event_id():
    store = _create_sample_store()
    state = store.get_run_state("run_api")
    events = store.list_events("run_api")
    node_completed_id = events[2]["eventId"]
    with _create_client_with_store(store) as client:
        response = client.get(f"/api/runs/run_api/events?lastEventId={node_completed_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["events"] == []
    assert body["cursor"]["eventSeq"] == state["eventSeq"]


def test_events_invalid_reconnect_query_returns_400():
    store = _create_sample_store()
    with _create_client_with_store(store) as client:
        response = client.get("/api/runs/run_api/events?fromSeq=abc")
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == RUN_ERROR_CODES["INVALID_RECONNECT_QUERY"]


def test_events_rejects_conflicting_reconnect_cursors():
    store = _create_sample_store()
    with _create_client_with_store(store) as client:
        response = client.get("/api/runs/run_api/events?fromSeq=1&lastEventId=evt-foo")
    assert response.status_code == 400
    assert response.json()["code"] == RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"]


def test_sse_stream_supports_from_seq():
    store = _create_sample_store()
    with _create_client_with_store(store) as client:
        with client.stream("GET", "/api/runs/run_api/events/stream?fromSeq=2") as response:
            chunk = next(response.iter_raw())
            assert response.status_code == 200
            text = chunk.decode("utf-8")
            assert "event: run-event" in text
            assert '"eventSeq": 3' in text
            assert "id: " in text
            assert "text/event-stream" in response.headers["content-type"]


def test_sse_stream_supports_last_event_id():
    store = _create_sample_store()
    events = store.list_events("run_api")
    node_started_id = events[1]["eventId"]
    node_completed_id = events[2]["eventId"]
    with _create_client_with_store(store) as client:
        with client.stream(
            "GET", f"/api/runs/run_api/events/stream?lastEventId={node_started_id}"
        ) as response:
            chunk = next(response.iter_raw())
            assert response.status_code == 200
            text = chunk.decode("utf-8")
            assert '"eventSeq": 3' in text
            assert node_started_id not in text
            assert f"id: {node_completed_id}" in text


def test_stream_invalid_reconnect_query_returns_400():
    store = _create_sample_store()
    with _create_client_with_store(store) as client:
        response = client.get("/api/runs/run_api/events/stream?fromSeq=abc")

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == RUN_ERROR_CODES["INVALID_RECONNECT_QUERY"]


def test_sse_stream_rejects_conflicting_reconnect_cursors():
    store = _create_sample_store()
    with _create_client_with_store(store) as client:
        response = client.get("/api/runs/run_api/events/stream?fromSeq=1&lastEventId=evt-foo")
    assert response.status_code == 400
    assert response.json()["code"] == RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"]


def test_sse_stream_invalid_empty_last_event_id_query_returns_400():
    store = _create_sample_store()
    with _create_client_with_store(store) as client:
        response = client.get("/api/runs/run_api/events/stream?lastEventId=%20")

    assert response.status_code == 400
    assert response.json()["code"] == RUN_ERROR_CODES["INVALID_RECONNECT_QUERY"]


def test_sse_stream_waits_with_heartbeat_when_no_new_events():
    store = _create_sample_store()
    with _create_client_with_store(store) as client:
        with client.stream("GET", "/api/runs/run_api/events/stream?fromSeq=10") as response:
            assert response.status_code == 200
            chunk = next(response.iter_raw())
            assert b"heartbeat" in chunk
            assert "text/event-stream" in response.headers["content-type"]


def test_non_get_methods_return_405():
    store = _create_sample_store()
    with _create_client_with_store(store) as client:
        response = client.post("/api/runs/run_api/state")
    assert response.status_code == 405
    assert response.headers.get("allow") == "GET"
    body = response.json()
    assert body["code"] == RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"]
