"""Smoke tests for aipt.web (DESIGN.md 5 / 10): the FastAPI app boots,
landing page renders, /api/config reports the 3-backend registry, a mock
run round-trips through /api/run and /api/runs, and local_llm surfaces as
501 rather than a raw traceback.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from aipt.web import routes_run
from aipt.web import store as run_store
from aipt.web.app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(routes_run.PUBLIC_AI_RECORDS_DIR_ENV, str(tmp_path / "public_ai_records"))
    monkeypatch.setenv(run_store.RUN_STORE_DIR_ENV, str(tmp_path / "runs"))
    run_store.clear()
    app = create_app()
    with TestClient(app) as c:
        yield c
    run_store.clear()


def _write_record(records_dir, record_id: str, questions: list[str]) -> None:
    records_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": 1,
        "system": "",
        "steps": [],
        "turns": [
            {
                "backend": "public_ai", "engine": "gemini", "arm": "stateless",
                "turn": i, "phase": "steady", "question": q,
                "measure": "bytes", "request_headers": {}, "request_json": None,
                "response_json": None, "response_text": "ok",
                "status": 200, "error": None, "wire_sent": 0, "wire_recv": 0,
                "recorded_at": 0.0,
            }
            for i, q in enumerate(questions)
        ],
    }
    (records_dir / f"{record_id}.json").write_text(json.dumps(doc))


def test_index_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "AIPT" in resp.text
    assert "Gemini" in resp.text
    assert "ChatGPT" in resp.text
    assert "Local LLM" in resp.text


def test_api_config_lists_all_three_backends(client):
    resp = client.get("/api/config")
    assert resp.status_code == 200
    body = resp.json()
    names = {b["name"] for b in body["backends"]}
    assert names == {"public_ai", "mock", "local_llm"}
    mock = next(b for b in body["backends"] if b["name"] == "mock")
    assert mock["implemented"] is True
    assert mock["ready"] is True
    assert "dummy" in mock["arms"]
    for b in body["backends"]:
        # every backend advertises whether it has a real Backend-protocol
        # class -- never absent, and never a 500 while computing it.
        assert isinstance(b["implemented"], bool)
        assert isinstance(b["ready"], bool)
    assert "congestion_algorithms" in body
    assert "cwnd" in body and "capture" in body


def test_api_run_mock_backend_dummy_mode_round_trips_and_lists(client):
    resp = client.post(
        "/api/run",
        json={
            "backend": "mock",
            "arm": "dummy",
            "input_mode": "dummy",
            "num_turns": 2,
            "turn_user_msg_bytes": 20,
            "system_prompt_bytes": 10,
            "measure": "bytes",
            "mock_response_bytes": 32,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    run = body["run"]
    assert run["backend"] == "mock"
    assert run["mock"] is True
    assert len(run["turns"]) == 2
    exec_id = run["exec_id"]

    # GET /api/runs lists it
    runs_resp = client.get("/api/runs")
    assert runs_resp.status_code == 200
    listed = runs_resp.json()
    assert any(r["exec_id"] == exec_id for r in listed)

    # GET /api/runs/{exec_id} returns the full doc
    detail = client.get(f"/api/runs/{exec_id}")
    assert detail.status_code == 200
    assert detail.json()["run"]["exec_id"] == exec_id

    # CSV endpoints don't blow up
    for path in ("turns.csv", "summary.csv", "cwnd.csv", "cwnd_summary.csv",
                 "packets.csv", "bundle.zip"):
        csv_resp = client.get(f"/api/runs/{exec_id}/{path}")
        assert csv_resp.status_code == 200, f"{path}: {csv_resp.text}"

    # DELETE removes it
    del_resp = client.delete(f"/api/runs/{exec_id}")
    assert del_resp.status_code == 200
    assert client.get(f"/api/runs/{exec_id}").status_code == 404


def test_api_run_mock_backend_record_mode_replays_a_record(client, tmp_path):
    _write_record(tmp_path / "public_ai_records", "rec-smoke", ["what is 2+2?"])
    resp = client.post(
        "/api/run",
        json={
            "backend": "mock",
            "arm": "record",
            "input_mode": "record",
            "record_id": "rec-smoke",
            "measure": "bytes",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True, body
    assert len(body["run"]["turns"]) > 0


def test_api_run_dummy_mode_rejected_for_non_mock_backend(client):
    resp = client.post(
        "/api/run",
        json={"backend": "local_llm", "arm": "chat", "input_mode": "dummy"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is False
    assert "dummy" in body["run"]["error"]


def test_api_run_unknown_backend_is_400(client):
    resp = client.post("/api/run", json={"backend": "nope", "arm": "x"})
    assert resp.status_code == 400


def test_api_run_local_llm_does_not_500(client, tmp_path):
    """local_llm (DESIGN.md 5 B4) may be a NotImplementedError stub (-> 501)
    or, once a parallel work stream lands it, a real backend that simply
    can't reach a live engine in this test environment (-> 200 with
    ok:false, or a connection-error 5xx from the run itself) -- either way
    the route must never leak a raw unhandled traceback (500 with no JSON
    'error' key)."""
    _write_record(tmp_path / "public_ai_records", "rec-smoke", ["hi"])
    resp = client.post(
        "/api/run",
        json={
            "backend": "local_llm", "arm": "chat",
            "input_mode": "record", "record_id": "rec-smoke",
        },
    )
    assert resp.status_code in (200, 501, 502)
    body = resp.json()
    assert "ok" in body
    if not body["ok"]:
        assert body.get("error")


def test_runs_not_found_is_404(client):
    assert client.get("/api/runs/does-not-exist").status_code == 404
    assert client.get("/api/runs/does-not-exist/turns.csv").status_code == 404
    assert client.delete("/api/runs/does-not-exist").status_code == 404


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def test_api_run_stream_mock_backend_emits_start_turn_done(client):
    resp = client.post(
        "/api/run/stream",
        json={
            "backend": "mock",
            "arm": "dummy",
            "input_mode": "dummy",
            "num_turns": 3,
            "turn_user_msg_bytes": 20,
            "system_prompt_bytes": 10,
            "measure": "bytes",
            "mock_response_bytes": 32,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)

    assert events[0]["type"] == "start"
    assert events[0]["total_turns"] == 3
    turn_events = [e for e in events if e["type"] == "turn"]
    assert [e["turn"] for e in turn_events] == [0, 1, 2]
    for e in turn_events:
        assert e["record"]["turn"] in (0, 1, 2)

    assert events[-1]["type"] == "done"
    result = events[-1]["result"]
    assert result["ok"] is True
    assert len(result["turns"]) == 3
    exec_id = result["exec_id"]

    # the streamed run was persisted to run_store exactly like /api/run does
    runs_resp = client.get("/api/runs")
    assert any(r["exec_id"] == exec_id for r in runs_resp.json())


def test_api_run_stream_unknown_backend_emits_error_event_not_400(client):
    resp = client.post("/api/run/stream", json={"backend": "nope", "arm": "x"})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert len(events) == 1
    assert events[0]["type"] == "error"


def test_api_run_stream_local_llm_emits_error_or_done_event(client, tmp_path):
    _write_record(tmp_path / "public_ai_records", "rec-smoke", ["hi"])
    resp = client.post(
        "/api/run/stream",
        json={
            "backend": "local_llm", "arm": "chat",
            "input_mode": "record", "record_id": "rec-smoke",
        },
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert events, "expected at least one SSE event"
    assert events[-1]["type"] in ("error", "done")


def test_api_run_stream_logs_events_to_jsonl(client, tmp_path):
    """Server-side logging structure: every streamed event for a run also
    lands, one JSON line per event, in
    ``<RUN_STORE_DIR>/<exec_id>.stream.jsonl`` -- so the stream can be
    inspected after the fact without any frontend having listened."""
    resp = client.post(
        "/api/run/stream",
        json={
            "backend": "mock",
            "arm": "dummy",
            "input_mode": "dummy",
            "num_turns": 2,
            "turn_user_msg_bytes": 20,
            "system_prompt_bytes": 10,
            "measure": "bytes",
            "mock_response_bytes": 32,
        },
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    exec_id = events[-1]["result"]["exec_id"]

    log_path = tmp_path / "runs" / f"{exec_id}.stream.jsonl"
    assert log_path.is_file(), f"expected stream log at {log_path}"
    lines = [json.loads(l) for l in log_path.read_text().splitlines() if l]
    types = [l["type"] for l in lines]
    assert types == ["start", "turn", "turn", "done"]
    for line in lines:
        assert "logged_at" in line


def test_api_run_stream_unknown_backend_error_not_written_to_disk(client, tmp_path):
    """A pre-backend error (no exec_id yet) has nothing to name a per-run
    log file after -- it must still hit the structured logger but must not
    attempt to write any file."""
    resp = client.post("/api/run/stream", json={"backend": "nope", "arm": "x"})
    assert resp.status_code == 200
    runs_dir = tmp_path / "runs"
    if runs_dir.is_dir():
        assert not any(runs_dir.glob("*.stream.jsonl"))
