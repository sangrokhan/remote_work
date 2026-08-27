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
