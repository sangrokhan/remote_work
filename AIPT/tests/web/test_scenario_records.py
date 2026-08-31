"""records/<name>.json (aipt.backends.mock.records.RECORD_DIR) support in
the "Record" input mode -- Sangrok's 2026-08-31 report: picking "Record" in
the web UI only ever saw data/public_ai_records/ (captured public_ai runs),
never the hand-authored records/perf.json (or records/smoke.json) fixtures
that actually ship with the repo. Two things had to change for that to work:

1. GET /api/config's "public_ai_records" list (the dropdown's option list,
   routes_config.public_ai_record_names()) must union both directories.
2. routes_run._load_record_scenario(record_id) must resolve a name from
   either directory -- data/public_ai_records/<id>.json first (a captured
   real exchange, byte-pattern-only replay), then records/<id>.json (a
   hand-authored scenario, replayed verbatim).

Behavioural contract per Sangrok's decision: Input=Record only ever changes
what *question* text gets sent -- every backend (public_ai/local_llm/mock)
sends the real question to get a real answer. The recorded/authored
*answer* text is only ever actually served back by the Mock backend (no
real model to ask); public_ai/local_llm always overwrite it with whatever
the live call returns. See _resolve_turns()/_build_backend() docstrings.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from aipt.backends.mock import records as mock_records
from aipt.web import routes_run
from aipt.web import store as run_store
from aipt.web.app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(routes_run.PUBLIC_AI_RECORDS_DIR_ENV, str(tmp_path / "public_ai_records"))
    monkeypatch.setenv(run_store.RUN_STORE_DIR_ENV, str(tmp_path / "runs"))
    # aipt.backends.mock.records.RECORD_DIR is a module-level Path computed
    # from __file__, not env-driven like the other two -- patch it directly
    # so this test never touches the real repo-root records/ directory.
    scenario_dir = tmp_path / "records"
    monkeypatch.setattr(mock_records, "RECORD_DIR", scenario_dir)
    monkeypatch.setattr(routes_run, "mock_records", mock_records)
    run_store.clear()
    app = create_app()
    with TestClient(app) as c:
        yield c, scenario_dir
    run_store.clear()


def _write_scenario(scenario_dir, name: str, *, steps=None, turns=None, system=None):
    scenario_dir.mkdir(parents=True, exist_ok=True)
    doc = {"name": name}
    if system is not None:
        doc["system"] = system
    if steps is not None:
        doc["steps"] = steps
    if turns is not None:
        doc["turns"] = turns
    (scenario_dir / f"{name}.json").write_text(json.dumps(doc))


def test_config_lists_hand_authored_scenario_records(client):
    c, scenario_dir = client
    _write_scenario(scenario_dir, "perf", system=["sys"], steps=[{"text": "q1", "answer": "a1"}])
    resp = c.get("/api/config")
    assert resp.status_code == 200
    assert "perf" in resp.json()["public_ai_records"]


def test_config_unions_both_record_directories_without_duplicates(client):
    c, scenario_dir = client
    _write_scenario(scenario_dir, "perf", system=["sys"], steps=[{"text": "q1", "answer": "a1"}])
    records_dir = routes_run.public_ai_records_dir()
    records_dir.mkdir(parents=True, exist_ok=True)
    captured_doc = {"schema_version": 1, "system": "s", "steps": [], "turns": []}
    (records_dir / "rec001.json").write_text(json.dumps(captured_doc))
    # same name in both dirs must not appear twice
    (records_dir / "perf.json").write_text(json.dumps(captured_doc))

    names = c.get("/api/config").json()["public_ai_records"]
    assert names == sorted(set(names))
    assert set(names) == {"perf", "rec001"}


def test_mock_run_replays_hand_authored_scenario_record_verbatim(client):
    """Mock backend + Input=Record on a records/<name>.json scenario must
    serve the authored answer text byte-for-byte, not a placeholder --
    unlike a captured public_ai record (which is deliberately hollowed to
    byte-count-only, see mock_replay.from_public_ai_record_doc), a
    hand-authored scenario's answer is real content meant to be replayed."""
    c, scenario_dir = client
    _write_scenario(
        scenario_dir,
        "perf",
        system=["You are ATLAS."],
        steps=[
            {"text": "What is 2+2?", "answer": "4"},
            {"text": "What is the capital of France?", "answer": "Paris"},
        ],
    )
    resp = c.post(
        "/api/run",
        json={
            "backend": "mock", "arm": "record",
            "input_mode": "record", "record_id": "perf",
            "measure": "bytes",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True, body
    turns = body["run"]["turns"]
    assert len(turns) == 2
    assert turns[0]["question"] == "What is 2+2?"
    # The response text mock actually served back must be the real authored
    # answer, not an "xxxx..." byte-count placeholder.
    assert turns[0].get("response_text") == "4" or turns[0].get("response") == "4"


def test_load_record_scenario_prefers_captured_over_hand_authored(client, tmp_path):
    """If the same name exists in both directories, the captured
    data/public_ai_records/ exchange wins (it is the more specific,
    per-run artifact) -- documents current resolution order."""
    c, scenario_dir = client
    _write_scenario(scenario_dir, "dup", system=["hand-authored"], steps=[{"text": "q", "answer": "hand"}])
    records_dir = routes_run.public_ai_records_dir()
    records_dir.mkdir(parents=True, exist_ok=True)
    captured_doc = {
        "schema_version": 1, "system": "captured",
        "steps": [{"text": "q"}],
        "turns": [
            {
                "backend": "public_ai", "engine": "gemini", "arm": "stateless",
                "turn": 0, "phase": "steady", "question": "captured question",
                "measure": "bytes", "request_headers": {}, "request_json": None,
                "response_json": None, "response_text": "captured answer",
                "status": 200, "error": None, "wire_sent": 1, "wire_recv": 1,
                "recorded_at": 0.0,
            },
        ],
    }
    (records_dir / "dup.json").write_text(json.dumps(captured_doc))

    scenario = routes_run._load_record_scenario("dup")
    assert scenario.turns[0].question == "captured question"
