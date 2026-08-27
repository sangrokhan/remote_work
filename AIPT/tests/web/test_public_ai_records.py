"""DESIGN.md 4.7.1 -- Public AI 실행 결과 자동 영속 저장 검증.

- mock 백엔드로 /api/run을 돌리면 data/public_ai_records/ 에 아무 파일도
  쓰이지 않아야 한다 (자동 저장은 public_ai 전용).
- public_ai 실경로는 실제 API 키가 없어 이 스위트에서 돌릴 수 없으므로,
  recorder.recording_backend()가 만드는 FixtureWriter가 실제로 디스크에
  JSON을 쓰고 mask_secrets가 적용되는지를 유닛 레벨로 직접 검증한다.
- GET /api/public-ai-records, GET /api/public-ai-records/{exec_id} 라우트를
  빈 디렉터리/파일 있는 경우 모두 확인한다.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from aipt.backends.public_ai import recorder as public_ai_recorder
from aipt.web import routes_run
from aipt.web import store as run_store
from aipt.web.app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    records_dir = tmp_path / "public_ai_records"
    monkeypatch.setenv(routes_run.PUBLIC_AI_RECORDS_DIR_ENV, str(records_dir))
    monkeypatch.setenv(run_store.RUN_STORE_DIR_ENV, str(tmp_path / "runs"))
    run_store.clear()
    app = create_app()
    with TestClient(app) as c:
        yield c, records_dir
    run_store.clear()


def test_mock_run_writes_nothing_to_public_ai_records_dir(client):
    c, records_dir = client
    resp = c.post(
        "/api/run",
        json={
            "backend": "mock",
            "arm": "dummy",
            "input_mode": "dummy",
            "num_turns": 1,
            "turn_user_msg_bytes": 10,
            "measure": "bytes",
            "mock_response_bytes": 16,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    run = body["run"]
    # mock runs never touch the record_saved/record_path fields at all --
    # the recording wrapper isn't even constructed for non-public_ai backends.
    assert "record_saved" not in run
    assert "record_path" not in run
    # ... and no directory/file gets created as a side effect.
    assert not records_dir.exists() or list(records_dir.glob("*.json")) == []


def test_list_public_ai_records_empty_dir(client):
    c, records_dir = client
    resp = c.get("/api/public-ai-records")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_and_get_public_ai_records_with_files(client):
    c, records_dir = client
    records_dir.mkdir(parents=True, exist_ok=True)
    doc = {"schema_version": 1, "system": "s", "steps": [], "turns": []}
    (records_dir / "abc123.json").write_text(json.dumps(doc))

    list_resp = c.get("/api/public-ai-records")
    assert list_resp.status_code == 200
    entries = list_resp.json()
    assert len(entries) == 1
    assert entries[0]["exec_id"] == "abc123"
    assert entries[0]["size_bytes"] > 0
    assert "timestamp" in entries[0]

    detail_resp = c.get("/api/public-ai-records/abc123")
    assert detail_resp.status_code == 200
    assert detail_resp.json() == doc

    missing_resp = c.get("/api/public-ai-records/does-not-exist")
    assert missing_resp.status_code == 404


def test_get_public_ai_record_rejects_path_traversal(client):
    c, records_dir = client
    resp = c.get("/api/public-ai-records/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code == 404


def test_mock_run_replays_a_public_ai_record(client):
    """input_mode='record', record_id='<exec_id>' -- the mock backend
    actually serves the captured record's answer bytes (byte-pattern-only
    placeholder, per aipt.backends.mock.replay), and the question text
    sent is the record's real captured question."""
    c, records_dir = client
    records_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": 1,
        "system": "sys prompt",
        "steps": [{"text": "hi"}],
        "turns": [
            {
                "backend": "public_ai", "engine": "gemini", "arm": "stateless",
                "turn": 0, "phase": "steady", "question": "what is TCP?",
                "measure": "bytes", "request_headers": {}, "request_json": None,
                "response_json": None, "response_text": "TCP is a protocol.",
                "status": 200, "error": None, "wire_sent": 12, "wire_recv": 18,
                "recorded_at": 0.0,
            },
        ],
    }
    (records_dir / "rec001.json").write_text(json.dumps(doc))

    resp = c.post(
        "/api/run",
        json={
            "backend": "mock", "arm": "record",
            "input_mode": "record", "record_id": "rec001",
            "measure": "bytes",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True, body
    turns = body["run"]["turns"]
    assert len(turns) == 1
    assert turns[0]["question"] == "what is TCP?"


# ---------------------------------------------------------------------------
# Unit-level: recorder.recording_backend actually writes a masked JSON file.
# No live API key is available in this environment, so the wrapped "backend"
# below is a minimal fake that mimics the connect/send_turn/close surface
# GeminiBackend/OpenAIBackend expose -- exactly the duck-typed contract
# recording_backend()'s docstring describes.
# ---------------------------------------------------------------------------


class _FakeExchange:
    def __init__(self):
        self.request_json = json.dumps(
            {"model": "x", "api_key": "sk-should-be-masked-1234567890"}
        )
        self.response_json = json.dumps({"text": "hi", "authorization": "Bearer secret-token"})
        self.text = "hi"
        self.status = 200
        self.error = None
        self.wire_sent = 10
        self.wire_recv = 20


class _FakeBackend:
    NAME = "public_ai"
    DEFAULT_MODEL = "fake-model"
    ARMS = ("fake_arm",)
    HEADLINE_ARMS = ("fake_arm",)
    transport = "http1"

    def ready(self):
        return True, ""

    def api_host(self):
        return "fake.example.com"

    def connect(self, arm, model, system):
        pass

    def send_turn(self, turn, question, measure, on_progress=None):
        return _FakeExchange()

    def close(self):
        pass


def test_recording_backend_writes_masked_json_to_disk(tmp_path):
    writer = public_ai_recorder.FixtureWriter(system="sys prompt", steps=["hi"])
    proxy = public_ai_recorder.recording_backend(_FakeBackend(), writer, engine="gemini")

    proxy.connect("fake_arm", "fake-model", "sys prompt")
    exchange = proxy.send_turn(0, "hi", "bytes")
    assert exchange.status == 200
    proxy.close()

    out_path = tmp_path / "exec123.json"
    written = writer.write(out_path)
    assert written == out_path
    assert out_path.is_file()

    doc = json.loads(out_path.read_text())
    assert doc["schema_version"] == public_ai_recorder.SCHEMA_VERSION
    assert doc["system"] == "sys prompt"
    assert len(doc["turns"]) == 1
    turn = doc["turns"][0]
    assert turn["backend"] == "public_ai"
    assert turn["engine"] == "gemini"
    assert turn["arm"] == "fake_arm"
    # mask_secrets must have scrubbed the api_key/authorization values before
    # anything touched disk.
    assert turn["request_json"]["api_key"] == "***MASKED***"
    assert turn["response_json"]["authorization"] == "***MASKED***"
    raw = out_path.read_text()
    assert "sk-should-be-masked" not in raw
    assert "secret-token" not in raw
