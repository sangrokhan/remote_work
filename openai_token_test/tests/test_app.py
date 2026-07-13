"""The UI must run a real experiment, stream it, persist it, and hand it back.

Everything here goes through the fake OpenAI server, so no key and no spend.
"""

from __future__ import annotations

import json
import zipfile

import pytest

import app as app_mod
import store
import wire
from fake_openai import FakeOpenAI


@pytest.fixture
def client(monkeypatch, tmp_path):
    srv = FakeOpenAI()
    monkeypatch.setenv("OPENAI_BASE_URL", srv.start())
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # keep runs out of the real results/ dir
    monkeypatch.setattr(store, "RUNS_DIR", tmp_path / "runs")
    wire.reset_session()
    app_mod.app.config["TESTING"] = True
    with app_mod.app.test_client() as c:
        yield c
    srv.stop()
    wire.reset_session()


def _events(resp) -> list[tuple[str, dict]]:
    """Parse the SSE body into (event, data) pairs."""
    out = []
    for chunk in resp.get_data(as_text=True).split("\n\n"):
        if not chunk.strip():
            continue
        ev = data = None
        for line in chunk.splitlines():
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if ev:
            out.append((ev, data))
    return out


def test_status_reports_model_arms_and_capture(client):
    d = client.get("/status").get_json()
    assert d["arms"] == ["chat_stateless", "responses_stateless", "responses_stateful"]
    assert d["key_present"] is True
    assert "available" in d["capture"]
    assert "perf" in d["fixtures"]
    assert d["fixtures"]["perf"]["system_chars"] > 1000


def test_run_stream_emits_a_turn_event_per_call_then_done(client):
    resp = client.post("/run/stream", json={"turns": 2, "repeats": 1})
    evs = _events(resp)
    kinds = [e for e, _ in evs]

    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert "error" not in kinds

    turns = [d for e, d in evs if e == "turn"]
    assert len(turns) == 3 * 2, "3 arms x 2 turns"
    assert all(t["upload_bytes"] > 0 for t in turns)

    done = evs[-1][1]
    ratios = done["summary"]["ratios"]
    assert ratios, "a run with all three arms must produce ratios"


def test_the_run_is_persisted_with_its_raw_bodies(client):
    done = _events(client.post("/run/stream", json={"turns": 2, "repeats": 1}))[-1][1]
    exec_id = done["exec_id"]

    doc = client.get(f"/run/{exec_id}").get_json()
    assert doc["config"]["turns"] == 2
    assert len(doc["runs"]) == 3

    # bodies are archived, and the metrics rows do NOT carry them
    assert done["manifest"]["bodies"] > 0
    assert "request_json" not in doc["runs"][0]["turns"][0]

    z = client.get(f"/download/{exec_id}/bodies.zip")
    assert z.status_code == 200
    names = zipfile.ZipFile(__import__("io").BytesIO(z.data)).namelist()
    assert any("chat_stateless_r1_turn01_request.json" in n for n in names)
    assert any("responses_stateful_r1_setup_request.json" in n for n in names)


def test_the_archived_body_is_the_history_the_client_actually_resent(client):
    """The whole claim in one file: turn 2 of the stateless arm contains the system
    prompt AND turn 1, uploaded again. If this ever stops being true, the byte
    numbers are measuring something else."""
    done = _events(client.post("/run/stream", json={"turns": 2, "repeats": 1}))[-1][1]
    d = store.run_dir(done["exec_id"])

    sl = json.loads((d / "bodies" / "chat_stateless_r1_turn02_request.json").read_text())
    roles = [m["role"] for m in sl["messages"]]
    assert roles == ["system", "user", "assistant", "user"]

    sf = json.loads((d / "bodies" / "responses_stateful_r1_turn02_request.json").read_text())
    assert [m["role"] for m in sf["input"]] == ["user"], "stateful resends nothing"
    assert len(json.dumps(sl)) > 10 * len(json.dumps(sf))


def test_downloads_exist_and_traversal_is_refused(client):
    done = _events(client.post("/run/stream", json={"turns": 1, "repeats": 1}))[-1][1]
    exec_id = done["exec_id"]

    for what, ctype in [("run.json", "application/json"),
                        ("summary.csv", "text/csv"),
                        ("charts.png", "image/png")]:
        r = client.get(f"/download/{exec_id}/{what}")
        assert r.status_code == 200, what
        assert ctype in r.headers["Content-Type"]

    assert client.get("/download/../../etc/passwd/run.json").status_code in (307, 404)
    assert client.get(f"/download/{exec_id}/../../../etc/passwd").status_code in (307, 404)
    assert client.get("/download/pcap/../../etc/passwd").status_code in (307, 404)
    assert client.get("/download/pcap/not_a_capture.pcap").status_code == 404


def test_history_lists_the_run(client):
    done = _events(client.post("/run/stream", json={"turns": 1, "repeats": 1}))[-1][1]
    runs = client.get("/history").get_json()["runs"]
    assert runs[0]["exec_id"] == done["exec_id"]
    assert runs[0]["turns"] == 1


def test_bad_exec_id_is_rejected_not_resolved(client):
    assert store.run_dir("../../etc") is None
    assert store.run_dir("nope") is None
    assert client.get("/run/nope").status_code == 404
