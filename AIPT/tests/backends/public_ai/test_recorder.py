"""aipt.backends.public_ai.recorder -- fixture capture and secret masking.

New tests (no direct token_traffic ancestor; DESIGN.md 5 B2 is new work).
"""

from __future__ import annotations

import json

from aipt.backends.public_ai import gemini
from aipt.backends.public_ai import recorder


def test_mask_secrets_replaces_known_header_names():
    headers = {"Content-Type": "application/json", "x-goog-api-key": "AIzaSy-real-key"}
    masked = recorder.mask_secrets(headers)
    assert masked["Content-Type"] == "application/json"
    assert masked["x-goog-api-key"] == recorder._MASK


def test_mask_secrets_replaces_bearer_authorization():
    headers = {"Authorization": "Bearer sk-abcdefgh12345"}
    masked = recorder.mask_secrets(headers)
    assert masked["Authorization"] == recorder._MASK


def test_mask_secrets_recurses_into_nested_bodies():
    body = {"model": "gpt", "auth": {"api_key": "sk-nested-secret"}}
    masked = recorder.mask_secrets(body)
    assert masked["model"] == "gpt"
    assert masked["auth"]["api_key"] == recorder._MASK


def test_mask_secrets_json_masks_a_json_string():
    raw = json.dumps({"x-goog-api-key": "AIzaSy-real-key", "contents": []})
    masked = json.loads(recorder.mask_secrets_json(raw))
    assert masked["x-goog-api-key"] == recorder._MASK
    assert masked["contents"] == []


def test_mask_secrets_json_falls_back_on_non_json():
    raw = "Bearer sk-abcdefgh12345"
    assert recorder.mask_secrets_json(raw) == recorder._MASK


def test_record_turn_masks_headers_and_parses_json_bodies():
    ex = gemini.call.Exchange(
        status=200, error="",
        request_json=json.dumps({"contents": []}),
        response_json=json.dumps({"candidates": []}),
        text="hi", wire_sent=10, wire_recv=20,
    )
    rt = recorder.record_turn(
        backend="public_ai", engine="gemini", arm="stateless", turn=1,
        phase="steady", question="q", measure="bytes", exchange=ex,
        headers={"x-goog-api-key": "AIzaSy-secret"},
    )
    as_dict = rt.to_dict()
    assert as_dict["request_headers"]["x-goog-api-key"] == recorder._MASK
    assert as_dict["request_json"] == {"contents": []}
    assert as_dict["response_json"] == {"candidates": []}
    assert as_dict["response_text"] == "hi"
    assert as_dict["wire_sent"] == 10


def test_fixture_writer_produces_perf_json_shaped_output(tmp_path):
    writer = recorder.FixtureWriter(system="sys prompt", steps=["q1", "q2"])
    ex = gemini.call.Exchange(status=200, request_json="{}", response_json="{}",
                               text="ok")
    rt = recorder.record_turn(backend="public_ai", engine="gemini", arm="stateless",
                               turn=1, phase="steady", question="q1", measure="bytes",
                               exchange=ex)
    writer.add(rt)

    out_path = writer.write(tmp_path / "captured.json")
    data = json.loads(out_path.read_text())
    assert data["system"] == "sys prompt"
    assert data["steps"] == [{"text": "q1"}, {"text": "q2"}]
    assert len(data["turns"]) == 1
    assert data["turns"][0]["engine"] == "gemini"
    assert data["turns"][0]["arm"] == "stateless"


def test_fixture_writer_never_writes_a_real_key_to_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFIC_MOCK", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-should-never-hit-disk")

    writer = recorder.FixtureWriter(system="sys", steps=["hello?"])
    backend = gemini.GeminiBackend()
    proxy = recorder.recording_backend(backend, writer, engine="gemini")

    proxy.connect(arm="stateless", model=gemini.DEFAULT_MODEL, system="sys")
    proxy.send_turn(1, "hello?", "bytes")
    proxy.close()

    out_path = writer.write(tmp_path / "captured.json")
    raw_text = out_path.read_text()
    assert "AIzaSy-should-never-hit-disk" not in raw_text
    data = json.loads(raw_text)
    assert len(data["turns"]) == 1
