"""DevAPI endpoint formats, encoded from the reference docs.

Host `generativelanguage.googleapis.com/v1beta`, auth header `x-goog-api-key`.
These are pure builders — no network — so they pin the exact request shapes the
comparison will send before any live call is made.

Sources: docs/devapi-endpoints.md (generateContent, cachedContents).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gemini_client as gc


def test_generate_url_is_devapi_models_path(monkeypatch):
    monkeypatch.delenv("GEMINI_API_HOST", raising=False)
    assert gc.generate_url("gemini-3.1-flash-lite") == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.1-flash-lite:generateContent"
    )


def test_host_is_overridable_for_tests(monkeypatch):
    monkeypatch.setenv("GEMINI_API_HOST", "127.0.0.1:8899")
    assert gc.generate_url("m").startswith("https://127.0.0.1:8899/v1beta/models/m")


def test_auth_headers_use_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    h = gc.auth_headers()
    assert h["x-goog-api-key"] == "test-key-123"
    assert "Authorization" not in h        # DevAPI takes a key, not a bearer token


def test_cache_base_url_is_devapi(monkeypatch):
    monkeypatch.delenv("GEMINI_API_HOST", raising=False)
    assert gc.cache_base_url() == (
        "https://generativelanguage.googleapis.com/v1beta/cachedContents"
    )


def test_cache_create_body_shape(monkeypatch):
    body = gc.cache_create_body(
        "gemini-3.1-flash-lite",
        contents=[{"role": "user", "parts": [{"text": "history"}]}],
        system_instruction="You are a fixture.",
        ttl_seconds=1800,
    )
    # The cache endpoint wants the model as "models/{id}", unlike generateContent's
    # path which carries the bare id.
    assert body["model"] == "models/gemini-3.1-flash-lite"
    assert body["ttl"] == "1800s"
    assert body["contents"] == [{"role": "user", "parts": [{"text": "history"}]}]
    # systemInstruction is a Content object, not a bare string.
    assert body["systemInstruction"] == {"parts": [{"text": "You are a fixture."}]}


def test_cache_create_body_omits_empty_system_instruction():
    body = gc.cache_create_body("m", contents=[], system_instruction="", ttl_seconds=60)
    assert "systemInstruction" not in body


def test_ready_needs_api_key_when_not_mock(monkeypatch):
    monkeypatch.delenv("GEMINI_MOCK", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    ok, reason = gc.ready()
    assert ok is False
    assert "GEMINI_API_KEY" in reason


def test_ready_ok_with_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_MOCK", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    ok, _ = gc.ready()
    assert ok is True
