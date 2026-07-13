"""The model list comes from the Developer API catalog, and says which arms each
model can actually serve.

Every arm of the comparison runs on generativelanguage with an API key, so a list
built from Vertex's publisher catalog was answering a question nobody asked -- and
would happily offer a model that cannot build a cache, which silently guts the
`cached` arm.

The catalog reports `supportedGenerationMethods`, and that is enough to decide the
generateContent arms (stateless, nocontext) and the cache arm (createCachedContent).
It says nothing about Interactions -- no interaction method is ever listed -- so
interaction support is not inferable here; that is what the probe is for.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gemini_client

CATALOG = {
    "models": [
        {"name": "models/gemini-3.1-flash-lite",
         "supportedGenerationMethods": ["generateContent", "countTokens", "createCachedContent"]},
        {"name": "models/gemini-3.1-flash-image",
         "supportedGenerationMethods": ["generateContent", "countTokens"]},
        {"name": "models/gemini-3.1-flash-live-preview",
         "supportedGenerationMethods": ["bidiGenerateContent"]},
        {"name": "models/embedding-001",
         "supportedGenerationMethods": ["embedContent"]},
    ]
}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(CATALOG).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _serve(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.delenv("GEMINI_MOCK", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_HOST", f"127.0.0.1:{srv.server_port}")
    monkeypatch.setenv("GEMINI_API_SCHEME", "http")
    gemini_client.reset_session()
    return srv


def _models(monkeypatch):
    srv = _serve(monkeypatch)
    try:
        return gemini_client.list_models()
    finally:
        srv.shutdown()
        gemini_client.reset_session()


def test_source_is_the_developer_api(monkeypatch):
    assert _models(monkeypatch)["source"] == "devapi"


def test_only_gemini_generate_content_models_are_offered(monkeypatch):
    ids = {m["id"] for m in _models(monkeypatch)["models"]}
    assert "gemini-3.1-flash-lite" in ids
    assert "embedding-001" not in ids            # cannot generate
    assert "gemini-3.1-flash-live-preview" not in ids   # bidi only


def test_cache_capability_is_reported(monkeypatch):
    by_id = {m["id"]: m for m in _models(monkeypatch)["models"]}
    assert by_id["gemini-3.1-flash-lite"]["can_cache"] is True
    assert by_id["gemini-3.1-flash-image"]["can_cache"] is False


def test_comparison_ready_needs_the_cache_arm(monkeypatch):
    # Without createCachedContent the `cached` arm has nothing to build, so the
    # model cannot cover the comparison even though it answers fine.
    by_id = {m["id"]: m for m in _models(monkeypatch)["models"]}
    assert by_id["gemini-3.1-flash-lite"]["comparison_ready"] is True
    assert by_id["gemini-3.1-flash-image"]["comparison_ready"] is False


def test_default_is_the_fixed_model(monkeypatch):
    out = _models(monkeypatch)
    assert out["default"] == "gemini-3.1-flash-lite" == gemini_client.DEFAULT_MODEL


def test_comparison_ready_models_sort_first(monkeypatch):
    models = _models(monkeypatch)["models"]
    ready = [i for i, m in enumerate(models) if m["comparison_ready"]]
    not_ready = [i for i, m in enumerate(models) if not m["comparison_ready"]]
    assert not ready or not not_ready or max(ready) < min(not_ready)


def test_no_key_falls_back_without_calling_vertex(monkeypatch):
    monkeypatch.delenv("GEMINI_MOCK", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "")
    out = gemini_client.list_models()
    assert out["source"] == "static"
    assert out["default"] == gemini_client.DEFAULT_MODEL


def test_static_fallback_offers_only_live_models():
    # gemini-2.5-* is retired and 404s for new users; a fallback that offers it
    # hands the operator a model that cannot run at all.
    ids = {m["id"] for m in gemini_client.STATIC_MODELS}
    assert not any(i.startswith("gemini-2.5") for i in ids)
    assert gemini_client.DEFAULT_MODEL in ids
    assert all("comparison_ready" in m for m in gemini_client.STATIC_MODELS)
