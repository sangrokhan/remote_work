"""create_cache / delete_cache, ported to the Developer API cachedContents endpoint.

Verified against a localhost server: create POSTs to /v1beta/cachedContents with
the model as `models/{id}`, the system prompt as a Content, and an x-goog-api-key
header; delete DELETEs /v1beta/cachedContents/{id}. No live API.
"""

import http.server
import json
import threading
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gemini_client as gc

_SEEN = {}


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        _SEEN["post_path"] = self.path
        _SEEN["post_key"] = self.headers.get("x-goog-api-key")
        _SEEN["post_body"] = json.loads(self.rfile.read(n) or b"{}")
        payload = json.dumps({
            "name": "cachedContents/abc123",
            "usageMetadata": {"totalTokenCount": 5000},
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_DELETE(self):
        _SEEN["delete_path"] = self.path
        _SEEN["delete_key"] = self.headers.get("x-goog-api-key")
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


def _server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _env(monkeypatch, host):
    monkeypatch.delenv("GEMINI_MOCK", raising=False)
    monkeypatch.setenv("GEMINI_API_SCHEME", "http")
    monkeypatch.setenv("GEMINI_API_HOST", host)
    monkeypatch.setenv("GEMINI_API_KEY", "cache-key")
    monkeypatch.setenv("MIN_CACHE_TOKENS", "0")   # don't pre-reject in the test
    gc.reset_session()


def test_create_cache_posts_devapi_shape(monkeypatch):
    srv = _server()
    host = f"127.0.0.1:{srv.server_address[1]}"
    try:
        _env(monkeypatch, host)
        out = gc.create_cache(
            "gemini-3.1-flash-lite",
            contents=[{"role": "user", "parts": [{"text": "prefix"}]}],
            ttl_seconds=1800,
            system_instruction="You are a fixture.",
        )
        assert out["error"] == ""
        assert out["name"] == "cachedContents/abc123"
        assert out["cached_tokens"] == 5000
        assert _SEEN["post_path"] == "/v1beta/cachedContents"
        assert _SEEN["post_key"] == "cache-key"
        assert _SEEN["post_body"]["model"] == "models/gemini-3.1-flash-lite"
        assert _SEEN["post_body"]["ttl"] == "1800s"
        assert _SEEN["post_body"]["systemInstruction"] == {
            "parts": [{"text": "You are a fixture."}]}
    finally:
        gc.reset_session()
        srv.shutdown()


def test_delete_cache_targets_devapi_resource(monkeypatch):
    srv = _server()
    host = f"127.0.0.1:{srv.server_address[1]}"
    try:
        _env(monkeypatch, host)
        gc.delete_cache("cachedContents/abc123")
        assert _SEEN["delete_path"] == "/v1beta/cachedContents/abc123"
        assert _SEEN["delete_key"] == "cache-key"
    finally:
        gc.reset_session()
        srv.shutdown()


def test_create_cache_reports_wire_and_latency(monkeypatch):
    # The cache-build upload is the whole cost of the setup bucket, so it must be
    # measured, not invisible.
    srv = _server()
    host = f"127.0.0.1:{srv.server_address[1]}"
    try:
        _env(monkeypatch, host)
        out = gc.create_cache(
            "gemini-3.1-flash-lite",
            contents=[{"role": "user", "parts": [{"text": "x" * 500}]}],
            ttl_seconds=1800, system_instruction="sys")
        assert out["error"] == ""
        assert out["wire_sent"] > len(out["request_raw"])   # headers counted
        assert out["wire_recv"] > 0
        assert out["elapsed_ms"] >= 0
        assert out["request_raw"]
    finally:
        gc.reset_session()
        srv.shutdown()


def test_create_cache_below_min_is_skipped_without_network(monkeypatch):
    # A tiny prefix must not even attempt a create.
    monkeypatch.delenv("GEMINI_MOCK", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("MIN_CACHE_TOKENS", "2048")
    out = gc.create_cache("gemini-3.1-flash-lite",
                          contents=[{"role": "user", "parts": [{"text": "hi"}]}],
                          ttl_seconds=60, system_instruction="")
    assert out["name"] is None
    assert "below_min" in out["error"]
