"""call_gemini, ported to the Developer API, exercised against a local server.

No live API: a localhost HTTP server mimics generateContent. The test asserts the
ported call (a) targets the DevAPI models path with an x-goog-api-key header,
(b) parses usageMetadata, (c) reports wire bytes from the socket counter (headers
included, so more than the JSON payload) rather than the old payload fallback, and
(d) records a non-negative elapsed_ms.
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
        _SEEN["path"] = self.path
        _SEEN["api_key"] = self.headers.get("x-goog-api-key")
        _SEEN["body"] = self.rfile.read(n).decode("utf-8")
        payload = json.dumps({
            "candidates": [{"content": {"role": "model",
                                        "parts": [{"text": "hello from stub"}]}}],
            "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7,
                              "cachedContentTokenCount": 5, "totalTokenCount": 23},
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


def _server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_call_gemini_hits_devapi_and_reports_real_metrics(monkeypatch):
    srv = _server()
    host = f"127.0.0.1:{srv.server_address[1]}"
    try:
        monkeypatch.delenv("GEMINI_MOCK", raising=False)
        monkeypatch.setenv("GEMINI_API_SCHEME", "http")   # local server has no TLS
        monkeypatch.setenv("GEMINI_API_HOST", host)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-xyz")
        gc.reset_session()

        contents = [{"role": "user", "parts": [{"text": "hi there"}]}]
        r = gc.call_gemini("gemini-3.1-flash-lite", contents, mode="stateless", turn=1)

        assert r.error == ""
        assert r.response_text == "hello from stub"
        assert r.prompt_tokens == 11
        assert r.resp_tokens == 7
        assert r.cached_tokens == 5
        assert r.total_tokens == 23
        # Wire bytes come from the socket counter, headers included, so they exceed
        # the JSON payload size (the old fallback would have equalled it).
        assert r.wire_sent > r.req_payload_bytes
        assert r.wire_recv > 0
        assert r.elapsed_ms >= 0
        # The request went where and how the DevAPI expects.
        assert _SEEN["path"] == "/v1beta/models/gemini-3.1-flash-lite:generateContent"
        assert _SEEN["api_key"] == "test-key-xyz"
    finally:
        gc.reset_session()
        srv.shutdown()
