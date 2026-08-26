"""Fake OpenAI-compatible chat/completions HTTP server, for testing
engine_adapter/gateway/LocalLLMBackend without a real llama.cpp/vLLM
process (task instruction: mock the HTTP server for non-live coverage).
"""

from __future__ import annotations

import http.server
import json
import threading


class FakeOpenAICompatHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    #: Class-level so the test can inspect requests the server received.
    hits: list = []
    #: Overridable per test: (status, body_dict) or a callable(body) -> (status, body_dict)
    response_factory = None

    def _read_body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n).decode("utf-8") if n else "{}"
        return json.loads(raw or "{}")

    def do_POST(self):
        body = self._read_body()
        type(self).hits.append({
            "path": self.path,
            "body": body,
            "headers": dict(self.headers.items()),
        })
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return

        factory = type(self).response_factory
        if factory is None:
            status, resp = 200, _default_response(body)
        elif callable(factory):
            status, resp = factory(body)
        else:
            status, resp = factory  # type: ignore[misc]

        self._send(status, resp)

    def _send(self, status, payload):
        if isinstance(payload, dict):
            payload = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


def _default_response(body: dict) -> dict:
    last_user = ""
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "user":
            last_user = m.get("content") or ""
            break
    answer = f"echo: {last_user}"
    return {
        "id": "chatcmpl-fake-1",
        "object": "chat.completion",
        "model": body.get("model", "local-model"),
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": answer},
             "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": max(1, len(last_user.split())),
            "completion_tokens": max(1, len(answer.split())),
            "total_tokens": max(2, len(last_user.split()) + len(answer.split())),
        },
    }


def start_fake_server(response_factory=None) -> http.server.ThreadingHTTPServer:
    FakeOpenAICompatHandler.hits = []
    FakeOpenAICompatHandler.response_factory = response_factory
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAICompatHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def base_url(srv) -> str:
    host, port = srv.server_address
    return f"http://{host}:{port}"
