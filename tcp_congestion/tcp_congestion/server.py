"""server: minimal HTTP/1.1 inference-mock server (stdlib only).

Endpoints:
  GET /ping                       → {"ts": <float>}
  GET /health                     → {"status": "ok"}
  GET /inference-mock?delay=<ms>&response_bytes=<n>
                                   → {"tokens": 100, "ts": <float>, ...}
                                    (sleeps delay ms before responding;
                                    response_bytes pads the body to exactly
                                    that many bytes on the wire -- see
                                    _pad_json_to_size)

Serves HTTP/1.1 keep-alive so the client reuses the same TCP connection
across multiple turns — the idle-gap experiment requires this.
"""

from __future__ import annotations

import json
import socketserver
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler


def _pad_json_to_size(base: dict, target_bytes: int) -> bytes:
    """Serialize *base* as JSON, padded with a "pad" field of literal
    filler characters so the encoded body is exactly *target_bytes* long.

    Without this, conversation.mock_response_bytes only ever influenced
    the *simulated* history size tcp_congestion.conversation uses to grow
    later turns' prompts -- the bytes that actually crossed the wire were
    always this fixed ~50-70 byte JSON blob, regardless of what the caller
    asked for. That mismatch is invisible in the JSON result (which never
    reports actual wire bytes) but shows up immediately in a pcap: the
    server's response segment never matches mock_response_bytes. Padding
    the real response body to size closes that gap so what the operator
    sees in Wireshark matches what the experiment configuration says.

    target_bytes <= the base (no-pad) JSON's encoded size is returned as
    the base JSON unpadded -- this floor is unavoidable (can't shrink
    JSON syntax/field names below their own size) and is reported back to
    the caller so it's visible rather than silently ignored.
    """
    encoded = json.dumps(base).encode()
    if target_bytes <= 0 or target_bytes <= len(encoded):
        return encoded
    padded = dict(base)
    # Binary-search-free: adding an N-char pad value grows the encoded JSON
    # by exactly N bytes (ASCII, inside an existing string literal), so one
    # direct computation lands on target_bytes exactly.
    padded["pad"] = ""
    deficit = target_bytes - len(json.dumps(padded).encode())
    padded["pad"] = "x" * max(deficit, 0)
    return json.dumps(padded).encode()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence access log
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/ping":
            self._json(200, {"ts": time.time()})
        elif path == "/health":
            self._json(200, {"status": "ok"})
        elif path == "/inference-mock":
            delay_ms = int(query.get("delay", ["0"])[0])
            response_bytes = int(query.get("response_bytes", ["0"])[0])
            if delay_ms > 0:
                time.sleep(delay_ms / 1000)
            self._json(200, {"tokens": 100, "ts": time.time()},
                       target_bytes=response_bytes)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/inference-mock":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            delay_ms = int(query.get("delay", ["0"])[0])
            response_bytes = int(query.get("response_bytes", ["0"])[0])
            if delay_ms > 0:
                time.sleep(delay_ms / 1000)
            self._json(200, {"tokens": 100, "ts": time.time(),
                              "prompt_bytes": len(body)},
                       target_bytes=response_bytes)
        else:
            self._json(404, {"error": "not found"})

    def _json(self, code: int, body: dict, target_bytes: int = 0) -> None:
        data = _pad_json_to_size(body, target_bytes) if target_bytes else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(data)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host: str = "0.0.0.0", port: int = 8888):
        super().__init__((host, port), _Handler)
        self.host, self.port = self.server_address
