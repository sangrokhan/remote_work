"""The dual pass, against a local server that answers both ways.

Ported from ``token_traffic/tests/test_call.py`` (DESIGN.md 5, A2) onto
``aipt.backends.public_ai._call``.
"""

from __future__ import annotations

import http.server
import json
import threading
import time

from aipt.backends.public_ai import _call as call
from aipt.backends.public_ai._call import Exchange, send

_HITS: list = []

_ANSWER = "Paris is the capital."
_PAD = "x" * 400


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _read_body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n).decode("utf-8")
        _HITS.append({"path": self.path, "body": json.loads(raw),
                      "auth": self.headers.get("Authorization")})
        return json.loads(raw)

    def do_POST(self):
        body = self._read_body()
        if self.path == "/fail":
            self._send(500, b'{"error":"boom"}', "application/json")
            return
        if self.path == "/stream":
            self._stream(body)
            return
        self._blocking(body)

    def _blocking(self, body):
        payload = json.dumps({
            "parts": [{"text": _ANSWER}],
            "usage": {"input_tokens": 11, "output_tokens": 7},
        }).encode("utf-8")
        self._send(200, payload, "application/json")

    def _stream(self, body):
        events = [
            {"parts": [{"text": "thinking", "thought": True}], "pad": _PAD},
            {"parts": [{"text": "Paris is "}], "pad": _PAD},
            {"parts": [{"text": "the capital."}], "pad": _PAD},
            {"usage": {"input_tokens": 11, "output_tokens": 7}, "pad": _PAD},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for ev in events:
            time.sleep(0.02)
            chunk = f"data: {json.dumps(ev)}\n\n".encode("utf-8")
            self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
            self.wfile.flush()
        time.sleep(0.05)
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _send(self, status, payload, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


def _server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _HITS.clear()
    return srv


def _base(srv) -> str:
    host, port = srv.server_address
    return f"http://{host}:{port}"


def _text_of(event: dict) -> str:
    out = []
    for part in event.get("parts") or []:
        if part.get("thought"):
            continue
        out.append(part.get("text") or "")
    return "".join(out)


def _rebuild(events: list) -> dict:
    parts, usage = [], {}
    for ev in events:
        parts.extend(ev.get("parts") or [])
        if ev.get("usage"):
            usage = ev["usage"]
    return {"parts": parts, "usage": usage}


BODY = {"contents": [{"role": "user", "text": "What is the capital of France?"}]}
STREAM_BODY = {**BODY, "stream": True}
HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer test-key"}


def test_bytes_pass_counts_the_socket_and_leaves_the_answer_marks_alone():
    srv = _server()
    try:
        ex = send(f"{_base(srv)}/blocking", HEADERS, BODY,
                  measure=call.BYTES, text_of=_text_of)

        assert ex.error == ""
        assert ex.status == 200
        assert ex.text == _ANSWER
        assert ex.response["usage"]["input_tokens"] == 11
        assert ex.wire_sent > ex.req_payload_bytes
        assert ex.wire_recv > ex.resp_payload_bytes > 0
        assert len(_HITS) == 1
        assert _HITS[0]["auth"] == "Bearer test-key"
        assert "stream" not in _HITS[0]["body"]
        assert ex.req_sent_ms >= 0
        assert ex.turn_end_ms == ex.elapsed_ms
        assert ex.ttfb_ms == 0 and ex.ttft_ms == 0 and ex.ttlt_ms == 0
    finally:
        srv.shutdown()


def test_latency_pass_produces_ordered_marks_and_a_rebuilt_body():
    srv = _server()
    try:
        ex = send(f"{_base(srv)}/blocking", HEADERS, BODY,
                  measure=call.LATENCY, text_of=_text_of,
                  stream_body=STREAM_BODY, stream_url=f"{_base(srv)}/stream",
                  rebuild=_rebuild)

        assert ex.error == ""
        assert ex.text == _ANSWER
        assert len(_HITS) == 1
        assert _HITS[0]["path"] == "/stream"
        assert _HITS[0]["body"]["stream"] is True
        assert 0 <= ex.req_sent_ms <= ex.ttfb_ms <= ex.ttft_ms <= ex.ttlt_ms <= ex.turn_end_ms
        assert ex.turn_end_ms - ex.ttlt_ms >= 40
        assert ex.response["usage"]["output_tokens"] == 7
        assert any(p.get("thought") for p in ex.response["parts"])
    finally:
        srv.shutdown()


def test_both_issues_two_requests_and_merges_each_half_from_the_right_pass():
    srv = _server()
    try:
        ex = send(f"{_base(srv)}/blocking", HEADERS, BODY,
                  measure=call.BOTH, text_of=_text_of,
                  stream_body=STREAM_BODY, stream_url=f"{_base(srv)}/stream",
                  rebuild=_rebuild)

        assert ex.error == ""
        assert len(_HITS) == 2
        assert [h["path"] for h in _HITS] == ["/blocking", "/stream"]

        assert ex.wire_recv < 4 * len(_PAD)
        assert ex.resp_payload_bytes == len(
            json.dumps({"parts": [{"text": _ANSWER}],
                        "usage": {"input_tokens": 11, "output_tokens": 7}}))
        assert ex.response["usage"]["input_tokens"] == 11
        assert ex.request_json == json.dumps(BODY)

        assert ex.ttft_ms > 0 and ex.ttlt_ms >= ex.ttft_ms
        assert ex.turn_end_ms - ex.ttlt_ms >= 40
        assert ex.text == _ANSWER
    finally:
        srv.shutdown()


def test_a_failed_call_pins_its_marks_instead_of_reporting_zero():
    srv = _server()
    try:
        ex = send(f"{_base(srv)}/fail", HEADERS, BODY,
                  measure=call.BYTES, text_of=_text_of)

        assert ex.status == 500
        assert ex.error.startswith("http_500")
        assert ex.ttft_ms == ex.turn_end_ms == ex.elapsed_ms
        assert ex.ttlt_ms == ex.turn_end_ms
        assert ex.wire_sent > 0
    finally:
        srv.shutdown()


def test_an_unreachable_host_comes_back_as_an_error_not_an_exception():
    ex = send("http://127.0.0.1:1/blocking", HEADERS, BODY,
              measure=call.LATENCY, text_of=_text_of)

    assert isinstance(ex, Exchange)
    assert ex.error.startswith("request_failed:")
    assert ex.status == 0
    assert ex.ttft_ms == ex.turn_end_ms
    assert ex.text == ""


def test_an_unknown_measure_is_named_rather_than_guessed():
    ex = send("http://127.0.0.1:1/", HEADERS, BODY,
              measure="fast", text_of=_text_of)
    assert ex.error.startswith("bad_measure:")
    assert ex.wire_sent == 0
