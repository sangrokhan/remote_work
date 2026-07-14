"""The dual pass, against a local server that answers both ways.

The thing being tested is not "does a POST work". It is that `both` sends exactly two
requests and takes each half of the answer from the pass entitled to give it: bytes and
the response body from the blocking call, the five marks from the streamed one. Get the
merge backwards and the run still produces a chart -- one whose byte column is SSE
framing and whose latency column is a blocking wait. Nothing about it would look wrong.

The stub deliberately makes the two passes distinguishable: the streamed body is padded
with obfuscation junk (as OpenAI's really is), so a merge that took bytes from the wrong
pass shows up as a byte count that is far too large rather than as a subtle skew.
"""

from __future__ import annotations

import http.server
import json
import threading
import time

from core import call
from core.call import Exchange, send

_HITS: list = []

_ANSWER = "Paris is the capital."
_PAD = "x" * 400          # stands in for include_obfuscation padding on the SSE frames


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
            time.sleep(0.02)          # real gaps, so the marks separate
            chunk = f"data: {json.dumps(ev)}\n\n".encode("utf-8")
            self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
            self.wfile.flush()
        # The stream stays open past the last token, as a stored interaction does while
        # the server persists it. That gap is what store_tail measures.
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
    """The provider's answer extractor. It reads a streamed event and a whole blocking
    body alike -- core hands it both, and reasoning parts are never the answer."""
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
        # Headers and request line are on the wire too, so the count exceeds the body.
        assert ex.wire_sent > ex.req_payload_bytes
        assert ex.wire_recv > ex.resp_payload_bytes > 0
        # Exactly one request, and it carried the blocking body.
        assert len(_HITS) == 1
        assert _HITS[0]["auth"] == "Bearer test-key"
        assert "stream" not in _HITS[0]["body"]
        # The two marks the socket can vouch for are honest; the three that need a
        # stream are absent, and `measure` on the record is what says so.
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
        assert ex.text == _ANSWER          # the reasoning delta is not in the answer
        assert len(_HITS) == 1
        assert _HITS[0]["path"] == "/stream"
        assert _HITS[0]["body"]["stream"] is True
        # The five marks, in the only order they can honestly appear in.
        assert 0 <= ex.req_sent_ms <= ex.ttfb_ms <= ex.ttft_ms <= ex.ttlt_ms <= ex.turn_end_ms
        # The server held the stream open ~50 ms after the last token. A blocking client
        # waits that out; this is the number that makes stored interactions expensive.
        assert ex.turn_end_ms - ex.ttlt_ms >= 40
        # The events, put back into the body a blocking call would have returned --
        # including the reasoning part, which a client-side history has to echo back.
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
        # Two requests, in order, one per pass. `both` is what doubles the bill.
        assert len(_HITS) == 2
        assert [h["path"] for h in _HITS] == ["/blocking", "/stream"]

        # Bytes come from the blocking pass. The streamed frames carried 4 x 400 bytes of
        # padding, so a merge that took the streamed count would be far larger than this.
        assert ex.wire_recv < 4 * len(_PAD)
        assert ex.resp_payload_bytes == len(
            json.dumps({"parts": [{"text": _ANSWER}],
                        "usage": {"input_tokens": 11, "output_tokens": 7}}))
        # The response body is the blocking one -- no reasoning part, because a blocking
        # response is what the caller echoes and what the tokens were counted from.
        assert ex.response["usage"]["input_tokens"] == 11
        assert ex.request_json == json.dumps(BODY)     # the request recorded is the real one

        # Marks come from the streamed pass, so all five are populated and ordered.
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
        # It failed at some point in time, not instantly. Zero marks would chart this
        # broken turn as the fastest one in the run.
        assert ex.ttft_ms == ex.turn_end_ms == ex.elapsed_ms
        assert ex.ttlt_ms == ex.turn_end_ms
        assert ex.wire_sent > 0            # it did go out; that is what makes it a failure
    finally:
        srv.shutdown()


def test_an_unreachable_host_comes_back_as_an_error_not_an_exception():
    # Port 1 on loopback: nothing listens, and the connection is refused immediately.
    ex = send("http://127.0.0.1:1/blocking", HEADERS, BODY,
              measure=call.LATENCY, text_of=_text_of)

    assert isinstance(ex, Exchange)
    assert ex.error.startswith("request_failed:")
    assert ex.status == 0
    assert ex.ttft_ms == ex.turn_end_ms      # pinned, not zero
    assert ex.text == ""


def test_an_unknown_measure_is_named_rather_than_guessed():
    ex = send("http://127.0.0.1:1/", HEADERS, BODY,
              measure="fast", text_of=_text_of)
    assert ex.error.startswith("bad_measure:")
    assert ex.wire_sent == 0                 # and nothing went out
