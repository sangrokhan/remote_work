"""The counter has to survive keep-alive, and its marks have to be in the right order.

Keep-alive is where a socket counter goes wrong: install it in `connect()`, read it
back per call, and every turn after the first reuses the pooled socket, `connect()`
never fires again, and the count comes back empty -- silently, which is the part that
matters. A run would still produce a chart. So these tests drive real localhost HTTP
round-trips over one kept-alive connection and assert the second request, the one a
naive counter loses, is counted independently of the first.

Plain HTTP is not a shortcut here: `core.wire` counts http:// with the same connection
class it counts https:// with, precisely so a localhost server exercises the code that
runs in production.
"""

from __future__ import annotations

import http.server
import json
import threading
import time

from core import wire


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"            # keep-alive

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        # A think-time before the first response byte, so first_recv_at cannot land on
        # last_send_at by accident and the ordering assertion means something.
        time.sleep(0.02)
        body = b"response-body-" + b"z" * 200
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _server():
    # Threading server: each keep-alive connection gets its own handler thread, so
    # shutdown() is not blocked by a handler parked on the next-request read.
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _url(srv) -> str:
    host, port = srv.server_address
    return f"http://{host}:{port}/"


def test_bytes_counted_over_a_reused_keepalive_socket():
    srv = _server()
    try:
        sess = wire.session()
        url = _url(srv)
        small = {"x": "a" * 100}
        big = {"y": "b" * 900}

        with wire.wire_counter() as w1:
            sess.post(url, data=json.dumps(small)).content

        # Second request on the SAME pooled socket -- no reconnect, so connect() does
        # not fire. This is the request a per-connection counter loses.
        with wire.wire_counter() as w2:
            sess.post(url, data=json.dumps(big)).content

        assert w1.sent > len(json.dumps(small))   # request line + headers, not just body
        assert w1.recv > 0
        assert w2.sent > len(json.dumps(big))
        assert w2.recv > 0
        # Two independent measurements, not one running total: the bigger body sent
        # more, and the two responses were the same size so the reads match closely.
        assert w2.sent > w1.sent
        assert abs(w2.recv - w1.recv) < 50
    finally:
        srv.shutdown()


def test_counter_measures_only_its_own_block():
    srv = _server()
    try:
        sess = wire.session()
        url = _url(srv)

        with wire.wire_counter() as w1:
            sess.post(url, data='{"a":1}').content
        frozen = w1.sent

        # A request made outside any counter must not leak into the next block.
        sess.post(url, data='{"b":2}').content

        with wire.wire_counter() as w2:
            sess.post(url, data='{"c":3}').content

        assert w1.sent == frozen
        assert 0 < w2.sent < frozen * 2
    finally:
        srv.shutdown()


def test_marks_bracket_the_request_in_order():
    srv = _server()
    try:
        sess = wire.session()
        with wire.wire_counter() as w:
            t0 = time.monotonic()
            sess.post(_url(srv), data=json.dumps({"q": "hello"})).content

        assert w.last_send_at is not None
        assert w.first_recv_at is not None
        # The request finished going out before the first byte came back, and both
        # happened inside the block. Without this ordering the two marks could not be
        # subtracted to get an upload time.
        assert t0 <= w.last_send_at <= w.first_recv_at
        # The handler slept 20 ms before answering, so the gap is real, not a rounding
        # artefact of two stamps taken in the same microsecond.
        assert w.first_recv_at - w.last_send_at >= 0.01
    finally:
        srv.shutdown()


def test_marks_reset_between_blocks():
    srv = _server()
    try:
        sess = wire.session()
        with wire.wire_counter() as w1:
            sess.post(_url(srv), data='{"a":1}').content
        with wire.wire_counter() as w2:
            sess.post(_url(srv), data='{"b":2}').content

        # The second block's marks belong to the second request, not to whatever was
        # left over from the first -- a stale first_recv would report a TTFB from the
        # previous turn.
        assert w2.last_send_at > w1.first_recv_at
    finally:
        srv.shutdown()


def test_reset_session_drops_the_pool():
    first = wire.session()
    assert wire.session() is first          # same session while the pool is warm
    wire.reset_session()
    assert wire.session() is not first      # and a fresh socket for the next capture
