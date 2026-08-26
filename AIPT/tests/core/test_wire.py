"""The counter has to survive keep-alive, and its marks have to be in the right
order.

Ported from ``token_traffic/tests/test_wire.py`` (DESIGN.md 5, A2) onto
``aipt.core.wire`` -- unchanged in behaviour, just the import path.
"""

from __future__ import annotations

import http.server
import json
import threading
import time

from aipt.core import wire


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"            # keep-alive

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        time.sleep(0.02)
        body = b"response-body-" + b"z" * 200
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _server():
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

        with wire.wire_counter() as w2:
            sess.post(url, data=json.dumps(big)).content

        assert w1.sent > len(json.dumps(small))
        assert w1.recv > 0
        assert w2.sent > len(json.dumps(big))
        assert w2.recv > 0
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
        assert t0 <= w.last_send_at <= w.first_recv_at
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

        assert w2.last_send_at > w1.first_recv_at
    finally:
        srv.shutdown()


def test_reset_session_drops_the_pool():
    first = wire.session()
    assert wire.session() is first
    wire.reset_session()
    assert wire.session() is not first
