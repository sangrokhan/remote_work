"""Wire byte counting must survive keep-alive connection reuse.

The old counter was installed in `_CountingHTTPSConnection.connect()`, which only
fires when a *new* TCP connection opens. `requests` pools connections, so every
turn after the first reused the socket, `connect()` never fired again, and the
per-call read of `_active_counter` came back `None` — silently falling back to the
JSON payload size with no HTTP headers and no content-encoding.

These tests drive a real localhost HTTP round-trip twice over one kept-alive
connection and assert both requests are counted, headers included, independently.
The second request is the one the old code got wrong.
"""

import http.client
import http.server
import threading
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemini_client import _CountingSocket, wire_counter


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"          # keep-alive

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        body = b"response-body-" + b"z" * 200
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _server():
    # Threading server: each keep-alive connection gets its own handler thread,
    # so shutdown() isn't blocked by a handler parked on the next-request read.
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


def _post(conn, body):
    conn.request("POST", "/", body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp.read()


def test_first_request_counts_headers_not_just_body():
    srv = _server()
    try:
        conn = http.client.HTTPConnection(*srv.server_address)
        conn.connect()
        conn.sock = _CountingSocket(conn.sock)     # wrap once, as production does

        body = b'{"x":"' + b"a" * 100 + b'"}'
        with wire_counter() as w:
            _post(conn, body)

        assert w.sent > len(body)      # request line + headers, not just the body
        assert w.recv > 0
    finally:
        conn.close()
        srv.shutdown()


def test_reused_keepalive_connection_is_still_counted():
    srv = _server()
    try:
        conn = http.client.HTTPConnection(*srv.server_address)
        conn.connect()
        conn.sock = _CountingSocket(conn.sock)

        body1 = b'{"x":"' + b"a" * 100 + b'"}'
        with wire_counter() as w1:
            _post(conn, body1)

        # Second request on the SAME socket — no reconnect. This is what the old
        # counter missed entirely.
        body2 = b'{"y":"' + b"b" * 300 + b'"}'
        with wire_counter() as w2:
            _post(conn, body2)

        assert w2.sent > len(body2)    # counted, and headers included
        assert w2.recv > 0
        # Independent measurements: the bigger body must produce more sent bytes.
        assert w2.sent > w1.sent
    finally:
        conn.close()
        srv.shutdown()


def test_counter_only_measures_its_own_block():
    srv = _server()
    try:
        conn = http.client.HTTPConnection(*srv.server_address)
        conn.connect()
        conn.sock = _CountingSocket(conn.sock)

        with wire_counter() as w1:
            _post(conn, b'{"a":1}')
        sent_after_first = w1.sent

        # A request made OUTSIDE any counter must not leak into the next block.
        _post(conn, b'{"b":2}')

        with wire_counter() as w2:
            _post(conn, b'{"c":3}')

        # w1 is frozen; the out-of-block request didn't inflate it.
        assert w1.sent == sent_after_first
        # w2 saw only its own one request, not the untracked one before it.
        assert 0 < w2.sent < sent_after_first * 3
    finally:
        conn.close()
        srv.shutdown()
