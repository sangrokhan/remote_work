"""server.py: HTTP/1.1 inference-mock server.

Tests cover:
  - /ping returns JSON with ts field
  - /inference-mock delays by requested ms and returns token count
  - /health returns ok
  - keep-alive connection is reused across requests (no re-handshake)
  - response Content-Length is correct so the client can drain exactly
"""

import json
import socket
import threading
import time

import pytest

from tcp_congestion import server


@pytest.fixture()
def srv():
    """Start a real server on a random port; yield (host, port); stop after."""
    s = server.Server(host="127.0.0.1", port=0)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    yield s
    s.shutdown()


def _get(host, port, path, conn=None):
    """One HTTP/1.1 GET. Reuses conn if provided."""
    if conn is None:
        conn = socket.create_connection((host, port), timeout=5)
    req = (f"GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: keep-alive\r\n\r\n"
           ).encode()
    conn.sendall(req)
    return _read_response(conn), conn


def _read_response(conn):
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += conn.recv(4096)
    head, _, body_start = buf.partition(b"\r\n\r\n")
    headers = head.decode()
    length = 0
    for line in headers.splitlines():
        if line.lower().startswith("content-length:"):
            length = int(line.split(":", 1)[1].strip())
    while len(body_start) < length:
        body_start += conn.recv(4096)
    status = int(headers.splitlines()[0].split()[1])
    return status, headers, body_start[:length].decode()


def test_ping_returns_200_with_ts(srv):
    status, _, body = _get(srv.host, srv.port, "/ping")[0]
    assert status == 200
    data = json.loads(body)
    assert "ts" in data


def test_health_returns_200(srv):
    status, _, body = _get(srv.host, srv.port, "/health")[0]
    assert status == 200
    assert json.loads(body)["status"] == "ok"


def _post(host, port, path, body: bytes, conn=None):
    """One HTTP/1.1 POST with a raw body. Reuses conn if provided."""
    if conn is None:
        conn = socket.create_connection((host, port), timeout=5)
    req = (f"POST {path} HTTP/1.1\r\nHost: {host}\r\n"
           f"Connection: keep-alive\r\nContent-Length: {len(body)}\r\n\r\n"
           ).encode() + body
    conn.sendall(req)
    return _read_response(conn), conn


def test_inference_mock_delays_and_returns_tokens(srv):
    t0 = time.monotonic()
    status, _, body = _get(srv.host, srv.port, "/inference-mock?delay=100")[0]
    elapsed_ms = (time.monotonic() - t0) * 1000
    assert status == 200
    data = json.loads(body)
    assert "tokens" in data
    assert elapsed_ms >= 90  # allow 10ms slack


def test_inference_mock_default_delay_is_zero(srv):
    t0 = time.monotonic()
    _get(srv.host, srv.port, "/inference-mock")[0]
    elapsed_ms = (time.monotonic() - t0) * 1000
    assert elapsed_ms < 500


def test_inference_mock_post_echoes_prompt_bytes(srv):
    """POST body size (prompt) is reported back so a client can confirm what
    was actually uploaded for this turn -- the quantity that matters for
    multi-turn cumulative-context experiments."""
    body = b"x" * 5000
    status, _, resp_body = _post(srv.host, srv.port,
                                  "/inference-mock?delay=10", body)[0]
    assert status == 200
    data = json.loads(resp_body)
    assert data["prompt_bytes"] == 5000
    assert "tokens" in data


def test_inference_mock_post_delays_by_requested_ms(srv):
    body = b"y" * 100
    t0 = time.monotonic()
    _post(srv.host, srv.port, "/inference-mock?delay=80", body)[0]
    elapsed_ms = (time.monotonic() - t0) * 1000
    assert elapsed_ms >= 70


def test_inference_mock_get_response_bytes_pads_body_to_exact_size(srv):
    """This is the pcap-visible behaviour the operator expects: the actual
    wire response size must equal response_bytes, not the fixed small JSON
    blob the server used to always send regardless of the request."""
    status, headers, body = _get(
        srv.host, srv.port, "/inference-mock?response_bytes=1000")[0]
    assert status == 200
    # Content-Length header (what tcpdump/Wireshark will show as the
    # response segment size) must match the wire body exactly.
    content_length = int(
        [l for l in headers.splitlines() if l.lower().startswith("content-length:")][0]
        .split(":", 1)[1].strip())
    assert content_length == 1000
    assert len(body.encode()) == 1000
    data = json.loads(body)
    assert "tokens" in data


def test_inference_mock_post_response_bytes_pads_body_to_exact_size(srv):
    body = b"x" * 5000
    status, headers, resp_body = _post(
        srv.host, srv.port, "/inference-mock?response_bytes=1000", body)[0]
    assert status == 200
    assert len(resp_body.encode()) == 1000
    data = json.loads(resp_body)
    assert data["prompt_bytes"] == 5000


def test_inference_mock_response_bytes_zero_keeps_original_small_body(srv):
    """Omitting response_bytes (or 0) must not change existing behaviour --
    callers who don't care about response size shouldn't pay for padding."""
    status, _, body = _get(srv.host, srv.port, "/inference-mock")[0]
    assert status == 200
    assert len(body.encode()) < 200  # the small fixed JSON, not padded


def test_inference_mock_response_bytes_below_json_floor_returns_unpadded(srv):
    """A response_bytes smaller than the base JSON's own encoded size can't
    be honoured (can't shrink field names/braces) -- must not crash, and
    must fall back to the un-padded JSON rather than truncating it."""
    status, _, body = _get(srv.host, srv.port, "/inference-mock?response_bytes=5")[0]
    assert status == 200
    data = json.loads(body)  # still valid JSON, not truncated garbage
    assert "tokens" in data


def test_keep_alive_reuses_connection(srv):
    """Two requests on the same socket: the second must not get ECONNRESET."""
    resp1, conn = _get(srv.host, srv.port, "/ping")
    resp2, _ = _get(srv.host, srv.port, "/ping", conn=conn)
    conn.close()
    assert resp1[0] == 200
    assert resp2[0] == 200


def test_unknown_path_returns_404(srv):
    status, _, _ = _get(srv.host, srv.port, "/does-not-exist")[0]
    assert status == 404
