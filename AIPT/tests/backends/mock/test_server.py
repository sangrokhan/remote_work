"""aipt.backends.mock.server: HTTP/1.1 inference-mock server.

Migrated from tcp_congestion/tests/test_server.py (DESIGN.md 5, A3), plus
new tests for scenario-record answer serving (B1).
"""

import json
import socket
import threading
import time

import pytest

from aipt.backends.mock import server
from aipt.backends.mock.records import ScenarioRecord, Turn


@pytest.fixture()
def srv():
    """Start a real server on a random port; yield (host, port); stop after."""
    s = server.Server(host="127.0.0.1", port=0)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    yield s
    s.shutdown()


@pytest.fixture()
def record_srv():
    record = ScenarioRecord(
        name="test",
        turns=[
            Turn(question="q0", answer="hello"),
            Turn(question="q1", answer="a much longer canned answer text"),
        ],
    )
    s = server.Server(host="127.0.0.1", port=0, record=record)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    yield s
    s.shutdown()


def _get(host, port, path, conn=None):
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


def _post(host, port, path, body: bytes, conn=None):
    if conn is None:
        conn = socket.create_connection((host, port), timeout=5)
    req = (f"POST {path} HTTP/1.1\r\nHost: {host}\r\n"
           f"Connection: keep-alive\r\nContent-Length: {len(body)}\r\n\r\n"
           ).encode() + body
    conn.sendall(req)
    return _read_response(conn), conn


def test_ping_returns_200_with_ts(srv):
    status, _, body = _get(srv.host, srv.port, "/ping")[0]
    assert status == 200
    data = json.loads(body)
    assert "ts" in data


def test_health_returns_200(srv):
    status, _, body = _get(srv.host, srv.port, "/health")[0]
    assert status == 200
    assert json.loads(body)["status"] == "ok"


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
    body = b"x" * 5000
    status, _, resp_body = _post(srv.host, srv.port,
                                  "/inference-mock?delay=10", body)[0]
    assert status == 200
    data = json.loads(resp_body)
    assert data["prompt_bytes"] == 5000
    assert "tokens" in data


def test_inference_mock_post_reports_recv_ms(srv):
    # recv_ms (2026-09-01 idle-reset experiment redesign, docs/experiments/
    # 2026-09-01-idle-reset-results.md): server-observed request-upload
    # latency, the metric the experiment actually needs -- present and
    # non-negative on every POST /inference-mock, GET-only requests never
    # carry it (there is no body to time).
    body = b"x" * 10000
    status, _, resp_body = _post(srv.host, srv.port, "/inference-mock", body)[0]
    assert status == 200
    data = json.loads(resp_body)
    assert "recv_ms" in data
    assert isinstance(data["recv_ms"], (int, float))
    assert data["recv_ms"] >= 0


def test_inference_mock_get_has_no_recv_ms(srv):
    status, _, resp_body = _get(srv.host, srv.port, "/inference-mock")[0]
    assert status == 200
    data = json.loads(resp_body)
    assert "recv_ms" not in data


def test_inference_mock_get_response_bytes_pads_body_to_exact_size(srv):
    status, headers, body = _get(
        srv.host, srv.port, "/inference-mock?response_bytes=1000")[0]
    assert status == 200
    content_length = int(
        [l for l in headers.splitlines() if l.lower().startswith("content-length:")][0]
        .split(":", 1)[1].strip())
    assert content_length == 1000
    assert len(body.encode()) == 1000
    data = json.loads(body)
    assert "tokens" in data


def test_inference_mock_response_bytes_zero_keeps_original_small_body(srv):
    status, _, body = _get(srv.host, srv.port, "/inference-mock")[0]
    assert status == 200
    assert len(body.encode()) < 200


def test_inference_mock_response_bytes_below_json_floor_returns_unpadded(srv):
    status, _, body = _get(srv.host, srv.port, "/inference-mock?response_bytes=5")[0]
    assert status == 200
    data = json.loads(body)
    assert "tokens" in data


def test_keep_alive_reuses_connection(srv):
    resp1, conn = _get(srv.host, srv.port, "/ping")
    resp2, _ = _get(srv.host, srv.port, "/ping", conn=conn)
    conn.close()
    assert resp1[0] == 200
    assert resp2[0] == 200


def test_unknown_path_returns_404(srv):
    status, _, _ = _get(srv.host, srv.port, "/nope")[0]
    assert status == 404



# --- scenario-record answer serving (new, DESIGN.md B1) --------------------


def test_inference_mock_with_record_serves_answer_text(record_srv):
    status, _, body = _get(record_srv.host, record_srv.port,
                            "/inference-mock?turn=0")[0]
    assert status == 200
    data = json.loads(body)
    assert data["answer"] == "hello"


def test_inference_mock_with_record_pads_to_answer_length_by_default(record_srv):
    status, headers, body = _get(record_srv.host, record_srv.port,
                                  "/inference-mock?turn=1")[0]
    assert status == 200
    data = json.loads(body)
    assert data["answer"] == "a much longer canned answer text"
    content_length = int(
        [l for l in headers.splitlines() if l.lower().startswith("content-length:")][0]
        .split(":", 1)[1].strip())
    # padded to at least the answer's own encoded length
    assert content_length >= len("a much longer canned answer text".encode())


def test_inference_mock_with_record_explicit_response_bytes_overrides(record_srv):
    status, headers, body = _get(
        record_srv.host, record_srv.port,
        "/inference-mock?turn=0&response_bytes=500")[0]
    assert status == 200
    content_length = int(
        [l for l in headers.splitlines() if l.lower().startswith("content-length:")][0]
        .split(":", 1)[1].strip())
    assert content_length == 500


def test_inference_mock_with_record_out_of_range_turn_falls_back_to_dummy(record_srv):
    status, _, body = _get(record_srv.host, record_srv.port,
                            "/inference-mock?turn=99")[0]
    assert status == 200
    data = json.loads(body)
    assert "answer" not in data
    assert "tokens" in data


def test_inference_mock_without_turn_param_falls_back_to_dummy(record_srv):
    status, _, body = _get(record_srv.host, record_srv.port, "/inference-mock")[0]
    assert status == 200
    data = json.loads(body)
    assert "answer" not in data


def test_inference_mock_no_record_bound_ignores_turn_param(srv):
    """A plain Server(record=None) must behave exactly as before even if
    a caller happens to pass turn= -- no record means no lookup at all."""
    status, _, body = _get(srv.host, srv.port, "/inference-mock?turn=0")[0]
    assert status == 200
    data = json.loads(body)
    assert "answer" not in data
