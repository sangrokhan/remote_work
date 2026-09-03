"""aipt.backends.mock.conversation: full run() + MockBackend against a real
server, with the live cwnd monitor.

Migrated from tcp_congestion/tests/test_conversation_live.py (DESIGN.md 5,
A3). Skipped where netlink/compiler is unavailable; also requires real
sockets, so the whole module is @pytest.mark.live.
"""

import threading
import time

import pytest

from aipt.backends.mock import conversation, server
from aipt.backends.mock.records import ScenarioRecord, Turn
from aipt.core import cwnd

pytestmark = pytest.mark.live

_available, _reason = cwnd.available()
skip_no_cwnd = pytest.mark.skipif(
    not _available, reason=f"cwnd monitor unavailable: {_reason}")


@pytest.fixture()
def srv():
    s = server.Server(host="127.0.0.1", port=0)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    yield s
    s.shutdown()


@skip_no_cwnd
def test_run_returns_one_entry_per_turn(srv):
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=3,
        turn_user_msg_bytes=200, mock_response_bytes=100,
        inference_delay_ms=20, idle_duration_ms=100, ping_interval_ms=20,
    )
    assert len(result["turns"]) == 3
    assert len(result["probes"]) == 3


@skip_no_cwnd
def test_run_prompt_bytes_increase_across_turns(srv):
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=4,
        turn_user_msg_bytes=200, mock_response_bytes=150,
        inference_delay_ms=10, idle_duration_ms=50, ping_interval_ms=20,
    )
    sizes = [t["prompt_bytes"] for t in result["turns"]]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


@skip_no_cwnd
def test_run_has_continuous_cwnd_samples(srv):
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=2,
        turn_user_msg_bytes=200, mock_response_bytes=100,
        inference_delay_ms=20, idle_duration_ms=200, ping_interval_ms=20,
    )
    assert result["sample_count"] > 0
    assert result["error"] == ""


@skip_no_cwnd
def test_run_label_is_used(srv):
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=1,
        turn_user_msg_bytes=100, mock_response_bytes=50,
        inference_delay_ms=10, idle_duration_ms=30, ping_interval_ms=10,
        label="my-conv",
    )
    assert result["label"] == "my-conv"


@skip_no_cwnd
def test_run_without_capture_has_no_pcap_field(srv):
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=1,
        turn_user_msg_bytes=100, mock_response_bytes=50,
        inference_delay_ms=10, idle_duration_ms=30, ping_interval_ms=10,
    )
    assert result.get("pcap") is None


@skip_no_cwnd
def test_run_with_available_algorithm_sets_it(srv):
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=1,
        turn_user_msg_bytes=100, mock_response_bytes=50,
        inference_delay_ms=10, idle_duration_ms=30, ping_interval_ms=10,
        algorithm="cubic",
    )
    assert result["algorithm_requested"] == "cubic"
    assert result["algorithm"] == "cubic"
    assert result["algorithm_error"] == ""


@skip_no_cwnd
def test_run_with_unavailable_algorithm_reports_error_without_crashing(srv):
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=1,
        turn_user_msg_bytes=100, mock_response_bytes=50,
        inference_delay_ms=10, idle_duration_ms=30, ping_interval_ms=10,
        algorithm="not-a-real-algorithm",
    )
    assert result["algorithm_requested"] == "not-a-real-algorithm"
    assert result["algorithm_error"] != ""
    assert result["algorithm"] != "not-a-real-algorithm"


@skip_no_cwnd
def test_run_sends_mock_response_bytes_as_actual_response_size(srv):
    result = conversation.run(
        host=srv.host, port=srv.port, num_turns=1,
        turn_user_msg_bytes=100, mock_response_bytes=1000,
        inference_delay_ms=10, idle_duration_ms=30, ping_interval_ms=10,
    )
    assert result["turns"][0]["prompt_bytes"] == 100


# --- MockBackend (new, DESIGN.md 4.5 Backend-protocol wrapper) -------------


@skip_no_cwnd
def test_mock_backend_full_lifecycle_with_record():
    record = ScenarioRecord(
        name="live-test",
        turns=[Turn(question="q0", answer="short answer"),
               Turn(question="q1", answer="a somewhat longer second answer")],
    )
    backend = conversation.MockBackend(
        record=record, host="127.0.0.1", port=0,
        inference_delay_ms=5, label="mockbackend-live-test",
    )
    ok, _ = backend.ready()
    assert ok

    backend.connect(arm="record", model="mock-record", system=record.system_prompt)
    try:
        exchanges = []
        for i, turn in enumerate(record.turns):
            exchange = backend.send_turn(turn=i, question=turn.question, measure="bytes")
            exchanges.append(exchange)
    finally:
        backend.close()

    assert len(exchanges) == 2
    assert exchanges[0].text == "short answer"
    assert exchanges[1].text == "a somewhat longer second answer"
    for exc in exchanges:
        assert exc.error is None
        assert exc.wire_sent > 0


@skip_no_cwnd
def test_mock_backend_cwnd_result_available_after_connect():
    backend = conversation.MockBackend(host="127.0.0.1", port=0,
                                        label="mockbackend-cwnd-check")
    backend.connect(arm="dummy", model="mock-record", system="")
    try:
        backend.send_turn(turn=0, question="hi", measure="bytes")
    finally:
        backend.close()
    result = backend.cwnd_result()
    assert result.get("label") == "mockbackend-cwnd-check"


# --- MockBackend against an "external" server via MOCK_SERVER_HOST/PORT ---
# (DESIGN.md 4.7/7.2, found+fixed 2026-08-31: the external-server branch
# had a real bug -- send_turn()'s connect()-guard and its host lookup
# both still referenced self._server, which is always None on this path,
# so every external-server run raised RuntimeError on its first turn. A
# real second server (srv fixture) stands in for the mock-server Docker
# container here -- same code path MockBackend actually exercises when
# MOCK_SERVER_HOST/PORT point at a real container, just without needing
# Docker to run this test.)


@skip_no_cwnd
def test_mock_backend_external_server_full_lifecycle(monkeypatch, srv):
    monkeypatch.setenv("MOCK_SERVER_HOST", srv.host)
    monkeypatch.setenv("MOCK_SERVER_PORT", str(srv.port))
    backend = conversation.MockBackend(
        mock_response_bytes=300, inference_delay_ms=5,
        label="mockbackend-external-test",
    )
    assert backend._external_host == srv.host
    assert backend._external_port == srv.port

    backend.connect(arm="dummy", model="mock-record", system="")
    try:
        # The bug this test guards: connect() must NOT spawn its own
        # server when an external target is configured.
        assert backend._server is None
        assert backend.api_host() == f"{srv.host}:{srv.port}"
        exchanges = [
            backend.send_turn(turn=i, question=f"q{i}", measure="bytes")
            for i in range(3)
        ]
    finally:
        backend.close()

    assert len(exchanges) == 3
    for exc in exchanges:
        assert exc.error is None
        assert exc.wire_sent > 0
        # dummy mode (no record bound): mock_response_bytes pads the JSON
        # body on the wire, but MockBackend.send_turn() only reports
        # wire_recv from the response's "answer" field -- which dummy
        # mode never sets -- so 0 here is correct, matching every other
        # dummy-mode test in this file, not a sign the external-server
        # path dropped bytes.
        assert exc.wire_recv == 0


# --- inference_delay_ms (server-side, aipt.backends.mock.server's `delay`
# query param) -- applies on both input modes. ------


@skip_no_cwnd
def test_mock_backend_inference_delay_ms_holds_reply_back():
    """A nonzero inference_delay_ms measurably delays send_turn()'s
    return -- not just an accepted-but-ignored parameter."""
    backend = conversation.MockBackend(
        host="127.0.0.1", port=0, mock_response_bytes=100,
        inference_delay_ms=400, label="mockbackend-delay-check",
    )
    backend.connect(arm="dummy", model="mock-record", system="")
    try:
        t0 = time.monotonic()
        exchange = backend.send_turn(turn=0, question="hi", measure="bytes")
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert exchange.error is None
        assert elapsed_ms >= 400 * 0.8
    finally:
        backend.close()


@skip_no_cwnd
def test_mock_backend_inference_delay_ms_applies_in_record_mode():
    """The delay must also apply in input_mode='record' (a ScenarioRecord
    bound) -- delay is a server-side query param independent of whether
    the reply body comes from record.turns[i].answer or dummy padding."""
    record = ScenarioRecord(
        name="tcp-delay-record-test",
        turns=[Turn(question="q0", answer="a real recorded answer")],
    )
    backend = conversation.MockBackend(
        record=record, host="127.0.0.1", port=0,
        inference_delay_ms=350, label="mockbackend-delay-record-check",
    )
    backend.connect(arm="record", model="mock-record", system=record.system_prompt)
    try:
        t0 = time.monotonic()
        exchange = backend.send_turn(turn=0, question=record.turns[0].question, measure="bytes")
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert exchange.error is None
        assert exchange.text == "a real recorded answer"
        assert elapsed_ms >= 350 * 0.8
    finally:
        backend.close()


@skip_no_cwnd
def test_mock_backend_zero_delay_replies_fast():
    """delay=0 (default) should not introduce a multi-hundred-ms wait."""
    backend = conversation.MockBackend(
        host="127.0.0.1", port=0, mock_response_bytes=100,
        label="mockbackend-delay-zero-check",
    )
    backend.connect(arm="dummy", model="mock-record", system="")
    try:
        t0 = time.monotonic()
        exchange = backend.send_turn(turn=0, question="hi", measure="bytes")
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert exchange.error is None
        assert elapsed_ms < 300
    finally:
        backend.close()
