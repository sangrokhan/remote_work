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
