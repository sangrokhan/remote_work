"""Live end-to-end tests for aipt.backends.quic_mock.backend.QuicMockBackend
-- real UDP sockets, real aioquic client+server, real Backend-protocol
lifecycle (connect/send_turn/close). Marked ``live`` per this repo's
convention for anything needing real OS-level I/O. Skipped entirely if
aioquic isn't installed.
"""
from __future__ import annotations

import time

import pytest

aioquic = pytest.importorskip("aioquic", reason="aioquic is an optional [quic] extra")

pytestmark = pytest.mark.live

from aipt.backends.quic_mock.backend import QuicMockBackend  # noqa: E402


def test_quic_mock_backend_ready():
    backend = QuicMockBackend()
    ok, reason = backend.ready()
    assert ok is True


def test_quic_mock_backend_full_lifecycle():
    """connect -> send_turn (x3) -> close, dummy mode (no record bound)."""
    backend = QuicMockBackend(mock_response_bytes=500, algorithm="reno", label="test-lifecycle")
    backend.connect("dummy", "", "")
    try:
        assert ":" in backend.api_host()
        for i in range(3):
            exchange = backend.send_turn(i, f"question {i}", "bytes")
            assert exchange.error is None
            assert exchange.wire_recv == 500
            assert exchange.wire_sent > 0
    finally:
        backend.close()


def test_quic_mock_backend_transport_is_http3():
    backend = QuicMockBackend()
    assert backend.transport == "http3"


def test_quic_mock_backend_cwnd_result_reports_final_snapshot():
    backend = QuicMockBackend(mock_response_bytes=200, algorithm="idle_probe")
    backend.connect("dummy", "", "")
    try:
        backend.send_turn(0, "hello", "bytes")
        result = backend.cwnd_result()
        assert result["final_cwnd"] is not None
        assert result["final_cwnd"] > 0
        assert "note" in result  # honest about not being a continuous trace
        assert result["samples"] == []
    finally:
        backend.close()


def test_quic_mock_backend_algorithm_actual_reflects_requested():
    backend = QuicMockBackend(algorithm="idle_probe")
    backend.connect("dummy", "", "")
    try:
        assert backend.algorithm_actual == "idle_probe"
        assert backend.algorithm_requested == "idle_probe"
    finally:
        backend.close()


def test_quic_mock_backend_default_algorithm_is_reno():
    """No algorithm requested -> defaults to plain reno, not idle_probe --
    DESIGN.md section 7.1's negative A/B result means idle_probe must
    never be an implicit default."""
    from aipt.backends.quic_mock.backend import DEFAULT_ALGORITHM
    assert DEFAULT_ALGORITHM == "reno"

    backend = QuicMockBackend()
    assert backend.algorithm == "reno"


def test_quic_mock_backend_reads_external_server_from_env(monkeypatch):
    """QUIC_MOCK_SERVER_HOST/QUIC_MOCK_SERVER_PORT (docker-compose.yml,
    DESIGN.md 4.7/7.2) must be read at construction time -- same
    "unset means spawn our own on loopback, set means use the
    gateway-routed external server" contract as
    aipt.backends.mock.conversation.MockBackend's MOCK_SERVER_HOST/
    MOCK_SERVER_PORT. Found and fixed 2026-08-31: before this, every
    web-UI QUIC run silently ignored the already-built quic-mock-server/
    gateway topology and only ever talked to itself over loopback."""
    monkeypatch.setenv("QUIC_MOCK_SERVER_HOST", "172.28.2.5")
    monkeypatch.setenv("QUIC_MOCK_SERVER_PORT", "4433")
    backend = QuicMockBackend()
    assert backend._external_host == "172.28.2.5"
    assert backend._external_port == 4433


def test_quic_mock_backend_no_external_server_when_env_unset(monkeypatch):
    monkeypatch.delenv("QUIC_MOCK_SERVER_HOST", raising=False)
    monkeypatch.delenv("QUIC_MOCK_SERVER_PORT", raising=False)
    backend = QuicMockBackend()
    assert backend._external_host == ""
    assert backend._external_port is None


# --- inference_delay_ms (server-side, per DESIGN.md 4.5 "latency knob is
# client-side... except the server-side inference_delay_ms sleep" posture
# MockBackend's TCP path already had) ---------------------------------------
#
# Found missing 2026-08-31 (user report): unlike aipt.backends.mock.server's
# `delay` query param, _MockEchoProtocol's original wire format had no delay
# field at all, so a QUIC mock run never waited before replying regardless
# of what inference_delay_ms was set to -- silently ignored, not merely
# defaulted. Fixed by adding a second uint32 (delay_ms) to the request's
# length-prefix header; these tests assert the server actually holds the
# reply back by roughly that long, not just that the parameter is accepted.


def test_quic_mock_backend_zero_delay_replies_fast():
    """delay=0 (default) should not introduce a multi-hundred-ms wait --
    guards against the fix accidentally making every reply block."""
    backend = QuicMockBackend(mock_response_bytes=100, label="test-delay-zero")
    backend.connect("dummy", "", "")
    try:
        t0 = time.monotonic()
        exchange = backend.send_turn(0, "hi", "bytes")
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert exchange.error is None
        assert elapsed_ms < 300
    finally:
        backend.close()


def test_quic_mock_backend_inference_delay_ms_holds_reply_back():
    """A nonzero inference_delay_ms must make send_turn() take at least
    that long -- the reply is measurably held back server-side, not just
    accepted as a parameter."""
    delay_ms = 400
    backend = QuicMockBackend(
        mock_response_bytes=100, inference_delay_ms=delay_ms, label="test-delay-applied",
    )
    backend.connect("dummy", "", "")
    try:
        t0 = time.monotonic()
        exchange = backend.send_turn(0, "hi", "bytes")
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert exchange.error is None
        # Generous floor (delay minus scheduling slack) rather than an
        # exact bound -- this is a live asyncio timer under test-runner
        # jitter, not a precision clock.
        assert elapsed_ms >= delay_ms * 0.8
    finally:
        backend.close()


def test_quic_mock_backend_inference_delay_ms_applies_in_record_mode():
    """The delay must also apply when a ScenarioRecord is bound
    (input_mode='record') -- the delay field is independent of whether
    the reply body comes from record.turns[i].answer or filler bytes."""
    from aipt.backends.mock.records import ScenarioRecord, Turn

    record = ScenarioRecord(
        name="quic-delay-record-test",
        turns=[Turn(question="q0", answer="a real recorded answer")],
    )
    delay_ms = 350
    backend = QuicMockBackend(
        record=record, inference_delay_ms=delay_ms, label="test-delay-record",
    )
    backend.connect("record", "", record.system_prompt)
    try:
        t0 = time.monotonic()
        exchange = backend.send_turn(0, record.turns[0].question, "bytes")
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert exchange.error is None
        assert exchange.text == "a real recorded answer"
        assert elapsed_ms >= delay_ms * 0.8
    finally:
        backend.close()


def test_quic_mock_backend_delay_applies_every_turn_not_just_first():
    """Multiple turns each pay their own inference_delay_ms -- guards
    against a regression where only turn 0's request carries the delay
    field (e.g. if a future change folded delay into a per-connection
    setting instead of a per-request one)."""
    delay_ms = 250
    backend = QuicMockBackend(
        mock_response_bytes=50, inference_delay_ms=delay_ms, label="test-delay-multi-turn",
    )
    backend.connect("dummy", "", "")
    try:
        for i in range(3):
            t0 = time.monotonic()
            exchange = backend.send_turn(i, f"q{i}", "bytes")
            elapsed_ms = (time.monotonic() - t0) * 1000
            assert exchange.error is None
            assert elapsed_ms >= delay_ms * 0.8, f"turn {i} did not wait for its own delay"
    finally:
        backend.close()
