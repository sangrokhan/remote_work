"""Live end-to-end tests for aipt.backends.quic_mock.backend.QuicMockBackend
-- real UDP sockets, real aioquic client+server, real Backend-protocol
lifecycle (connect/send_turn/close). Marked ``live`` per this repo's
convention for anything needing real OS-level I/O. Skipped entirely if
aioquic isn't installed.
"""
from __future__ import annotations

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
