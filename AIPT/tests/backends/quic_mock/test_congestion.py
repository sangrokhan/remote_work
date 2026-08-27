"""Unit tests for aipt.backends.quic_mock.congestion.IdleProbeCongestionControl
-- pure math/state-machine tests, no network. Skipped entirely if aioquic
isn't installed (it's an optional [quic] extra, see pyproject.toml).
"""
from __future__ import annotations

import pytest

aioquic = pytest.importorskip("aioquic", reason="aioquic is an optional [quic] extra")

from aipt.backends.quic_mock.congestion import (  # noqa: E402
    MAX_REACTED_GROWTH,
    IdleProbeCongestionControl,
)


class _FakePacket:
    def __init__(self, sent_bytes: int = 1200, sent_time: float = 0.0):
        self.sent_bytes = sent_bytes
        self.sent_time = sent_time
        self.is_ack_eliciting = True
        self.in_flight = True


def test_registered_under_idle_probe_name():
    from aioquic.quic.congestion.base import create_congestion_control

    cc = create_congestion_control("idle_probe", max_datagram_size=1200)
    assert isinstance(cc, IdleProbeCongestionControl)


def test_initial_cwnd_matches_reno():
    from aioquic.quic.congestion.reno import RenoCongestionControl

    reno = RenoCongestionControl(max_datagram_size=1200)
    idle_cc = IdleProbeCongestionControl(max_datagram_size=1200)
    assert idle_cc.congestion_window == reno.congestion_window


def test_no_adjustment_without_pending_probe():
    """A normal RTT measurement (no mark_idle_probe_sent() call first)
    must never trigger the idle-adjustment path -- only Reno's own
    accounting should run."""
    cc = IdleProbeCongestionControl(max_datagram_size=1200)
    before = cc.congestion_window
    cc.on_rtt_measurement(now=1.0, rtt=0.05)
    assert cc.idle_adjustments == []
    assert cc.congestion_window == before  # reno's own on_rtt_measurement is a no-op for cwnd


def test_rtt_growth_during_probe_shrinks_cwnd():
    cc = IdleProbeCongestionControl(max_datagram_size=1200)
    cc.congestion_window = 20000

    cc.mark_idle_probe_sent(last_known_rtt=0.05)  # 50ms baseline
    assert cc.awaiting_probe_result is True

    cc.on_rtt_measurement(now=1.0, rtt=0.10)  # RTT doubled (100% growth)

    assert cc.awaiting_probe_result is False
    assert len(cc.idle_adjustments) == 1
    adj = cc.idle_adjustments[0]
    assert adj["growth_pct"] == 100.0
    # growth capped at MAX_REACTED_GROWTH (0.5) -> factor = 1 - 0.5 = 0.5
    assert cc.congestion_window == 10000
    assert cc.congestion_window < 20000


def test_rtt_improvement_during_probe_leaves_cwnd_untouched():
    cc = IdleProbeCongestionControl(max_datagram_size=1200)
    cc.congestion_window = 20000

    cc.mark_idle_probe_sent(last_known_rtt=0.10)
    cc.on_rtt_measurement(now=1.0, rtt=0.05)  # RTT halved -- path got better

    assert cc.idle_adjustments == []
    assert cc.congestion_window == 20000  # untouched, Reno's normal growth handles it


def test_cwnd_never_drops_below_minimum_window():
    from aioquic.quic.congestion.base import K_MINIMUM_WINDOW

    cc = IdleProbeCongestionControl(max_datagram_size=1200)
    cc.congestion_window = K_MINIMUM_WINDOW * 1200 + 100  # just above floor

    cc.mark_idle_probe_sent(last_known_rtt=0.01)
    cc.on_rtt_measurement(now=1.0, rtt=1.0)  # massive growth (10000%)

    assert cc.congestion_window >= K_MINIMUM_WINDOW * 1200


def test_growth_cap_prevents_overreaction_to_extreme_samples():
    """A single noisy sample showing e.g. 300% RTT growth must not crash
    cwnd by more than MAX_REACTED_GROWTH allows (defends against acting
    too hard on one probe -- see module docstring)."""
    cc = IdleProbeCongestionControl(max_datagram_size=1200)
    cc.congestion_window = 100_000

    cc.mark_idle_probe_sent(last_known_rtt=0.01)
    cc.on_rtt_measurement(now=1.0, rtt=0.04)  # 300% growth

    expected_factor = max(0.5, 1.0 - MAX_REACTED_GROWTH)
    assert cc.congestion_window == int(100_000 * expected_factor)


def test_get_log_data_reports_adjustment_count():
    cc = IdleProbeCongestionControl(max_datagram_size=1200)
    cc.mark_idle_probe_sent(last_known_rtt=0.05)
    cc.on_rtt_measurement(now=1.0, rtt=0.10)

    data = cc.get_log_data()
    assert data["idle_adjustments"] == 1
    assert "cwnd" in data


def test_accounting_delegates_to_reno_for_packet_lifecycle():
    """Sent/acked/lost accounting should track Reno's own bytes_in_flight
    exactly -- this class must not reimplement that math."""
    cc = IdleProbeCongestionControl(max_datagram_size=1200)
    pkt = _FakePacket(sent_bytes=1200, sent_time=0.0)

    cc.on_packet_sent(packet=pkt)
    assert cc.bytes_in_flight == cc._reno.bytes_in_flight == 1200

    cc.on_packet_acked(now=0.05, packet=pkt)
    assert cc.bytes_in_flight == cc._reno.bytes_in_flight == 0
    assert cc.congestion_window == cc._reno.congestion_window
