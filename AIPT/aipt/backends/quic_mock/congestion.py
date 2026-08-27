"""IdleProbeCongestionControl: QUIC congestion control that actively probes
RTT during idle gaps and adjusts cwnd based on the measured delta, instead
of leaving cwnd untouched through idle (the aioquic default) or
mechanically halving it a fixed number of times based only on elapsed
time (TCP's ``tcp_cwnd_restart()``, see this package's ``__init__.py``
docstring for the full comparison).

Mechanism (validated end-to-end in a standalone spike before landing
here, 2026-08-27):
  1. Base accounting is fully delegated to aioquic's stock
     ``RenoCongestionControl`` -- this module only *adds* the idle-probe
     adjustment on top, never reimplements slow-start/loss-recovery math.
  2. The caller (``spike_runner.py``, or eventually a real Backend) is
     responsible for idle-gap *detection* (there is no kernel timer to
     hook here -- see __init__.py) and for calling
     ``mark_idle_probe_sent()`` immediately before
     ``connection.send_ping(uid)``.
  3. QUIC's own recovery loop delivers the PING's RTT sample to
     ``on_rtt_measurement()`` exactly like any data ACK (confirmed via
     aioquic source: PING is ack-eliciting, its ACK produces a normal
     ``latest_rtt`` sample). This callback compares the probe RTT to the
     RTT recorded right before idle and shrinks cwnd proportionally to
     any *growth* (a shrunk/unchanged RTT is left alone -- Reno's normal
     growth handles that case; being optimistic off one probe sample is
     riskier than being conservative).
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

from aioquic.quic.congestion.base import (
    K_MINIMUM_WINDOW,
    QuicCongestionControl,
    register_congestion_control,
)
from aioquic.quic.congestion.reno import RenoCongestionControl
from aioquic.quic.packet_builder import QuicSentPacket

log = logging.getLogger(__name__)

#: A single probe sample is noisy; cap how much of the measured growth we
#: act on so one bad sample can't collapse the window (mirrors TCP's own
#: K_MINIMUM_WINDOW floor philosophy for tcp_cwnd_restart()).
MAX_REACTED_GROWTH = 0.5


class IdleProbeCongestionControl(QuicCongestionControl):
    """Reno accounting + an idle-probe-aware cwnd adjustment layered on top.

    Registered under the name ``"idle_probe"`` (see bottom of this module)
    so ``QuicConfiguration(congestion_control_algorithm="idle_probe")``
    picks it the same way any other aioquic algorithm name would.
    """

    def __init__(self, *, max_datagram_size: int) -> None:
        super().__init__(max_datagram_size=max_datagram_size)
        self._reno = RenoCongestionControl(max_datagram_size=max_datagram_size)
        self.congestion_window = self._reno.congestion_window
        self._max_datagram_size_value = max_datagram_size

        # idle-probe bookkeeping
        self.pre_idle_rtt: Optional[float] = None
        self.probe_rtt: Optional[float] = None
        self.awaiting_probe_result = False
        #: Audit trail for the spike -- every adjustment this instance
        #: actually made, for post-run CSV/plot inspection.
        self.idle_adjustments: list[dict] = []

    # --- accounting delegated to reno, cwnd re-synced after each call ---
    def on_packet_sent(self, *, packet: QuicSentPacket) -> None:
        self._reno.on_packet_sent(packet=packet)
        self.bytes_in_flight = self._reno.bytes_in_flight
        self.congestion_window = self._reno.congestion_window

    def on_packet_acked(self, *, now: float, packet: QuicSentPacket) -> None:
        self._reno.on_packet_acked(now=now, packet=packet)
        self.bytes_in_flight = self._reno.bytes_in_flight
        self.congestion_window = self._reno.congestion_window

    def on_packets_expired(self, *, packets: Iterable[QuicSentPacket]) -> None:
        self._reno.on_packets_expired(packets=packets)
        self.bytes_in_flight = self._reno.bytes_in_flight

    def on_packets_lost(self, *, now: float, packets: Iterable[QuicSentPacket]) -> None:
        self._reno.on_packets_lost(now=now, packets=packets)
        self.bytes_in_flight = self._reno.bytes_in_flight
        self.congestion_window = self._reno.congestion_window
        self.ssthresh = self._reno.ssthresh

    # --- the idle-probe adjustment itself -----------------------------
    def on_rtt_measurement(self, *, now: float, rtt: float) -> None:
        self._reno.on_rtt_measurement(now=now, rtt=rtt)
        self.ssthresh = self._reno.ssthresh

        if not self.awaiting_probe_result:
            return
        self.awaiting_probe_result = False
        self.probe_rtt = rtt
        if self.pre_idle_rtt is None or self.pre_idle_rtt <= 0:
            return

        growth = (rtt - self.pre_idle_rtt) / self.pre_idle_rtt
        if growth <= 0:
            log.debug(
                "idle-probe: RTT unchanged/improved (%.1fms -> %.1fms), "
                "cwnd left at %d bytes for Reno to grow normally",
                self.pre_idle_rtt * 1000, rtt * 1000, self.congestion_window,
            )
            return

        factor = max(0.5, 1.0 - min(growth, MAX_REACTED_GROWTH))
        before = self.congestion_window
        new_cwnd = max(
            int(self.congestion_window * factor),
            K_MINIMUM_WINDOW * self._max_datagram_size_value,
        )
        self.congestion_window = new_cwnd
        self._reno.congestion_window = new_cwnd
        self.idle_adjustments.append({
            "t": now, "pre_idle_rtt": self.pre_idle_rtt, "probe_rtt": rtt,
            "growth_pct": round(growth * 100, 1),
            "cwnd_before": before, "cwnd_after": new_cwnd,
        })
        log.info(
            "idle-probe: RTT grew %.1f%% (%.1fms -> %.1fms), cwnd %d -> %d bytes",
            growth * 100, self.pre_idle_rtt * 1000, rtt * 1000, before, new_cwnd,
        )

    def mark_idle_probe_sent(self, last_known_rtt: float) -> None:
        """Call this right before ``connection.send_ping(uid)`` during an
        idle gap -- records the pre-idle RTT baseline the probe's own RTT
        sample will be compared against."""
        self.pre_idle_rtt = last_known_rtt
        self.awaiting_probe_result = True

    def get_log_data(self) -> dict:
        data = super().get_log_data()
        data["idle_adjustments"] = len(self.idle_adjustments)
        return data


register_congestion_control("idle_probe", IdleProbeCongestionControl)

__all__ = ["IdleProbeCongestionControl", "MAX_REACTED_GROWTH"]
