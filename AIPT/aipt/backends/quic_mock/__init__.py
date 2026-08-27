"""aipt.backends.quic_mock -- QUIC transport spike for the idle-RTT-probe
congestion control question (DESIGN.md "QUIC idle-probe spike" section).

Background (2026-08-27 conversation): the user wants to know whether an
idle gap can be actively probed (send a 0-size/PING packet, measure RTT,
adjust cwnd on the next real transmission) rather than TCP's current
purely time-based ``tcp_cwnd_restart()`` (halve cwnd floor(idle/RTO)
times, with zero reference to actual measured RTT). Working through the
real kernel congestion_ops interface (net/tcp.h's ``struct
tcp_congestion_ops``) established that:

  * TCP's own keepalive/window-probe packets are deliberately excluded
    from the RTT-sampling pipeline (Karn's algorithm, RFC 6298) -- so an
    idle-probe built on top of TCP's existing probe mechanism would not
    even get a usable RTT sample.
  * ``tcp_congestion_ops`` callbacks (cong_avoid/cong_control/etc.) have
    no authority to originate new packets -- congestion control is
    strictly a "how much/how fast" decision, never a "when" decision in
    the Linux TCP stack. Building an idle-probe TCP algorithm would
    require extending the socket layer itself (a new xmit timer plus
    RTT/RTO-triggered resets), well outside what a
    ``tcp_congestion_ops`` kernel module can do -- and the user
    correctly identified this as "이상한"(architecturally wrong) before
    it was even attempted.
  * QUIC (via aioquic) has no such restriction: ``QuicConnection.send_ping()``
    is a public, always-available application-level API, and PING is an
    ack-eliciting frame that flows through the *same* RTT measurement
    path (``aioquic.quic.recovery.QuicPacketRecovery.on_ack_received()``)
    as ordinary data -- confirmed by reading aioquic's actual source.
    aioquic also already exposes a pluggable per-connection congestion
    control registry (``aioquic.quic.congestion.base.register_congestion_control``),
    so a new algorithm needs zero kernel changes, zero C code, and zero
    root/CAP_SYS_MODULE privilege -- just a Python class.

This package is the first concrete step: a QUIC-based mock backend
(server + client) exercising a custom ``IdleProbeCongestionControl``
(``congestion.py``) that fires a PING mid-idle-gap and shrinks cwnd
proportionally to any measured RTT growth, instead of leaving cwnd
untouched (aioquic's own reno/cubic have *no* idle-restart logic at all
today, unlike TCP).

Scope for this first landing (per user direction, 2026-08-27):
  1. Mock-only, wired through the existing ``aipt/gateway/`` netem
     container so idle-probe vs baseline can be compared under a *real*
     injected delay/loss profile, not just loopback noise.
  2. HTTP/3 (real local_llm traffic over QUIC) is explicitly deferred --
     this backend is a measurement spike, not yet a `Backend` protocol
     implementation callable from ``aipt/web/routes_run.py``. See
     ``spike_runner.py``'s module docstring for exactly what is and
     isn't wired up yet.
"""

from __future__ import annotations

NAME = "quic_mock"

__all__ = ["NAME"]
