"""tcpinfo: read TCP_INFO via getsockopt for a connected socket.

Fields returned:
  cwnd            snd_cwnd (segments)
  rtt_ms          smoothed RTT (ms)
  rto_ms          retransmission timeout (ms)
  delivery_rate   bytes/sec (Linux 4.9+, 0 on older kernels)

All values are best-effort: a platform without TCP_INFO (non-Linux, or a
kernel too old) returns zeros rather than crashing the experiment.
"""

from __future__ import annotations

import ctypes
import socket
import struct
import sys

# struct tcp_info offsets we care about (Linux kernel ABI, stable since 2.6)
# See /usr/include/linux/tcp.h: struct tcp_info
#
# The struct has grown over time; we only read up to delivery_rate which was
# added in 4.9 (offset 200, u64).  We request the whole struct and check the
# returned length: if shorter, the field was not written.

_TCP_INFO_FMT = (
    "B"   # tcpi_state
    "B"   # tcpi_ca_state
    "B"   # tcpi_retransmits
    "B"   # tcpi_probes
    "B"   # tcpi_backoff
    "B"   # tcpi_options
    "B"   # tcpi_snd_wscale : 4, tcpi_rcv_wscale : 4
    "B"   # tcpi_delivery_rate_app_limited : 1
    "I"   # tcpi_rto          (us)
    "I"   # tcpi_ato
    "I"   # tcpi_snd_mss
    "I"   # tcpi_rcv_mss
    "I"   # tcpi_unacked
    "I"   # tcpi_sacked
    "I"   # tcpi_lost
    "I"   # tcpi_retrans
    "I"   # tcpi_fackets
    "I"   # tcpi_last_data_sent
    "I"   # tcpi_last_ack_sent
    "I"   # tcpi_last_data_recv
    "I"   # tcpi_last_ack_recv
    "I"   # tcpi_pmtu
    "I"   # tcpi_rcv_ssthresh
    "I"   # tcpi_rtt            (us)
    "I"   # tcpi_rttvar
    "I"   # tcpi_snd_ssthresh
    "I"   # tcpi_snd_cwnd
    "I"   # tcpi_advmss
    "I"   # tcpi_reordering
    "I"   # tcpi_rcv_rtt
    "I"   # tcpi_rcv_space
    "I"   # tcpi_total_retrans
    "Q"   # tcpi_pacing_rate    (bytes/s, 4.0+)
    "Q"   # tcpi_max_pacing_rate
    "Q"   # tcpi_bytes_acked    (4.2+)
    "Q"   # tcpi_bytes_received
    "I"   # tcpi_segs_out       (4.2+)
    "I"   # tcpi_segs_in
    "I"   # tcpi_notsent_bytes  (4.5+)
    "I"   # tcpi_min_rtt
    "I"   # tcpi_data_segs_in   (4.6+)
    "I"   # tcpi_data_segs_out
    "Q"   # tcpi_delivery_rate  (4.9+)
)

_FMT = "=" + "".join(_TCP_INFO_FMT)
_SIZE = struct.calcsize(_FMT)

# Field indices in the unpacked tuple (0-based)
_IDX_RTO    = 8
_IDX_RTT    = 23
_IDX_CWND   = 26
_IDX_DR     = 42   # delivery_rate (field 42 counting from 0 in _TCP_INFO_FMT)

TCP_INFO = getattr(socket, "TCP_INFO", 11)   # 11 on Linux


def snapshot(sock: socket.socket) -> dict:
    """Return {cwnd, rtt_ms, rto_ms, delivery_rate} for *sock* right now.

    Returns zeros on any platform or kernel that does not support TCP_INFO.
    """
    if not sys.platform.startswith("linux"):
        return _zeros()
    try:
        buf = sock.getsockopt(socket.IPPROTO_TCP, TCP_INFO, _SIZE)
    except OSError:
        return _zeros()

    if len(buf) < struct.calcsize("=" + "".join(_TCP_INFO_FMT[:_IDX_CWND + 1])):
        return _zeros()

    # Unpack only as far as the data allows
    fmt_fields = list(_TCP_INFO_FMT)
    partial_fmt = "=" + "".join(fmt_fields[:_IDX_DR + 1])
    partial_size = struct.calcsize(partial_fmt)
    if len(buf) >= partial_size:
        vals = struct.unpack_from(partial_fmt, buf)
        delivery_rate = vals[_IDX_DR]
    else:
        vals = struct.unpack_from(
            "=" + "".join(fmt_fields[:_IDX_CWND + 1]), buf)
        delivery_rate = 0

    return {
        "cwnd": int(vals[_IDX_CWND]),
        "rtt_ms": vals[_IDX_RTT] / 1000,
        "rto_ms": vals[_IDX_RTO] / 1000,
        "delivery_rate": int(delivery_rate),
    }


def _zeros() -> dict:
    return {"cwnd": 0, "rtt_ms": 0.0, "rto_ms": 0.0, "delivery_rate": 0}
