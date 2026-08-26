"""aipt.export -- the 3-layer CSV set every backend produces, plus the zip
that bundles them for download.

DESIGN.md 4.6 ("통계/CSV 3-레이어 통합"): whichever of the three backends
(``public_ai`` / ``mock`` / ``local_llm``) a run talked to, it leaves behind
the same three CSVs:

  * :mod:`aipt.export.connection` -- ``cwnd.csv`` / ``cwnd_summary.csv``,
    one row per (label, tick, socket) from ``aipt.core.cwnd.Monitor``.
  * :mod:`aipt.export.turns` -- ``turns.csv``, one row per
    ``aipt.backends.record.turn_record()`` dict, with the new
    ``goodput_bps`` column (B7) filled in.
  * :mod:`aipt.export.packets` -- ``packets.csv`` (B6, new), one row per
    packet in a pcap: arrival time, inter-arrival gap, size.

:mod:`aipt.export.bundle` zips all of the above (plus the raw pcap) the way
``tcp_congestion``'s ``download_bundle_zip`` route did.
"""

from __future__ import annotations
