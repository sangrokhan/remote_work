"""export: a conversation run's result as CSV.

Two views:
  cwnd_csv   -- the raw congestion-window series, one row per (tick, socket).
                Meant to be plotted: snd_cwnd against t_ms is the picture a
                reader wants -- climbing while a turn uploads, then dropping
                back to 10 after the idle gap is the reset this project
                exists to measure.
  turns_csv  -- one row per turn: what was sent (prompt_bytes), how long the
                idle gap was, and a summary of the RTT probes taken during
                that gap. The full per-sample probe data stays in the JSON;
                a spreadsheet wants the mean, not forty raw points per turn.
"""

from __future__ import annotations

import csv
import io
import statistics

from tcp_congestion import cwnd as cwndmon

CWND_COLUMNS = list(dict.fromkeys(
    ["label", "host", "port", *cwndmon.SAMPLE_FIELDS]))

TURN_COLUMNS = [
    "turn", "prompt_bytes", "request_ms", "idle_ms",
    "probe_count", "probe_rtt_mean_ms", "probe_rtt_min_ms", "probe_rtt_max_ms",
]


def cwnd_csv(result: dict) -> str:
    """Every congestion sample in the run, one row per (tick, socket)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CWND_COLUMNS, extrasaction="ignore")
    w.writeheader()
    head = {"label": result.get("label", ""), "host": result.get("host", ""),
            "port": result.get("port", "")}
    for s in result.get("samples") or []:
        row = dict(head)
        row.update({c: s.get(c, "") for c in cwndmon.SAMPLE_FIELDS})
        w.writerow(row)
    return buf.getvalue()


def _probe_stats(samples: list[dict]) -> dict:
    rtts = [s["rtt_ms"] for s in samples if "rtt_ms" in s]
    if not rtts:
        return {"probe_count": 0, "probe_rtt_mean_ms": "",
                "probe_rtt_min_ms": "", "probe_rtt_max_ms": ""}
    return {
        "probe_count": len(rtts),
        "probe_rtt_mean_ms": round(statistics.mean(rtts), 3),
        "probe_rtt_min_ms": round(min(rtts), 3),
        "probe_rtt_max_ms": round(max(rtts), 3),
    }


def turns_csv(result: dict) -> str:
    """One row per turn: prompt size, timing, and a probe RTT summary."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=TURN_COLUMNS, extrasaction="ignore")
    w.writeheader()

    probes_by_turn = {p.get("turn"): p.get("samples") or []
                       for p in result.get("probes") or []}

    for t in result.get("turns") or []:
        row = {c: t.get(c, "") for c in TURN_COLUMNS}
        row.update(_probe_stats(probes_by_turn.get(t.get("turn"), [])))
        w.writerow(row)
    return buf.getvalue()
