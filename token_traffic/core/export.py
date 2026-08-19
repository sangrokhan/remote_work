"""A run as CSV: one row per record, and two rows that are not comparable never look
comparable.

The columns that make that possible ride at the front, before any number:
`provider` and `arm` (a run holds both vendors, so `stateless` alone names nothing),
`phase` (a cache build is not a turn), and `measure` (bytes off a streamed pass are
padded and framed; bytes off a blocking pass are not, and averaging the two is the
mistake this column exists to prevent).

The raw request and response bodies are not in the CSV. They are the evidence and they
are in the run's JSON; putting a 40 KB history echo into a spreadsheet cell makes the
file unopenable and the numbers unreadable.
"""

from __future__ import annotations

import csv
import io

from core import cwnd as cwndmon
from core import metrics

RECORD_COLUMNS = [
    "provider", "arm", "phase", "turn", "measure",
    "wire_sent", "wire_recv", "req_payload_bytes", "resp_payload_bytes",
    "req_sent_ms", "ttfb_ms", "ttft_ms", "ttlt_ms", "turn_end_ms", "store_tail_ms",
    "input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens",
    "total_tokens",
    "error",
]

SUMMARY_COLUMNS = [
    "provider", "arm", "measure", "turns",
    "wire_sent", "wire_recv", "wire",
    "input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens",
    "total_tokens",
    # Mean and median for each mark. A mean alone hides the one turn that took eight
    # seconds; a median alone hides that it happened at all.
    *[f"{m}_{s}" for m in metrics.MARKS for s in ("mean", "median")],
    "call_ms", "wall_ms",
    "prep_calls", "prep_wire_sent", "prep_wire_recv",
    "errors",
]


def records_csv(run: dict) -> str:
    """Every record, prep rows included -- phased, not dropped. A reader who wants only
    the steady turns can filter; a reader who is never shown the cache build cannot
    discover what the arm paid before its first question."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=RECORD_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for rec in run.get("records") or []:
        w.writerow({c: rec.get(c, "") for c in RECORD_COLUMNS})
    return buf.getvalue()


def summary_csv(run: dict) -> str:
    """One row per (provider, arm): the totals, the marks, and what prep cost."""
    summary = run.get("summary") or metrics.summarize(run)
    totals = summary.get("totals") or {}
    prep = summary.get("prep") or {}

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for key in summary.get("keys") or list(totals):
        t = totals.get(key) or {}
        p = prep.get(key) or {}
        row = {c: t.get(c, "") for c in SUMMARY_COLUMNS}
        for mark in metrics.MARKS:
            stats = (t.get("marks") or {}).get(mark) or {}
            row[f"{mark}_mean"] = stats.get("mean", "")
            row[f"{mark}_median"] = stats.get("median", "")
        # Zero, not blank, when an arm has no prep: blank reads as "not measured", and
        # a stateless arm's zero setup cost is a measurement and the point of the row.
        row["prep_calls"] = p.get("calls", 0)
        row["prep_wire_sent"] = p.get("wire_sent", 0)
        row["prep_wire_recv"] = p.get("wire_recv", 0)
        w.writerow(row)
    return buf.getvalue()


# The congestion samples, flattened. Same rule as the record CSV: the columns that say
# which arm a row belongs to come first, because a run holds several arms and two rows
# from different arms are not comparable. `local` and `remote` name the socket and are
# already in SAMPLE_FIELDS, so the de-dup keeps one copy of each rather than emitting
# the column twice.
CWND_COLUMNS = list(dict.fromkeys(
    ["provider", "arm", "kind", "host", "local", "remote", *cwndmon.SAMPLE_FIELDS]))

CWND_SUMMARY_COLUMNS = [
    "provider", "arm", "kind", "host", "ips", "interval_ms",
    "samples", "ticks", "seconds", "sockets",
    "peak_cwnd", "final_cwnd", "idle_resets",
    "truncated", "error",
]


def _monitors(run: dict):
    """Every monitor result in the run, as (kind, result). `cwnd` is
    {"provider:arm": {"bytes": {...}, "latency": {...}}}, the same shape as `pcaps`."""
    for _key, by_kind in sorted((run.get("cwnd") or {}).items()):
        for kind, mon in sorted((by_kind or {}).items()):
            if mon:
                yield kind, mon


def cwnd_csv(run: dict) -> str:
    """Every congestion sample in the run, one row per (arm, tick, socket).

    This is the raw series -- a hundred rows a second per socket -- and it is meant to
    be plotted rather than read. `snd_cwnd` against `t_ms` is the picture: a window
    that climbs while a turn uploads and then drops back to 10 after the model spends
    a few seconds thinking is the idle reset. `snd_ssthresh` says where slow start
    will hand off to congestion avoidance, and `rtt_us` says what each round trip the
    reset costs is worth in milliseconds.
    """
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CWND_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for kind, mon in _monitors(run):
        head = {"provider": mon.get("provider", ""), "arm": mon.get("arm", ""),
                "kind": mon.get("kind") or kind, "host": mon.get("host", "")}
        for s in mon.get("samples") or []:
            row = dict(head)
            row.update({c: s.get(c, "") for c in cwndmon.SAMPLE_FIELDS})
            w.writerow(row)
    return buf.getvalue()


def cwnd_summary_csv(run: dict) -> str:
    """One row per monitored arm: how long it watched, what it saw, and how many times
    a grown window went back to the initial one.

    `idle_resets` is the number the monitoring exists to produce. Zero alongside a
    `peak_cwnd` well above 10 means the window survived the idle gaps; anything above
    zero is the slow-start-after-idle tax, paid once per occurrence, on a connection
    the kernel had already taught to go faster.
    """
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CWND_SUMMARY_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for kind, mon in _monitors(run):
        w.writerow({
            "provider": mon.get("provider", ""),
            "arm": mon.get("arm", ""),
            "kind": mon.get("kind") or kind,
            "host": mon.get("host", ""),
            "ips": " ".join(mon.get("ips") or []),
            "interval_ms": mon.get("interval_ms", ""),
            "samples": mon.get("sample_count", 0),
            "ticks": mon.get("ticks", 0),
            "seconds": mon.get("seconds", 0),
            "sockets": " ".join(mon.get("sockets") or []),
            "peak_cwnd": mon.get("peak_cwnd", 0),
            "final_cwnd": mon.get("final_cwnd", 0),
            "idle_resets": mon.get("idle_resets", 0),
            "truncated": mon.get("truncated", False),
            "error": mon.get("error", ""),
        })
    return buf.getvalue()
