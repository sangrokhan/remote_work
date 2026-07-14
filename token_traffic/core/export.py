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
