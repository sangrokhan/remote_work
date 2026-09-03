"""connection.py -- ``cwnd.csv`` / ``cwnd_summary.csv``: layer 1 of DESIGN.md
4.6's 3-layer export set.

Built directly on ``aipt.core.cwnd.Monitor.result()`` (see ``aipt/core/cwnd.py``
-- the file this module reads is the merged, backend-agnostic monitor that
replaced both ``token_traffic/core/cwnd.py`` and
``tcp_congestion/tcp_congestion/cwnd.py``). A caller collects one or more
monitor results over a run (one per label -- a run may watch several
connections, e.g. one per arm) and hands the list here.

Two views, same split as both ancestors (``token_traffic/core/export.py``'s
``cwnd_csv``/``cwnd_summary_csv`` and ``tcp_congestion/tcp_congestion/export.py``'s
``cwnd_csv``):

  ``connection_csv``          the raw series, one row per (label, tick, socket).
                               Meant to be plotted -- ``snd_cwnd`` against ``t_ms``
                               is the picture: a window that climbs while a turn
                               uploads and drops back to 10 after the model spends
                               a few seconds thinking is the idle reset this whole
                               package exists to measure.
  ``connection_summary_csv``  one row per monitored label: how long it watched,
                               what it saw, and how many times a grown window
                               went back to the initial one (``idle_resets`` --
                               the number the monitoring exists to produce).

``label`` replaces token_traffic's split ``provider``/``arm``/``kind`` columns
(DESIGN.md section 6, decision #1): the monitor itself only knows a single
opaque label string, and a caller that wants the structured form assembles it
before constructing the ``Monitor`` (e.g. ``f"{backend}:{arm}:{kind}"``). The
CSV keeps that single ``label`` column rather than re-splitting it, so the
export layer never has an opinion about a format the monitor itself does not
enforce.
"""

from __future__ import annotations

import csv
import io

from aipt.core import cwnd as cwndmon

# The raw per-tick series. `label` rides at the front because a run can watch
# several connections (one per arm, or a whole conversation under one label)
# and two rows from different labels are not comparable.
CONNECTION_COLUMNS = list(dict.fromkeys(
    ["label", "host", "port", *cwndmon.SAMPLE_FIELDS]))

CONNECTION_SUMMARY_COLUMNS = [
    "label", "host", "port", "ips", "interval_ms",
    "interval_reason", "measurement_confidence",
    "samples", "ticks", "seconds", "sockets", "announced",
    "dumps", "exact_queries", "tracked",
    "peak_cwnd", "final_cwnd", "idle_resets",
    "truncated", "error",
]


def connection_csv(monitors: list[dict]) -> str:
    """Every congestion sample across every monitored label, one row per
    (label, tick, socket).

    ``monitors`` is a list of ``Monitor.result()`` dicts (or anything with the
    same shape -- ``label``, ``host``, ``port``, ``samples``). A run that
    monitored nothing (cwnd unavailable, or monitoring turned off) passes an
    empty list and gets a header with no rows -- "monitored nothing" rather
    than "saw nothing", the same distinction ``docs/outputs.md`` draws for the
    token_traffic original.
    """
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CONNECTION_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for mon in monitors or []:
        head = {"label": mon.get("label", ""), "host": mon.get("host", ""),
                "port": mon.get("port", "")}
        for s in mon.get("samples") or []:
            row = dict(head)
            row.update({c: s.get(c, "") for c in cwndmon.SAMPLE_FIELDS})
            w.writerow(row)
    return buf.getvalue()


def connection_summary_csv(monitors: list[dict]) -> str:
    """One row per monitored label: how long it watched, what it saw, and
    ``idle_resets`` -- the count the monitoring exists to produce.

    ``peak_cwnd`` well above the initial 10 segments with ``idle_resets`` at
    zero is a real result too: on that connection the idle gaps cost nothing.
    """
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CONNECTION_SUMMARY_COLUMNS,
                        extrasaction="ignore")
    w.writeheader()
    for mon in monitors or []:
        w.writerow({
            "label": mon.get("label", ""),
            "host": mon.get("host", ""),
            "port": mon.get("port", ""),
            "ips": " ".join(mon.get("ips") or []),
            "interval_ms": mon.get("interval_ms", ""),
            # B12 (DESIGN.md 4.9): why the sampling period is what it is, and
            # how much to trust samples taken at it -- mirrors
            # Monitor.result()'s own field comment in aipt/core/cwnd.py.
            # ``or ""`` covers both a missing key (older monitor shape) and
            # an explicit ``None`` -- export never raises on either.
            "interval_reason": mon.get("interval_reason") or "",
            "measurement_confidence": mon.get("measurement_confidence") or "",
            "samples": mon.get("sample_count", 0),
            "ticks": mon.get("ticks", 0),
            "seconds": mon.get("seconds", 0),
            "sockets": " ".join(mon.get("sockets") or []),
            "announced": mon.get("announced", 0),
            "dumps": mon.get("dumps", 0),
            "exact_queries": mon.get("exact_queries", 0),
            "tracked": mon.get("tracked", 0),
            "peak_cwnd": mon.get("peak_cwnd", 0),
            "final_cwnd": mon.get("final_cwnd", 0),
            "idle_resets": mon.get("idle_resets", 0),
            "truncated": mon.get("truncated", False),
            "error": mon.get("error", ""),
        })
    return buf.getvalue()
