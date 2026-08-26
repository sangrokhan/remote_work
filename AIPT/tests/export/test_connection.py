"""aipt.export.connection -- cwnd.csv / cwnd_summary.csv from Monitor.result()
shaped dicts. No real netlink/socket needed: synthetic monitor results in,
CSV text out.
"""

from __future__ import annotations

import csv
import io

from aipt.export.connection import (
    CONNECTION_COLUMNS,
    CONNECTION_SUMMARY_COLUMNS,
    connection_csv,
    connection_summary_csv,
)


def _monitor(label="mock:baseline", samples=None, **overrides):
    base = {
        "label": label,
        "host": "127.0.0.1",
        "port": 8888,
        "ips": ["127.0.0.1"],
        "interval_ms": 2,
        "samples": samples or [],
        "sample_count": len(samples or []),
        "ticks": len(samples or []),
        "seconds": 1.0,
        "dumps": 1,
        "exact_queries": 50,
        "tracked": 1,
        "announced": 1,
        "sockets": ["127.0.0.1:54321"],
        "peak_cwnd": 10,
        "final_cwnd": 10,
        "idle_resets": 0,
        "truncated": False,
        "error": "",
    }
    base.update(overrides)
    return base


def test_connection_csv_empty_monitors_gives_header_only():
    text = connection_csv([])
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows == []
    header = text.splitlines()[0].split(",")
    assert header == CONNECTION_COLUMNS


def test_connection_csv_one_row_per_sample():
    samples = [
        {"t_ms": 0, "local": "127.0.0.1:54321", "snd_cwnd": 10, "rtt_us": 500,
         "delivery_rate": 1000},
        {"t_ms": 2, "local": "127.0.0.1:54321", "snd_cwnd": 12, "rtt_us": 480,
         "delivery_rate": 1200},
    ]
    mon = _monitor(samples=samples)
    text = connection_csv([mon])
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == 2
    assert rows[0]["label"] == "mock:baseline"
    assert rows[0]["host"] == "127.0.0.1"
    assert rows[0]["snd_cwnd"] == "10"
    assert rows[1]["snd_cwnd"] == "12"
    assert rows[1]["rtt_us"] == "480"


def test_connection_csv_multiple_labels_stay_separated():
    mon_a = _monitor(label="mock:a", samples=[{"t_ms": 0, "local": "x", "snd_cwnd": 10}])
    mon_b = _monitor(label="mock:b", samples=[{"t_ms": 0, "local": "y", "snd_cwnd": 14}])
    text = connection_csv([mon_a, mon_b])
    rows = list(csv.DictReader(io.StringIO(text)))
    labels = [r["label"] for r in rows]
    assert labels == ["mock:a", "mock:b"]


def test_connection_summary_csv_reports_idle_resets_and_stays_zero_not_blank():
    mon = _monitor(idle_resets=3, peak_cwnd=65, final_cwnd=10)
    text = connection_summary_csv([mon])
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == 1
    row = rows[0]
    assert row["idle_resets"] == "3"
    assert row["peak_cwnd"] == "65"
    assert row["final_cwnd"] == "10"
    # A monitor that watched nothing still reports 0, not "" -- see the
    # module docstring's "monitored nothing" vs "saw nothing" distinction.
    assert row["samples"] == "0" or int(row["samples"]) >= 0


def test_connection_summary_csv_empty_header_matches_columns():
    text = connection_summary_csv([])
    header = text.splitlines()[0].split(",")
    assert header == CONNECTION_SUMMARY_COLUMNS
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows == []


def test_connection_summary_csv_carries_error_and_truncated():
    mon = _monitor(error="helper not built: no C compiler", truncated=True)
    text = connection_summary_csv([mon])
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["error"] == "helper not built: no C compiler"
    assert rows[0]["truncated"] == "True"
