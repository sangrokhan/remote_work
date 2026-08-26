"""aipt.export.turns -- turns.csv, including the new goodput_bps column
(DESIGN.md 4.6 B7). Built from aipt.backends.record.turn_record() dicts so
this test exercises the actual producer, not a hand-rolled stand-in.
"""

from __future__ import annotations

import csv
import io

from aipt.backends.record import Exchange, turn_record
from aipt.export.turns import TURN_COLUMNS, goodput_bps, turns_csv


def _record(**overrides):
    exchange = Exchange(
        wire_sent=1000,
        wire_recv=8000,
        req_payload_bytes=900,
        resp_payload_bytes=7800,
        req_sent_ms=100,
        ttfb_ms=150,
        ttft_ms=200,
        ttlt_ms=900,
        turn_end_ms=1100,
        text="answer",
        request_json={"q": "hi"},
        response_json={"a": "answer"},
        error=None,
    )
    rec = turn_record(
        backend="mock",
        arm="baseline",
        phase="steady",
        turn=1,
        question="hi",
        measure="bytes",
        exchange=exchange,
        usage={"input_tokens": 10, "output_tokens": 20, "cached_tokens": 0},
    )
    rec.update(overrides)
    return rec


def test_goodput_bps_computed_from_wire_recv_and_window():
    rec = _record()
    # window = turn_end_ms(1100) - req_sent_ms(100) = 1000ms = 1s
    # bytes = wire_recv = 8000 -> bits = 64000 -> 64000 bps over 1s
    assert goodput_bps(rec) == 64000.0


def test_goodput_bps_falls_back_to_resp_payload_bytes_without_wire_recv():
    rec = _record(wire_recv=0, resp_payload_bytes=4000)
    assert goodput_bps(rec) == (4000 * 8) / 1.0


def test_goodput_bps_zero_on_nonpositive_window():
    rec = _record(turn_end_ms=100, req_sent_ms=100)  # window == 0
    assert goodput_bps(rec) == 0.0
    rec2 = _record(turn_end_ms=50, req_sent_ms=100)  # window < 0
    assert goodput_bps(rec2) == 0.0


def test_goodput_bps_zero_when_no_bytes_at_all():
    rec = _record(wire_recv=0, resp_payload_bytes=0)
    assert goodput_bps(rec) == 0.0


def test_turns_csv_header_matches_columns():
    text = turns_csv([])
    header = text.splitlines()[0].split(",")
    assert header == TURN_COLUMNS


def test_turns_csv_row_has_backend_arm_phase_and_goodput():
    rec = _record()
    text = turns_csv([rec])
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == 1
    row = rows[0]
    assert row["backend"] == "mock"
    assert row["arm"] == "baseline"
    assert row["phase"] == "steady"
    assert row["turn"] == "1"
    assert row["wire_sent"] == "1000"
    assert row["wire_recv"] == "8000"
    assert row["input_tokens"] == "10"
    assert row["total_tokens"] == "30"
    assert float(row["goodput_bps"]) == 64000.0
    # request_raw/response_raw/question/response_text stay out of the CSV --
    # they are the evidence, not a spreadsheet cell (see module docstring).
    assert "request_raw" not in row
    assert "question" not in row


def test_turns_csv_prep_rows_included_not_dropped():
    steady = _record(phase="steady", turn=1)
    prep = _record(phase="cachegen", turn=0)
    text = turns_csv([prep, steady])
    rows = list(csv.DictReader(io.StringIO(text)))
    phases = [r["phase"] for r in rows]
    assert phases == ["cachegen", "steady"]


def test_turns_csv_optional_synthetic_mock_columns_default_blank():
    """A backend that never populates the tcp_congestion-origin optional
    columns (prompt_bytes/request_ms/idle_ms/probe_*) via turn_record()'s
    `extra` dict gets "" -- present column, absent value -- not a missing
    column."""
    rec = _record()
    text = turns_csv([rec])
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["prompt_bytes"] == ""
    assert rows[0]["probe_rtt_mean_ms"] == ""


def test_turns_csv_carries_optional_columns_when_backend_supplies_them():
    rec = _record(prompt_bytes=512, idle_ms=2500, probe_count=4,
                  probe_rtt_mean_ms=12.5)
    text = turns_csv([rec])
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["prompt_bytes"] == "512"
    assert rows[0]["idle_ms"] == "2500"
    assert rows[0]["probe_count"] == "4"
    assert rows[0]["probe_rtt_mean_ms"] == "12.5"


def test_turns_csv_error_row_preserved():
    rec = _record(error="timeout")
    text = turns_csv([rec])
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["error"] == "timeout"
