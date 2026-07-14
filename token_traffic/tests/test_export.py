"""A CSV outlives the run that produced it. Every column that says what a number means
-- provider, arm, phase, measure -- has to be in the file, because a spreadsheet has no
other memory."""

from __future__ import annotations

import csv
import io

from core import export, metrics, runner

FIXTURE = {"system": "You are terse.", "steps": ["2+2?", "times three?"]}


def _run(**kw):
    run = runner.run(system=FIXTURE["system"], steps=FIXTURE["steps"], **kw)
    run["summary"] = metrics.summarize(run)
    return run


def _rows(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


class TestRecordsCsv:
    def test_a_row_says_which_provider_and_arm_it_came_from(self):
        # "stateless" alone names nothing: a run holds both vendors.
        rows = _rows(export.records_csv(
            _run(providers={"gemini": ["stateless"], "openai": ["chat_stateless"]})))
        assert {r["provider"] for r in rows} == {"gemini", "openai"}
        assert all(r["arm"] for r in rows)

    def test_a_row_says_how_it_was_measured(self):
        rows = _rows(export.records_csv(
            _run(providers={"gemini": ["stateless"]}, measure="latency")))
        assert {r["measure"] for r in rows} == {"latency"}

    def test_prep_rows_are_present_and_labelled_rather_than_dropped(self):
        # A reader who is never shown the cache build cannot discover what the arm paid
        # before its first question.
        rows = _rows(export.records_csv(_run(providers={"gemini": ["cached"]})))
        assert any(r["phase"] == "cachegen" for r in rows)
        assert any(r["phase"] == "steady" for r in rows)

    def test_the_raw_bodies_stay_out_of_the_spreadsheet(self):
        # They are the evidence, and they are in the run's JSON. A 40 KB history echo
        # in a cell makes the file unopenable and the numbers unreadable.
        assert "request_raw" not in export.RECORD_COLUMNS
        assert "response_raw" not in export.RECORD_COLUMNS

    def test_every_mark_has_a_column(self):
        for mark in metrics.MARKS:
            assert mark in export.RECORD_COLUMNS


class TestSummaryCsv:
    def test_one_row_per_provider_arm(self):
        run = _run(providers={"gemini": ["stateless", "nocontext"],
                              "openai": ["chat_stateless"]})
        rows = _rows(export.summary_csv(run))
        assert len(rows) == 3
        assert {(r["provider"], r["arm"]) for r in rows} == {
            ("gemini", "stateless"), ("gemini", "nocontext"),
            ("openai", "chat_stateless")}

    def test_marks_carry_both_a_mean_and_a_median(self):
        # A mean alone hides the one turn that took eight seconds; a median alone hides
        # that it happened at all.
        rows = _rows(export.summary_csv(_run(providers={"gemini": ["stateless"]},
                                             measure="latency")))
        assert rows[0]["ttft_ms_mean"] != ""
        assert rows[0]["ttft_ms_median"] != ""

    def test_prep_is_its_own_column_and_is_not_in_the_totals(self):
        run = _run(providers={"gemini": ["cached"]})
        row = _rows(export.summary_csv(run))[0]
        steady = sum(r["wire_sent"] for r in run["records"]
                     if r["phase"] == "steady")
        assert int(row["wire_sent"]) == steady
        assert int(row["prep_wire_sent"]) > 0

    def test_an_arm_with_no_prep_reports_zero_not_blank(self):
        # Blank reads as "not measured". A stateless arm's zero setup cost is a
        # measurement, and it is the point of the row.
        row = _rows(export.summary_csv(_run(providers={"gemini": ["stateless"]})))[0]
        assert row["prep_calls"] == "0"
        assert row["prep_wire_sent"] == "0"
