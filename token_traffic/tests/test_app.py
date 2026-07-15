"""The routes, and the two promises they make: nothing is billed without being counted
first, and nothing synthetic is ever allowed to pass for something measured."""

from __future__ import annotations

import json

import pytest

from core import app as web
from core import store


@pytest.fixture
def client():
    web.app.config["TESTING"] = True
    return web.app.test_client()


class TestConfig:
    def test_the_preflight_says_what_is_ready_and_why_not(self, client):
        cfg = client.get("/api/config").get_json()
        for p in cfg["providers"]:
            assert p["arms"] and p["headline_arms"]
            assert p["ready"] or p["reason"]     # a bare False helps nobody
        assert cfg["capture"]["available"] or cfg["capture"]["reason"]
        assert cfg["fixture"]["turns"] > 0

    def test_mock_mode_is_declared(self, client):
        assert client.get("/api/config").get_json()["mock"] is True


class TestPreflight:
    def test_it_counts_the_calls_before_they_go_out(self, client):
        r = client.post("/api/preflight", json={
            "providers": {"gemini": ["stateless", "cached"]},
            "measure": "both", "turns": 3})
        body = r.get_json()
        assert body["pairs"] == ["gemini:stateless", "gemini:cached"]
        # 2 arms x 3 turns x 2 passes -- but this suite runs in mock, and a mock call
        # is not billable. The number the operator is shown must say which world it is
        # in, or the confirmation dialog is theatre.
        assert body["billable_calls"] == 0
        assert body["mock"] is True

    def test_it_warns_about_the_pass_that_would_double_bill_a_stateful_arm(self, client):
        body = client.post("/api/preflight", json={
            "providers": {"openai": ["responses_inline"]},
            "measure": "both"}).get_json()
        assert any("twice" in w for w in body["warnings"])

    def test_a_bad_selection_is_refused_with_a_reason(self, client):
        r = client.post("/api/preflight", json={"providers": {"gemini": ["telepathy"]}})
        assert r.status_code == 400
        assert "telepathy" in r.get_json()["error"]

    def test_a_bad_measure_is_refused(self, client):
        r = client.post("/api/preflight", json={"measure": "vibes"})
        assert r.status_code == 400


class TestRun:
    def test_a_run_is_saved_summarized_and_marked_mock(self, client):
        body = client.post("/api/run", json={
            "providers": {"gemini": ["stateless"], "openai": ["chat_stateless"]},
            "turns": 2}).get_json()
        run = body["run"]

        assert run["mock"] is True
        assert run["summary"]["keys"] == ["gemini:stateless", "openai:chat_stateless"]
        assert store.get_run(run["exec_id"]) is not None

    def test_the_summary_is_frozen_into_the_run_not_recomputed_per_view(self, client):
        # Recomputing on every page view means an old run's numbers change when the
        # metrics code changes -- which is how a chart quietly disagrees with the CSV
        # next to it.
        run = client.post("/api/run", json={"providers": {"gemini": ["stateless"]},
                                            "turns": 1}).get_json()["run"]
        stored = store.get_run(run["exec_id"])
        assert stored["summary"] == run["summary"]

    def test_the_stream_reports_progress_before_it_reports_the_run(self, client):
        r = client.post("/api/run/stream", json={
            "providers": {"gemini": ["stateless"]}, "turns": 2})
        events = [json.loads(line[6:]) for line in r.get_data(as_text=True).splitlines()
                  if line.startswith("data: ")]

        progress = [e for e in events if "event" not in e]
        assert progress and progress[0]["turns"] == 2
        assert all(e["provider"] == "gemini" for e in progress)
        assert events[-1]["event"] == "done"
        assert events[-1]["run"]["records"]

    def test_a_bad_selection_never_opens_the_stream(self, client):
        # A 400 inside an SSE body is a 200 as far as the browser is concerned.
        r = client.post("/api/run/stream", json={"providers": {"gemini": ["nope"]}})
        assert r.status_code == 400


class TestHistoryAndDownloads:
    def test_mock_runs_are_listed_apart_from_live_ones(self, client):
        client.post("/api/run", json={"providers": {"gemini": ["stateless"]},
                                      "turns": 1})
        listing = client.get("/api/runs").get_json()
        assert listing["mock_runs"]
        assert listing["runs"] == []      # nothing live was ever run here

    def test_a_mock_csv_says_so_in_its_filename(self, client):
        # A number lifted out of a spreadsheet has no other way of remembering it was
        # never measured.
        run = client.post("/api/run", json={"providers": {"gemini": ["stateless"]},
                                            "turns": 1}).get_json()["run"]
        r = client.get(f"/api/runs/{run['exec_id']}/records.csv")
        assert "mock_records_" in r.headers["Content-Disposition"]
        assert r.get_data(as_text=True).splitlines()[0].startswith("provider,arm,phase")

    def test_summary_csv_downloads(self, client):
        run = client.post("/api/run", json={"providers": {"gemini": ["stateless"]},
                                            "turns": 1}).get_json()["run"]
        r = client.get(f"/api/runs/{run['exec_id']}/summary.csv")
        assert r.status_code == 200
        assert "gemini,stateless" in r.get_data(as_text=True)

    def test_an_unknown_run_is_a_404_not_an_empty_csv(self, client):
        assert client.get("/api/runs/nope/records.csv").status_code == 404
        assert client.get("/api/runs/nope").status_code == 404

    def test_a_run_can_be_deleted(self, client):
        run = client.post("/api/run", json={"providers": {"gemini": ["stateless"]},
                                            "turns": 1}).get_json()["run"]
        assert client.delete(f"/api/runs/{run['exec_id']}").get_json()["ok"] is True
        assert client.get(f"/api/runs/{run['exec_id']}").status_code == 404

    def test_a_pcap_name_cannot_escape_the_pcap_directory(self, client):
        assert client.get("/api/pcaps/..%2f..%2fetc%2fpasswd").status_code in (308, 404)
        assert client.get("/api/pcaps/nope.pcap").status_code == 404
