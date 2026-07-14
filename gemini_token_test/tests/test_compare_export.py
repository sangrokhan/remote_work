"""A comparison run has to be auditable after the fact: the raw request/response
of every case, a flat metrics table, and a visible list of which cases failed.

Without these the run is a set of numbers you have to trust. With them you can
check the arms actually answered the same questions, and see which call broke.
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiment
import metrics


def _client(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    import app as app_module
    importlib.reload(app_module)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def _run(client, turns=2, arms=("stateless", "cached")):
    return client.post("/compare", json={"turns": turns, "arms": list(arms)}).get_json()


# --- raw request/response export ------------------------------------------

def test_json_export_carries_raw_request_and_response(monkeypatch):
    c = _client(monkeypatch)
    body = _run(c)
    r = c.get(f"/download/comparison/{body['exec_id']}.json")
    assert r.status_code == 200
    doc = r.get_json()
    cases = doc["cases"]
    assert cases
    for case in cases:
        assert set(case) >= {"arm", "phase", "turn", "question", "response_text",
                             "request", "response", "error"}
    # The raw bodies are parsed back into objects, not left as escaped strings.
    steady = [c2 for c2 in cases if c2["phase"] == "steady"]
    assert any(isinstance(c2["request"], dict) for c2 in steady)


def test_json_export_404s_for_unknown_run(monkeypatch):
    c = _client(monkeypatch)
    assert c.get("/download/comparison/nope.json").status_code == 404


# --- flat metrics CSV -----------------------------------------------------

def test_csv_export_has_one_row_per_case(monkeypatch):
    c = _client(monkeypatch)
    body = _run(c, turns=2, arms=["stateless"])
    r = c.get(f"/download/comparison/{body['exec_id']}.csv")
    assert r.status_code == 200
    text = r.get_data(as_text=True)
    header = text.splitlines()[0].lstrip("﻿")
    assert header.split(",") == ["arm", "phase", "turn", "wire_sent", "wire_recv",
                                 "req_sent_ms", "ttfb_ms", "ttft_ms", "ttlt_ms",
                                 "turn_end_ms", "store_tail_ms", "elapsed_ms",
                                 "input_tokens", "cached_tokens",
                                 "output_tokens", "thought_tokens", "total_tokens",
                                 "error"]
    assert len(text.strip().splitlines()) == 1 + 2  # header + 2 steady turns


# --- which case failed ----------------------------------------------------

def test_summary_lists_the_failing_cases(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=2, arms=["stateless"])
    out["records"][1]["error"] = "429 quota exceeded"
    s = metrics.summarize_comparison(out)
    assert s["failures"] == [
        {"arm": "stateless", "phase": "steady", "turn": 2, "error": "429 quota exceeded"}
    ]


def test_no_failures_is_an_empty_list(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=1, arms=["stateless"])
    assert metrics.summarize_comparison(out)["failures"] == []


# --- total transaction time ------------------------------------------------

def test_totals_carry_call_time_and_wall_time(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=2, arms=["stateless"])
    s = metrics.summarize_comparison(out)
    t = s["totals"]["stateless"]
    # call_ms is time spent inside calls; wall_ms is the arm start-to-finish clock,
    # so it also covers whatever the arm does between calls (cache deletes, etc).
    assert t["call_ms"] == sum(r["elapsed_ms"] for r in out["records"])
    assert t["wall_ms"] >= 0
