"""One row per step, one column per arm: did the arms actually answer the same?

The metrics CSV says how many bytes each arm spent. It cannot say whether the arms
were having the same conversation -- and an arm that silently degraded (a cache
that never hit, a history the server dropped) still produces perfectly good-looking
bytes. Put the answers side by side and that becomes visible at a glance.
"""

import csv
import importlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _client(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    import app as app_module
    importlib.reload(app_module)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def _rows(client, turns=2, arms=("stateless", "cached", "interaction")):
    body = client.post("/compare", json={"turns": turns, "arms": list(arms)}).get_json()
    r = client.get(f"/download/comparison/{body['exec_id']}-responses.csv")
    assert r.status_code == 200
    text = r.get_data(as_text=True).lstrip("﻿")
    return list(csv.reader(io.StringIO(text)))


def test_one_row_per_step(monkeypatch):
    rows = _rows(_client(monkeypatch), turns=3)
    assert len(rows) == 1 + 3                      # header + 3 steps
    assert [r[0] for r in rows[1:]] == ["1", "2", "3"]


def test_columns_are_the_arms(monkeypatch):
    header = _rows(_client(monkeypatch))[0]
    assert header[:5] == ["turn", "question", "stateless_response",
                          "cached_response", "interaction_response"]


def test_the_raw_request_of_each_arm_is_still_there(monkeypatch):
    header = _rows(_client(monkeypatch))[0]
    assert header[5:] == ["stateless_request", "cached_request",
                          "interaction_request"]


def test_every_arm_answers_every_step(monkeypatch):
    rows = _rows(_client(monkeypatch), turns=2)
    for row in rows[1:]:
        assert row[1]                              # the question
        assert all(cell for cell in row[2:5]), row  # and an answer from each arm


def test_cachegen_rows_are_not_steps(monkeypatch):
    # The cache builds answer nothing; they are preparation. A row for each would
    # be a step that never happened.
    rows = _rows(_client(monkeypatch), turns=2, arms=["cached"])
    assert len(rows) == 1 + 2


def test_only_the_requested_arms_get_columns(monkeypatch):
    header = _rows(_client(monkeypatch), turns=1, arms=["stateless", "interaction"])[0]
    assert "cached_response" not in header
    assert "interaction_response" in header


def test_unknown_run_404s(monkeypatch):
    c = _client(monkeypatch)
    assert c.get("/download/comparison/nope-responses.csv").status_code == 404
