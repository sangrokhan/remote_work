"""The /compare route runs the arm comparison in mock mode and returns a summary
with per-arm series/totals. No live API, no dollar cost."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _client(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    import importlib
    import app as app_module
    importlib.reload(app_module)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_compare_returns_summary(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/compare", json={"turns": 2, "arms": ["stateless", "cached"]})
    assert r.status_code == 200
    body = r.get_json()
    assert body["mode"] == "comparison"
    assert set(body["summary"]["series"]) == {"stateless", "cached"}
    assert body["records"]


def test_compare_rejects_unknown_arms_falls_back(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/compare", json={"turns": 1, "arms": ["bogus"]})
    assert r.status_code == 200
    # Unknown arms are dropped; the full default set runs instead.
    import experiment
    assert set(r.get_json()["summary"]["series"]) == set(experiment.COMPARE_ARMS)


def test_compare_no_dollar_cost(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/compare", json={"turns": 1, "arms": ["stateless"]})
    body = r.get_json()
    assert "cost_usd" not in body["summary"].get("totals", {})
