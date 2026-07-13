"""The page reports Developer API readiness, not Vertex.

The comparison runs on generativelanguage with an API key; a header that still
advertises a Vertex project/location tells the operator to check the wrong thing
when a call fails.
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _client(monkeypatch, **env):
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    import app as app_module
    importlib.reload(app_module)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def _page(client):
    return client.get("/").get_data(as_text=True)


def test_page_names_the_developer_api_host(monkeypatch):
    html = _page(_client(monkeypatch, GEMINI_MOCK=None, GEMINI_API_KEY="k"))
    assert "generativelanguage.googleapis.com" in html


def test_page_does_not_advertise_vertex(monkeypatch):
    html = _page(_client(monkeypatch, GEMINI_MOCK=None, GEMINI_API_KEY="k"))
    assert "Vertex" not in html
    assert "aiplatform" not in html


def test_missing_key_shows_how_to_activate(monkeypatch):
    html = _page(_client(monkeypatch, GEMINI_MOCK=None, GEMINI_API_KEY=None))
    assert "GEMINI_API_KEY" in html


def test_comparison_has_its_own_turns_and_pause_inputs(monkeypatch):
    html = _page(_client(monkeypatch, GEMINI_MOCK="1"))
    assert 'id="cmpTurns"' in html
    assert 'id="cmpPause"' in html


def test_interaction_has_its_own_turns_input(monkeypatch):
    html = _page(_client(monkeypatch, GEMINI_MOCK="1"))
    assert 'id="ixTurns"' in html


def test_vertex_run_controls_are_gone(monkeypatch):
    html = _page(_client(monkeypatch, GEMINI_MOCK="1"))
    for gone in ('id="mode"', 'id="start"', 'id="capture"', 'id="tokenChart"'):
        assert gone not in html, gone
