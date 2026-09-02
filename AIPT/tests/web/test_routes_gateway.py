"""tests/web/test_routes_gateway.py -- aipt.web.routes_gateway (Network
Gateway profile proxy, DESIGN.md 4.7 B11; idle-reset toggle proxy,
2026-09-01 ooo interview). Mocks ``requests.get``/``requests.post`` so
these tests never touch a real network -- the actual Gateway/mock-server/
local-llm containers are exercised in the ooo audit's manual Docker
verification, not here (same "never actually run privileged operations
in unit tests" posture as tests/gateway/test_forwarding.py).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aipt.web.app import create_app


class _FakeResponse:
    def __init__(self, json_body, status_code=200):
        self._json = json_body
        self.status_code = status_code

    def json(self):
        return self._json


@pytest.fixture()
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


# -- Gateway profile proxy ------------------------------------------------

def test_get_gateway_profile_proxies(client, monkeypatch):
    def fake_get(url, timeout=None):
        assert url == "http://gateway:8080/gateway/profile"
        return _FakeResponse({"ok": True, "profile": "custom"})

    monkeypatch.setattr("aipt.web.routes_gateway.requests.get", fake_get)
    res = client.get("/api/gateway/profile")
    assert res.status_code == 200
    assert res.json() == {"ok": True, "profile": "custom"}


def test_set_gateway_profile_proxies_with_profile_body(client, monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse({"ok": True, "profile": "3g"})

    monkeypatch.setattr("aipt.web.routes_gateway.requests.post", fake_post)
    res = client.post("/api/gateway/profile?profile=3g")
    assert res.status_code == 200
    assert res.json() == {"ok": True, "profile": "3g"}
    assert captured["url"] == "http://gateway:8080/gateway/profile"
    assert captured["json"] == {"profile": "3g"}


def test_get_gateway_profile_unreachable_returns_ok_false_not_500(client, monkeypatch):
    import requests

    def fake_get(url, timeout=None):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr("aipt.web.routes_gateway.requests.get", fake_get)
    res = client.get("/api/gateway/profile")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is False
    assert "unreachable" in body["reason"]


def test_gateway_host_port_env_override(client, monkeypatch):
    monkeypatch.setenv("GATEWAY_HOST", "custom-gw")
    monkeypatch.setenv("GATEWAY_PORT", "9999")
    captured = {}

    def fake_get(url, timeout=None):
        captured["url"] = url
        return _FakeResponse({"ok": True})

    monkeypatch.setattr("aipt.web.routes_gateway.requests.get", fake_get)
    client.get("/api/gateway/profile")
    assert captured["url"] == "http://custom-gw:9999/gateway/profile"


# -- idle-reset toggle (CLIENT side, `web` itself, no backend param) -------
#
# Redesigned 2026-09-02: it's `web`'s own send-side cwnd that
# slow-start-after-idle resets for the metric that matters (next-turn
# request upload latency), so /api/idle-reset always targets `web` itself
# (aipt.core.idle_reset, in-process, no network hop, no `requests` mock
# needed) -- the old mock/local_llm admin-proxy path (and this test file's
# coverage of it) was removed as dead code once the redesign made it
# unreachable from the UI.

def test_get_idle_reset_calls_idle_reset_read(client, monkeypatch):
    monkeypatch.setattr(
        "aipt.web.routes_gateway._idle_reset.read",
        lambda path=None: (True, "ready"),
    )
    res = client.get("/api/idle-reset")
    assert res.status_code == 200
    assert res.json() == {"ok": True, "enabled": True, "reason": "ready"}


def test_set_idle_reset_calls_idle_reset_write(client, monkeypatch):
    calls = []

    def fake_write(enabled, path=None):
        calls.append(enabled)
        return True, "ready"

    monkeypatch.setattr("aipt.web.routes_gateway._idle_reset.write", fake_write)
    monkeypatch.setattr("aipt.web.routes_gateway._idle_reset.read", lambda path=None: (False, "ready"))
    res = client.post("/api/idle-reset?enabled=false")
    assert res.status_code == 200
    assert calls == [False]
    body = res.json()
    assert body["write_ok"] is True
    assert body["enabled"] is False


def test_set_idle_reset_write_failure_reported(client, monkeypatch):
    monkeypatch.setattr(
        "aipt.web.routes_gateway._idle_reset.write",
        lambda enabled, path=None: (False, "could not be written: denied"),
    )
    monkeypatch.setattr("aipt.web.routes_gateway._idle_reset.read", lambda path=None: (None, "n/a"))
    res = client.post("/api/idle-reset?enabled=true")
    assert res.status_code == 200
    body = res.json()
    assert body["write_ok"] is False
    assert "denied" in body["write_reason"]


def test_idle_reset_never_makes_http_call(client, monkeypatch):
    # Guard against a regression that routes idle-reset through the (now
    # deleted) HTTP proxy path by mistake -- both requests.get/post must
    # stay untouched.
    def fail(*a, **k):
        raise AssertionError("idle-reset must not go through requests.get/post")

    monkeypatch.setattr("aipt.web.routes_gateway.requests.get", fail)
    monkeypatch.setattr("aipt.web.routes_gateway.requests.post", fail)
    monkeypatch.setattr("aipt.web.routes_gateway._idle_reset.read", lambda path=None: (True, "ready"))
    monkeypatch.setattr("aipt.web.routes_gateway._idle_reset.write", lambda enabled, path=None: (True, "ready"))
    assert client.get("/api/idle-reset").status_code == 200
    assert client.post("/api/idle-reset?enabled=true").status_code == 200
