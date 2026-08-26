"""aipt.gateway.app -- FastAPI route tests via TestClient (DESIGN.md 4.7
B9). No real tc execution -- netem_control.apply_profile is exercised
through its own "no tc / no CAP_NET_ADMIN" honest-failure path, which the
sandbox this runs in naturally hits (no mocking needed for that part)."""

import pytest
from fastapi.testclient import TestClient

from aipt.gateway import netem_control
from aipt.gateway.app import app


@pytest.fixture
def client():
    netem_control._STATE.clear()
    return TestClient(app)


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "netem_available" in body
    assert "iface" in body


def test_get_profile_defaults_to_clean(client):
    resp = client.get("/gateway/profile")
    assert resp.status_code == 200
    body = resp.json()
    assert body["profile"] == "clean"
    assert body["delay_ms"] == 0


def test_post_profile_preset(client):
    resp = client.post("/gateway/profile", json={"profile": "3g"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["profile"]["profile"] == "3g"
    # Whether ok is True/False depends on whether this sandbox has tc/NET_ADMIN
    # -- either way the response must not 500 and must explain itself.
    assert "ok" in body
    if not body["ok"]:
        assert body["reason"]


def test_post_profile_custom(client):
    resp = client.post(
        "/gateway/profile",
        json={"profile": "custom", "delay_ms": 33, "jitter_ms": 5, "loss_pct": 0.2, "reorder_pct": 0.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["profile"]["profile"] == "custom"
    assert body["profile"]["delay_ms"] == 33


def test_post_profile_unknown_name_rejected_without_500(client):
    resp = client.post("/gateway/profile", json={"profile": "bogus"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "unknown profile" in body["reason"]


def test_post_profile_missing_field_is_422(client):
    resp = client.post("/gateway/profile", json={})
    assert resp.status_code == 422


def test_post_then_get_reflects_applied_profile_when_tc_available(client, monkeypatch):
    # Force the "tc is available and succeeds" branch so GET afterwards is
    # meaningfully exercised regardless of the sandbox's real capabilities.
    monkeypatch.setattr(netem_control.shutil, "which", lambda name: "/sbin/tc")

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(netem_control.subprocess, "run", lambda *a, **k: _Proc())

    post_resp = client.post("/gateway/profile", json={"profile": "satellite"})
    assert post_resp.json()["ok"] is True

    get_resp = client.get("/gateway/profile")
    assert get_resp.json()["profile"] == "satellite"
