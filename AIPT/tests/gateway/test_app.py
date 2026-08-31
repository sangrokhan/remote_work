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
    # DESIGN.md 4.7 확정 설계: two-interface + kernel ip_forward reporting.
    assert "client_iface" in body
    assert "backend_iface" in body
    assert "ip_forward_available" in body
    assert "ip_forward_reason" in body


def test_get_profile_defaults_to_clean(client):
    resp = client.get("/gateway/profile")
    assert resp.status_code == 200
    body = resp.json()
    # DESIGN.md 4.7 확정 설계: profile is now reported per-interface
    # (client-facing + backend-facing), both defaulting to clean.
    assert body["client"]["profile"] == "clean"
    assert body["client"]["delay_ms"] == 0
    assert body["backend"]["profile"] == "clean"
    assert body["backend"]["delay_ms"] == 0
    assert "client_iface" in body and "backend_iface" in body


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
    body = get_resp.json()
    assert body["client"]["profile"] == "satellite"
    assert body["backend"]["profile"] == "satellite"


# --- Startup lifespan: env-derived profile must actually be *applied*, not
# just read (regression test for the gap that let GATEWAY_DELAY_MS sit in
# the environment unapplied until a manual POST /gateway/profile -- caught
# 2026-08-31 because no test here ever entered the TestClient as a context
# manager, and FastAPI/Starlette only run lifespan startup/shutdown hooks
# when TestClient is used via `with TestClient(app) as client: ...`; the
# module-level `client` fixture above deliberately does NOT do that, so it
# cannot exercise this path -- these tests must always enter their own
# `with` block). ---------------------------------------------------------


def test_startup_applies_env_derived_profile(monkeypatch):
    """Booting the app (entering the TestClient's lifespan context) must
    call netem_control.apply_profile_both with the profile derived from
    GATEWAY_PROFILE/GATEWAY_DELAY_MS -- this is the actual "does 20ms get
    installed at container boot" behaviour, not just "is the env var
    readable"."""
    netem_control._STATE.clear()
    monkeypatch.setenv("GATEWAY_PROFILE", "custom")
    monkeypatch.setenv("GATEWAY_DELAY_MS", "20")
    monkeypatch.setenv("GATEWAY_JITTER_MS", "0")
    monkeypatch.setenv("GATEWAY_LOSS_PCT", "0")
    monkeypatch.setenv("GATEWAY_REORDER_PCT", "0")

    calls = []
    monkeypatch.setattr(
        netem_control,
        "apply_profile_both",
        lambda client_iface, backend_iface, profile, **k: calls.append(
            (client_iface, backend_iface, profile)
        )
        or {"ok": True, "profile": profile.as_dict()},
    )

    with TestClient(app):
        pass  # lifespan startup runs on __enter__, shutdown on __exit__

    assert len(calls) == 1, (
        "expected exactly one netem_control.apply_profile_both call during "
        "startup -- if this is 0, the env-derived profile is being read but "
        "never installed via tc (the original bug); if >1, startup is "
        "re-applying on every request instead of once at boot"
    )
    client_iface, backend_iface, profile = calls[0]
    assert client_iface == netem_control.DEFAULT_CLIENT_IFACE
    assert backend_iface == netem_control.DEFAULT_BACKEND_IFACE
    assert profile.delay_ms == 20
    assert profile.name == "custom"


def test_startup_with_default_env_installs_clean(monkeypatch):
    """No GATEWAY_* env set at all -> startup still calls
    apply_profile_both, just with the "clean" (no-op) profile -- startup
    application happens unconditionally, not only when impairment is
    requested."""
    netem_control._STATE.clear()
    for name in (
        "GATEWAY_PROFILE",
        "GATEWAY_DELAY_MS",
        "GATEWAY_JITTER_MS",
        "GATEWAY_LOSS_PCT",
        "GATEWAY_REORDER_PCT",
        "CLIENT_NETEM_DELAY_MS",
        "SERVER_NETEM_DELAY_MS",
    ):
        monkeypatch.delenv(name, raising=False)

    calls = []
    monkeypatch.setattr(
        netem_control,
        "apply_profile_both",
        lambda client_iface, backend_iface, profile, **k: calls.append(profile)
        or {"ok": True, "profile": profile.as_dict()},
    )

    with TestClient(app):
        pass

    assert len(calls) == 1
    assert calls[0].name == "clean"
    assert calls[0].delay_ms == 0


def test_startup_profile_is_actually_installed_end_to_end(monkeypatch):
    """End-to-end (still without real tc/CAP_NET_ADMIN): after entering the
    TestClient's lifespan with GATEWAY_DELAY_MS=20 in the environment, a
    plain GET /gateway/profile (no POST in between) must already report
    delay_ms=20 on both interfaces -- proving the startup hook, not a
    request handler, is what installed it."""
    netem_control._STATE.clear()
    monkeypatch.setenv("GATEWAY_PROFILE", "custom")
    monkeypatch.setenv("GATEWAY_DELAY_MS", "20")
    monkeypatch.setattr(netem_control.shutil, "which", lambda name: "/sbin/tc")

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(netem_control.subprocess, "run", lambda *a, **k: _Proc())

    with TestClient(app) as client:
        resp = client.get("/gateway/profile")  # no POST first
        body = resp.json()
        assert body["client"]["delay_ms"] == 20
        assert body["backend"]["delay_ms"] == 20
