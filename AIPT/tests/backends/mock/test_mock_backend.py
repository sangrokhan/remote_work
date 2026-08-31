"""aipt.backends.mock's MockBackend against the sync (non-live) surfaces
that don't need real sockets/cwnd -- import shape and construction only.
Full lifecycle coverage is in test_conversation_live.py (@pytest.mark.live).
"""

from aipt.backends.base import Backend
from aipt.backends.mock.conversation import MockBackend


def test_mock_backend_satisfies_protocol_uninitialized():
    backend = MockBackend()
    assert isinstance(backend, Backend)
    assert backend.NAME == "mock"
    assert backend.transport == "http1"


def test_mock_backend_ready_is_always_true():
    backend = MockBackend()
    ok, reason = backend.ready()
    assert ok
    assert reason


def test_mock_backend_api_host_before_connect_uses_bind_args():
    backend = MockBackend(host="127.0.0.1", port=9999)
    assert backend.api_host() == "127.0.0.1:9999"


def test_mock_backend_reads_external_server_from_env(monkeypatch):
    """MOCK_SERVER_HOST/MOCK_SERVER_PORT (docker-compose.yml, DESIGN.md
    4.7) must be read at construction time -- unset means "spawn our own
    in-process server on loopback" (the original, still-default
    behaviour); set means "point at the external mock-server container
    instead", so this backend's traffic traverses the real Gateway-routed
    L3 topology and netem applies. Found and fixed 2026-08-31: before
    this, every web-UI Mock run silently ignored the already-built
    mock-server/gateway topology and only ever talked to itself over
    loopback inside the `web` container."""
    monkeypatch.setenv("MOCK_SERVER_HOST", "172.28.2.3")
    monkeypatch.setenv("MOCK_SERVER_PORT", "8888")
    backend = MockBackend()
    assert backend._external_host == "172.28.2.3"
    assert backend._external_port == 8888


def test_mock_backend_no_external_server_when_env_unset(monkeypatch):
    monkeypatch.delenv("MOCK_SERVER_HOST", raising=False)
    monkeypatch.delenv("MOCK_SERVER_PORT", raising=False)
    backend = MockBackend()
    assert backend._external_host == ""
    assert backend._external_port is None


def test_mock_backend_no_external_server_when_only_host_set(monkeypatch):
    """Both env vars must be present -- a half-configured environment
    (host set, port missing or vice versa) must fall back to spawning an
    in-process server rather than trying to connect to an incomplete
    address."""
    monkeypatch.setenv("MOCK_SERVER_HOST", "172.28.2.3")
    monkeypatch.delenv("MOCK_SERVER_PORT", raising=False)
    backend = MockBackend()
    assert backend._external_port is None
