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
