"""aipt.backends.local_llm.LocalLLMBackend -- the Backend protocol
implementation, against the fake OpenAI-compatible server. No real
llama.cpp/vLLM process involved -- see tests.backends.local_llm.fake_server
and test_engine_live.py (@pytest.mark.live) for that boundary."""

from __future__ import annotations

from aipt.backends.base import Backend
from aipt.backends.local_llm import LocalLLMBackend

from tests.backends.local_llm.fake_server import base_url, start_fake_server


def _backend(srv, **kw) -> LocalLLMBackend:
    return LocalLLMBackend(engine_url=base_url(srv), model="m", **kw)


def test_local_llm_backend_satisfies_protocol_uninitialized():
    backend = LocalLLMBackend()
    assert isinstance(backend, Backend)
    assert backend.NAME == "local_llm"
    assert backend.transport == "http1"
    assert "chat" in backend.ARMS


def test_ready_reports_configured_engine_url():
    backend = LocalLLMBackend(engine_url="http://127.0.0.1:8080")
    ok, reason = backend.ready()
    assert ok
    assert "8080" in reason


def test_api_host_before_connect_is_configured_engine_url():
    backend = LocalLLMBackend(engine_url="http://engine.local:9000")
    assert backend.api_host() == "http://engine.local:9000"


def test_connect_rejects_unknown_arm():
    backend = LocalLLMBackend()
    try:
        backend.connect(arm="not_a_real_arm", model="m", system="")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "not_a_real_arm" in str(exc)


def test_send_turn_before_connect_raises():
    backend = LocalLLMBackend()
    try:
        backend.send_turn(turn=1, question="hi", measure="bytes")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_full_lifecycle_one_turn():
    srv = start_fake_server()
    try:
        backend = _backend(srv)
        backend.connect(arm="chat", model="m", system="you are helpful")
        exchange = backend.send_turn(turn=1, question="What is 2+2?", measure="bytes")
        backend.close()

        assert exchange.error is None
        assert "echo:" in exchange.text
        assert exchange.wire_sent > 0
        assert exchange.wire_recv > 0
        assert exchange.req_payload_bytes > 0
        assert exchange.resp_payload_bytes > 0
        assert exchange.turn_end_ms >= 0
        assert exchange.request_json["model"] == "m"
        assert exchange.response_json["id"] == "chatcmpl-fake-1"
    finally:
        srv.shutdown()


def test_multi_turn_history_accumulates_across_turns():
    srv = start_fake_server()
    try:
        backend = _backend(srv)
        backend.connect(arm="chat", model="m", system="sys prompt")
        backend.send_turn(turn=1, question="first question", measure="bytes")
        backend.send_turn(turn=2, question="second question", measure="bytes")
        backend.close()

        from tests.backends.local_llm.fake_server import FakeOpenAICompatHandler
        second_call_body = FakeOpenAICompatHandler.hits[-1]["body"]
        roles = [m["role"] for m in second_call_body["messages"]]
        # system + user1 + assistant1 + user2
        assert roles == ["system", "user", "assistant", "user"]
    finally:
        srv.shutdown()


def test_progress_callback_fires_before_call():
    srv = start_fake_server()
    try:
        backend = _backend(srv)
        backend.connect(arm="chat", model="m", system="")
        events = []
        backend.send_turn(turn=1, question="hi", measure="bytes",
                           on_progress=events.append)
        backend.close()
        assert events and events[0]["backend"] == "local_llm"
        assert events[0]["phase"] == "steady"
    finally:
        srv.shutdown()


def test_close_is_safe_to_call_without_connect():
    backend = LocalLLMBackend()
    backend.close()  # must not raise


def test_transport_header_reaches_the_engine_request():
    srv = start_fake_server()
    try:
        backend = _backend(srv, transport="http1")
        backend.connect(arm="chat", model="m", system="")
        backend.send_turn(turn=1, question="hi", measure="bytes")
        backend.close()

        from tests.backends.local_llm.fake_server import FakeOpenAICompatHandler
        from aipt.backends.local_llm.gateway import TRANSPORT_HEADER
        assert FakeOpenAICompatHandler.hits[-1]["headers"].get(TRANSPORT_HEADER) == "http1"
    finally:
        srv.shutdown()


def test_cwnd_result_available_after_close():
    srv = start_fake_server()
    try:
        backend = _backend(srv)
        backend.connect(arm="chat", model="m", system="")
        backend.send_turn(turn=1, question="hi", measure="bytes")
        backend.close()
        result = backend.cwnd_result()
        # Best-effort: on a box without the native helper this is still a
        # dict (possibly with an error explaining why), never an exception.
        assert isinstance(result, dict)
    finally:
        srv.shutdown()


def test_error_turn_still_returns_an_exchange_not_an_exception():
    backend = LocalLLMBackend(engine_url="http://127.0.0.1:1", model="m", timeout=2)
    backend.connect(arm="chat", model="m", system="")
    exchange = backend.send_turn(turn=1, question="hi", measure="bytes")
    backend.close()
    assert exchange.error is not None
    assert exchange.text == ""
