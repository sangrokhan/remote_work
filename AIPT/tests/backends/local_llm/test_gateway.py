"""aipt.backends.local_llm.gateway -- the engine-gateway proxy layer,
against the fake OpenAI-compatible server (tests.backends.local_llm.fake_server)."""

from __future__ import annotations

from aipt.backends.local_llm.engine_adapter import EngineAdapter
from aipt.backends.local_llm.gateway import TRANSPORT_HEADER, Gateway

from tests.backends.local_llm.fake_server import base_url, start_fake_server


def _gateway(srv, **kw) -> Gateway:
    adapter = EngineAdapter(base_url=base_url(srv), model="m")
    return Gateway(adapter, **kw)


def test_send_returns_answer_and_wire_counts():
    srv = start_fake_server()
    try:
        gw = _gateway(srv)
        result = gw.send([{"role": "user", "content": "hello there"}])
        assert result.error == ""
        assert result.status == 200
        assert "echo: hello there" in result.text
        assert result.wire_sent > 0
        assert result.wire_recv > 0
        assert result.req_payload_bytes > 0
        assert result.resp_payload_bytes > 0
    finally:
        srv.shutdown()


def test_send_sets_transport_header_from_gateway_transport():
    srv = start_fake_server()
    try:
        gw = _gateway(srv, transport="http1")
        gw.send([{"role": "user", "content": "hi"}])
        from tests.backends.local_llm.fake_server import FakeOpenAICompatHandler
        assert FakeOpenAICompatHandler.hits[-1]["headers"].get(TRANSPORT_HEADER) == "http1"
    finally:
        srv.shutdown()


def test_on_request_hook_can_mutate_headers():
    srv = start_fake_server()
    try:
        gw = _gateway(srv)

        def add_header(req):
            req["headers"]["X-Experiment"] = "flag-1"
            return req

        gw.on_request(add_header)
        gw.send([{"role": "user", "content": "hi"}])
        from tests.backends.local_llm.fake_server import FakeOpenAICompatHandler
        assert FakeOpenAICompatHandler.hits[-1]["headers"].get("X-Experiment") == "flag-1"
    finally:
        srv.shutdown()


def test_on_request_hook_unsubscribe_stops_future_calls():
    srv = start_fake_server()
    try:
        gw = _gateway(srv)
        calls = []
        unsubscribe = gw.on_request(lambda req: calls.append(1) or None)
        gw.send([{"role": "user", "content": "hi"}])
        unsubscribe()
        gw.send([{"role": "user", "content": "hi again"}])
        assert len(calls) == 1
    finally:
        srv.shutdown()


def test_on_response_hook_can_read_and_replace_response():
    srv = start_fake_server()
    try:
        gw = _gateway(srv)
        seen = []

        def note(resp):
            seen.append(resp.get("id"))
            return resp

        gw.on_response(note)
        result = gw.send([{"role": "user", "content": "hi"}])
        assert seen == ["chatcmpl-fake-1"]
        assert result.response_body["id"] == "chatcmpl-fake-1"
    finally:
        srv.shutdown()


def test_broken_hook_does_not_crash_the_call():
    srv = start_fake_server()
    try:
        gw = _gateway(srv)
        gw.on_request(lambda req: (_ for _ in ()).throw(RuntimeError("boom")))
        result = gw.send([{"role": "user", "content": "hi"}])
        assert result.error == ""
        assert result.status == 200
    finally:
        srv.shutdown()


def test_http_error_status_is_reported_as_error():
    srv = start_fake_server(response_factory=(500, {"error": "boom"}))
    try:
        gw = _gateway(srv)
        result = gw.send([{"role": "user", "content": "hi"}])
        assert result.status == 500
        assert "http_500" in result.error
    finally:
        srv.shutdown()


def test_connection_refused_is_reported_as_request_failed():
    adapter = EngineAdapter(base_url="http://127.0.0.1:1", model="m")
    gw = Gateway(adapter, timeout=2)
    result = gw.send([{"role": "user", "content": "hi"}])
    assert result.error.startswith("request_failed:")
    assert result.status == 0
