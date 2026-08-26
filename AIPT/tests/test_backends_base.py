"""Smoke tests for aipt.backends.base -- protocol shape, not real backends.

Verifies:
  * a minimal dummy object satisfies the Backend Protocol structurally
    (runtime_checkable isinstance check);
  * the transport slot defaults to "http1" and accepts the reserved
    "http3" literal without erroring at the type level;
  * aipt.backends.record.turn_record() produces the expected columns,
    including the new backend-neutral ``backend``/``transport``/
    ``goodput_bps`` fields;
  * aipt.backends.get()/names() resolve the three registered backend
    packages (currently NotImplementedError stubs) and reject unknown
    names, mirroring token_traffic.providers.base's get()/names().
"""

from __future__ import annotations

import pytest

from aipt.backends import get, names
from aipt.backends.base import Backend, DEFAULT_TRANSPORT, progress
from aipt.backends.record import Exchange, turn_record


class DummyBackend:
    """The smallest object that satisfies aipt.backends.base.Backend."""

    NAME = "dummy"
    DEFAULT_MODEL = "dummy-model"
    ARMS = ("baseline",)
    HEADLINE_ARMS = ("baseline",)
    transport = "http1"

    def __init__(self) -> None:
        self._connected = False
        self._turns = 0

    def ready(self) -> tuple[bool, str]:
        return True, "ok"

    def api_host(self) -> str:
        return "127.0.0.1:0"

    def connect(self, arm: str, model: str, system: str) -> None:
        self._connected = True

    def send_turn(self, turn: int, question: str, measure: str, on_progress=None):
        assert self._connected, "send_turn called before connect"
        self._turns += 1
        return Exchange(
            wire_sent=100,
            wire_recv=200,
            req_payload_bytes=90,
            resp_payload_bytes=180,
            req_sent_ms=0,
            ttfb_ms=10,
            ttft_ms=20,
            ttlt_ms=50,
            turn_end_ms=60,
            text="hello",
            request_json={"q": question},
            response_json={"a": "hello"},
            error=None,
        )

    def close(self) -> None:
        self._connected = False


def test_dummy_backend_satisfies_protocol():
    backend = DummyBackend()
    assert isinstance(backend, Backend)


def test_lifecycle_connect_send_turn_close():
    backend = DummyBackend()
    ok, reason = backend.ready()
    assert ok
    assert reason

    backend.connect(arm="baseline", model="dummy-model", system="")
    exchange = backend.send_turn(turn=1, question="hi?", measure="bytes")
    backend.close()

    assert exchange.text == "hello"
    assert backend._turns == 1
    assert backend._connected is False


def test_transport_slot_defaults_and_accepts_reserved_value():
    assert DEFAULT_TRANSPORT == "http1"

    backend = DummyBackend()
    assert backend.transport == "http1"

    # The slot is a plain attribute -- nothing stops assigning the
    # reserved-for-later value; no backend has to *do* anything with it yet.
    backend.transport = "http3"
    assert backend.transport == "http3"


def test_progress_emits_expected_event_shape():
    events = []
    progress(events.append, backend="dummy", arm="baseline", phase="steady",
              turn=2, turns=5)
    assert events == [
        {
            "backend": "dummy",
            "arm": "baseline",
            "phase": "steady",
            "turn": 2,
            "turns": 5,
        }
    ]

    # on_progress=None must be a silent no-op, matching token_traffic's
    # providers.base.progress().
    progress(None, backend="dummy", arm="baseline", phase="steady", turn=1, turns=1)


def test_turn_record_has_backend_neutral_columns():
    exchange = Exchange(
        wire_sent=1000,
        wire_recv=2000,
        req_payload_bytes=900,
        resp_payload_bytes=1800,
        req_sent_ms=0,
        ttfb_ms=100,
        ttft_ms=150,
        ttlt_ms=400,
        turn_end_ms=420,
        text="answer",
        request_json={"prompt": "q"},
        response_json={"text": "answer"},
        error=None,
    )
    usage = {"input_tokens": 10, "output_tokens": 5, "cached_tokens": 2}

    record = turn_record(
        backend="mock",
        arm="baseline",
        phase="steady",
        turn=1,
        question="q",
        measure="bytes",
        exchange=exchange,
        usage=usage,
    )

    assert record["backend"] == "mock"
    assert record["arm"] == "baseline"
    assert record["transport"] == "http1"
    assert record["wire_sent"] == 1000
    assert record["wire_recv"] == 2000
    assert record["input_tokens"] == 10
    assert record["output_tokens"] == 5
    assert record["total_tokens"] == 15
    assert record["store_tail_ms"] == max(0, 420 - 400)
    assert "goodput_bps" in record
    assert "schema_version" in record


def test_turn_record_transport_override():
    exchange = Exchange(ttlt_ms=0, turn_end_ms=0)
    record = turn_record(
        backend="local_llm",
        arm="baseline",
        phase="steady",
        turn=1,
        question="q",
        measure="bytes",
        exchange=exchange,
        usage={},
        transport="http3",
    )
    assert record["transport"] == "http3"
    # ttlt_ms <= 0 means no last-token mark was taken -> store_tail is 0,
    # not a negative or nonsensical wait.
    assert record["store_tail_ms"] == 0


def test_backend_registry_names_and_get():
    assert set(names()) == {"public_ai", "mock", "local_llm"}

    for name in names():
        module = get(name)
        if not hasattr(module, "NotImplementedBackend"):
            # aipt.backends.mock has landed its real implementation
            # (DESIGN.md 5, A3/B1/B3) -- it no longer carries the
            # NotImplementedError stub, so it is excluded from the
            # "still a placeholder" assertion below instead of the whole
            # test failing as each backend graduates independently.
            continue
        # Each remaining package is still a NotImplementedError stub --
        # resolving the module must succeed, but instantiating the
        # placeholder must fail loudly rather than silently pretending to
        # work.
        with pytest.raises(NotImplementedError):
            module.NotImplementedBackend()


def test_mock_backend_is_implemented_and_satisfies_protocol():
    """aipt.backends.mock has graduated from the NotImplementedError stub
    (DESIGN.md 5, A3/B1/B3) -- get("mock") resolves a real MockBackend that
    structurally satisfies the Backend protocol, unlike the other two
    backends which (as of this test) are still placeholders."""
    module = get("mock")
    assert hasattr(module, "MockBackend")
    backend = module.MockBackend()
    assert isinstance(backend, Backend)


def test_public_ai_backend_is_implemented_and_satisfies_protocol():
    """aipt.backends.public_ai has graduated from the NotImplementedError
    stub (DESIGN.md 5, A2/B2) -- get("public_ai") resolves a real
    PublicAIBackend (facade over GeminiBackend/OpenAIBackend) that
    structurally satisfies the Backend protocol."""
    module = get("public_ai")
    assert hasattr(module, "PublicAIBackend")
    for engine in ("gemini", "openai"):
        backend = module.PublicAIBackend(engine=engine)
        assert isinstance(backend, Backend)


def test_local_llm_backend_is_implemented_and_satisfies_protocol():
    """aipt.backends.local_llm has graduated from the NotImplementedError
    stub (DESIGN.md 5, B4) -- get("local_llm") resolves a real
    LocalLLMBackend (engine_adapter + gateway) that structurally satisfies
    the Backend protocol, same as mock/public_ai."""
    module = get("local_llm")
    assert hasattr(module, "LocalLLMBackend")
    backend = module.LocalLLMBackend()
    assert isinstance(backend, Backend)


def test_backend_registry_rejects_unknown_name():
    with pytest.raises(KeyError):
        get("not_a_real_backend")
