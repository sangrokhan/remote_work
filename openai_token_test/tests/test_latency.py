"""TTFT and TTLT, and the compromise that measuring them forces.

TTFT only exists in a stream. But streaming changes what comes back on the wire:
SSE framing per chunk, the chat envelope (id/model/created) repeated on every
delta, and on the Responses API the whole Response object shipped again inside
response.created and response.completed. So a streamed run cannot be used to
compare DOWNLOAD bytes against a non-streamed one.

What it can be used for, and what these tests pin:
  - upload bytes stay comparable — streaming only adds the stream flags (~66 B)
  - obfuscation must be OFF, or the SSE deltas carry random padding and even the
    download figure is a fiction
  - TTFT is timed off the first CONTENT token, not the role chunk
  - TTLT is timed off finish_reason / response.completed, not the usage chunk,
    which the chat endpoint sends afterwards
"""

from __future__ import annotations

import pytest

import experiment
import openai_client as oc
import wire
from fake_openai import FIRST_TOKEN_DELAY, FakeOpenAI


@pytest.fixture
def fake(monkeypatch):
    srv = FakeOpenAI()
    monkeypatch.setenv("OPENAI_BASE_URL", srv.start())
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    wire.reset_session()
    yield srv
    srv.stop()
    wire.reset_session()


def _call(arm, fake, **kw):
    fx = experiment.fixture_mod.load("perf")
    conv = None
    if arm == "responses_stateful":
        conv, _ = oc.create_conversation(fx.system)
    return oc.call(arm, model="test-model", system=fx.system, history=[],
                   question=fx.steps[0], turn=1, conversation=conv,
                   cache_key="k", **kw)


@pytest.mark.parametrize("arm", oc.ARMS)
def test_streaming_records_ttft_and_ttlt(fake, arm):
    res = _call(arm, fake, stream=True)

    assert res.text, "a streamed call must still assemble the completion text"
    assert res.input_tokens > 0, "usage must be captured from the stream"

    # the fake stalls before the first token; TTFT must see that stall, and TTLT
    # must land after it
    assert res.ttft_ms >= FIRST_TOKEN_DELAY * 1000 * 0.7, res.ttft_ms
    assert res.ttlt_ms >= res.ttft_ms
    assert res.ttlt_ms <= res.latency_ms + 5


@pytest.mark.parametrize("arm", oc.ARMS)
def test_non_streaming_leaves_ttft_unset(fake, arm):
    """Total latency is not TTFT. A non-streamed call never saw a first token, and
    must not pretend it did — a zero here is honest, a copy of latency would be a
    lie that quietly becomes a chart."""
    res = _call(arm, fake, stream=False)
    assert res.ttft_ms == 0
    assert res.ttlt_ms == 0
    assert res.latency_ms > 0


@pytest.mark.parametrize("arm", oc.ARMS)
def test_streaming_disables_obfuscation_padding(fake, arm):
    """Left on, the deltas carry random padding to normalize payload sizes, and
    every byte we report downstream is noise."""
    _call(arm, fake, stream=True)
    body = [r["body"] for r in fake.requests
            if not r["path"].endswith("/conversations")][-1]
    assert body["stream"] is True
    assert body["stream_options"]["include_obfuscation"] is False
    if arm == "chat_stateless":
        # the chat endpoint withholds usage from a stream unless asked
        assert body["stream_options"]["include_usage"] is True


@pytest.mark.parametrize("arm", oc.ARMS)
def test_streaming_barely_changes_the_upload(fake, arm):
    """The thesis is about upload bytes. Streaming must not disturb them, or the
    latency run and the byte run stop being the same experiment."""
    plain = _call(arm, fake, stream=False)
    fake.requests.clear()
    streamed = _call(arm, fake, stream=True)

    grew = streamed.req_payload_bytes - plain.req_payload_bytes
    assert 0 < grew < 120, f"stream flags should cost ~66 B, not {grew}"


def test_streamed_usage_matches_the_non_streamed_call(fake):
    plain = _call("chat_stateless", fake, stream=False)
    streamed = _call("chat_stateless", fake, stream=True)
    assert streamed.input_tokens == plain.input_tokens
    assert streamed.cached_tokens == plain.cached_tokens
