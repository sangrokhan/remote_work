"""Live coverage requiring an actual llama.cpp `llama-server` / vLLM
OpenAI-compatible server reachable at LOCAL_LLM_ENGINE_URL (default
http://127.0.0.1:8080). Not run by default -- see pyproject.toml's `live`
marker and `pytest tests/ -q -m "not live"`.

Standing up a real engine (downloading a GGUF, `llama-server --model ...`,
or `vllm serve ...`) is out of scope for this change (task instructions);
this test only documents the contract a real engine must satisfy for
LocalLLMBackend to work end to end, so it exists ready to run the moment
someone points LOCAL_LLM_ENGINE_URL at a real server.
"""

from __future__ import annotations

import pytest

from aipt.backends.local_llm import LocalLLMBackend

pytestmark = pytest.mark.live


def test_local_llm_backend_against_real_engine():
    backend = LocalLLMBackend()
    ok, reason = backend.ready()
    assert ok, reason

    backend.connect(arm="chat", model=backend.DEFAULT_MODEL, system="You are concise.")
    try:
        exchange = backend.send_turn(turn=1, question="Say hello in one word.",
                                      measure="bytes")
        assert exchange.error is None
        assert exchange.text
        assert exchange.wire_sent > 0
        assert exchange.wire_recv > 0
    finally:
        backend.close()
