"""aipt.backends.local_llm.engine_adapter -- request/response shaping,
against no server at all (pure unit-level, no sockets)."""

from __future__ import annotations

import os

from aipt.backends.local_llm.engine_adapter import (
    DEFAULT_ENGINE_URL,
    DEFAULT_MODEL,
    EngineAdapter,
    api_key,
    default_model,
    engine_kind,
    engine_url,
    ready,
)


def test_engine_url_defaults_and_env_override(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_ENGINE_URL", raising=False)
    assert engine_url() == DEFAULT_ENGINE_URL

    monkeypatch.setenv("LOCAL_LLM_ENGINE_URL", "http://example:9999/")
    assert engine_url() == "http://example:9999"


def test_engine_kind_defaults_to_llama_cpp_and_rejects_unknown(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_ENGINE_KIND", raising=False)
    assert engine_kind() == "llama_cpp"

    monkeypatch.setenv("LOCAL_LLM_ENGINE_KIND", "vllm")
    assert engine_kind() == "vllm"

    monkeypatch.setenv("LOCAL_LLM_ENGINE_KIND", "not_a_real_engine")
    assert engine_kind() == "llama_cpp"


def test_default_model_and_api_key_env(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
    assert default_model() == DEFAULT_MODEL
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen-local")
    assert default_model() == "qwen-local"

    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    assert api_key() == ""
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "sekret")
    assert api_key() == "sekret"


def test_ready_is_true_by_default_with_reason():
    ok, reason = ready()
    assert ok
    assert reason


def test_ready_reflects_configured_url_and_kind():
    ok, reason = ready()
    assert ok
    assert engine_url() in reason
    assert engine_kind() in reason


def test_adapter_defaults_from_env(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_ENGINE_URL", "http://engine.local:8080")
    monkeypatch.setenv("LOCAL_LLM_ENGINE_KIND", "vllm")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "my-model")
    adapter = EngineAdapter()
    assert adapter.base_url == "http://engine.local:8080"
    assert adapter.kind == "vllm"
    assert adapter.model == "my-model"


def test_adapter_explicit_args_override_env(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_ENGINE_URL", "http://ignored:1")
    adapter = EngineAdapter(base_url="http://explicit:2/", kind="llama_cpp", model="m")
    assert adapter.base_url == "http://explicit:2"
    assert adapter.chat_completions_url() == "http://explicit:2/v1/chat/completions"


def test_headers_include_bearer_only_when_key_set(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    adapter = EngineAdapter(base_url="http://x")
    assert "Authorization" not in adapter.headers()

    adapter2 = EngineAdapter(base_url="http://x", api_key_value="abc")
    assert adapter2.headers()["Authorization"] == "Bearer abc"


def test_build_body_shape_and_extra_merge():
    adapter = EngineAdapter(base_url="http://x", model="m")
    body = adapter.build_body(
        [{"role": "user", "content": "hi"}],
        temperature=0.2, max_tokens=64, extra={"top_p": 0.9},
    )
    assert body["model"] == "m"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["stream"] is False
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 64
    assert body["top_p"] == 0.9


def test_build_body_extra_body_from_construction_merges_too():
    adapter = EngineAdapter(base_url="http://x", extra_body={"seed": 7})
    body = adapter.build_body([{"role": "user", "content": "hi"}])
    assert body["seed"] == 7


def test_text_of_blocking_message_shape():
    body = {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}
    assert EngineAdapter.text_of(body) == "hello"


def test_text_of_streamed_delta_shape():
    chunk = {"choices": [{"delta": {"content": "partial"}}]}
    assert EngineAdapter.text_of(chunk) == "partial"


def test_text_of_legacy_text_field_shape():
    chunk = {"choices": [{"text": "legacy completion"}]}
    assert EngineAdapter.text_of(chunk) == "legacy completion"


def test_text_of_no_choices_is_empty_string():
    assert EngineAdapter.text_of({}) == ""
    assert EngineAdapter.text_of({"choices": []}) == ""
    assert EngineAdapter.text_of("not a dict") == ""  # defensive: never raises


def test_usage_of_maps_openai_fields_to_backend_neutral_names():
    body = {"usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}
    usage = EngineAdapter.usage_of(body)
    assert usage == {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}


def test_usage_of_missing_usage_defaults_to_zeros():
    assert EngineAdapter.usage_of({}) == {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
    }
