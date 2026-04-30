import pytest

from spar.llm.config import LLMSettings, get_settings, reset_settings


def setup_function():
    reset_settings()


def test_defaults():
    s = LLMSettings()
    assert s.llm_main_url == "http://localhost:8000/v1"
    assert s.llm_main_model == "qwen2.5-72b-instruct"
    assert s.llm_main_api_key == "dummy"
    assert s.llm_router_url == "http://localhost:8001/v1"
    assert s.llm_router_model == "qwen2.5-7b-instruct"
    assert s.llm_router_api_key == "dummy"


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MAIN_URL", "http://gpu1:9000/v1")
    monkeypatch.setenv("LLM_MAIN_MODEL", "llama3-70b")
    s = LLMSettings()
    assert s.llm_main_url == "http://gpu1:9000/v1"
    assert s.llm_main_model == "llama3-70b"


def test_get_settings_singleton():
    a = get_settings()
    b = get_settings()
    assert a is b


def test_reset_settings_clears_singleton():
    a = get_settings()
    reset_settings()
    b = get_settings()
    assert a is not b
