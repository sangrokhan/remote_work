from __future__ import annotations

import pytest

from spar.reranker.config import EncoderSettings, get_settings, reset_settings


@pytest.fixture(autouse=True)
def clean_settings():
    reset_settings()
    yield
    reset_settings()


def test_defaults():
    s = EncoderSettings()
    assert s.encoder_reranker_model == "BAAI/bge-reranker-v2-m3"
    assert s.encoder_reranker_backend == "local"
    assert s.encoder_reranker_device == "cpu"
    assert "8002" in s.encoder_reranker_url


def test_populate_by_name():
    s = EncoderSettings(
        encoder_reranker_backend="remote",
        encoder_reranker_device="cuda",
    )
    assert s.encoder_reranker_backend == "remote"
    assert s.encoder_reranker_device == "cuda"


def test_env_alias(monkeypatch):
    monkeypatch.setenv("RERANKER_BACKEND", "remote")
    monkeypatch.setenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    s = EncoderSettings()
    assert s.encoder_reranker_backend == "remote"
    assert s.encoder_reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_get_settings_singleton():
    a = get_settings()
    b = get_settings()
    assert a is b


def test_reset_settings_breaks_singleton():
    a = get_settings()
    reset_settings()
    b = get_settings()
    assert a is not b
