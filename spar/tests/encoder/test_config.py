from __future__ import annotations

from spar.encoder.config import EncoderSettings, get_settings, reset_settings


def test_defaults():
    s = EncoderSettings()
    assert s.encoder_provider == "sentence_transformers"
    assert s.encoder_model == "BAAI/bge-small-en-v1.5"
    assert s.encoder_device == "cpu"


def test_get_settings_singleton():
    reset_settings()
    a = get_settings()
    b = get_settings()
    assert a is b


def test_reset_settings():
    reset_settings()
    a = get_settings()
    reset_settings()
    b = get_settings()
    assert a is not b
