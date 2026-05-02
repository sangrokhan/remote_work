from spar.encoder.config import EncoderSettings, get_settings, reset_settings


def test_default_values():
    s = EncoderSettings()
    assert s.encoder_reranker_url == "http://localhost:8002/rerank"
    assert s.encoder_reranker_model == "BAAI/bge-reranker-v2-m3"


def test_get_settings_singleton():
    reset_settings()
    a = get_settings()
    b = get_settings()
    assert a is b


def test_reset_settings_clears_cache():
    reset_settings()
    a = get_settings()
    reset_settings()
    b = get_settings()
    assert a is not b
