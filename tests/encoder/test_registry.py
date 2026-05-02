import pytest
from unittest.mock import patch

from spar.encoder.client import CrossEncoderClient
from spar.encoder.config import EncoderSettings
from spar.encoder.factory import EncoderRole
from spar.encoder.registry import get_reranker, reset_registry

_TEST_SETTINGS = EncoderSettings(
    encoder_reranker_url="http://test:8002/rerank",
    encoder_reranker_model="test-reranker",
)


@pytest.fixture(autouse=True)
def clean_registry():
    reset_registry()
    yield
    reset_registry()


async def test_get_reranker_returns_client():
    with patch("spar.encoder.registry.get_settings", return_value=_TEST_SETTINGS):
        client = await get_reranker(EncoderRole.RERANKER)
    assert isinstance(client, CrossEncoderClient)
    assert client.model == "test-reranker"


async def test_get_reranker_singleton():
    with patch("spar.encoder.registry.get_settings", return_value=_TEST_SETTINGS):
        a = await get_reranker()
        b = await get_reranker()
    assert a is b


async def test_reset_registry_clears_cache():
    with patch("spar.encoder.registry.get_settings", return_value=_TEST_SETTINGS):
        a = await get_reranker()
        reset_registry()
        b = await get_reranker()
    assert a is not b
