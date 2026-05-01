from __future__ import annotations

import pytest
from unittest.mock import patch

from spar.encoder.base import EncoderClient
from spar.encoder.config import EncoderSettings
from spar.encoder.registry import get_encoder, reset_registry

_TEST_SETTINGS = EncoderSettings(
    encoder_provider="sentence_transformers",
    encoder_model="BAAI/bge-small-en-v1.5",
    encoder_device="cpu",
)


@pytest.fixture(autouse=True)
def clean_registry():
    reset_registry()
    yield
    reset_registry()


async def test_get_encoder_returns_encoder_client():
    with patch("spar.encoder.client.SentenceTransformer"), \
         patch("spar.encoder.registry.get_settings", return_value=_TEST_SETTINGS):
        encoder = await get_encoder()
    assert isinstance(encoder, EncoderClient)


async def test_get_encoder_singleton():
    with patch("spar.encoder.client.SentenceTransformer"), \
         patch("spar.encoder.registry.get_settings", return_value=_TEST_SETTINGS):
        a = await get_encoder()
        b = await get_encoder()
    assert a is b


async def test_reset_clears_singleton():
    with patch("spar.encoder.client.SentenceTransformer"), \
         patch("spar.encoder.registry.get_settings", return_value=_TEST_SETTINGS):
        a = await get_encoder()
        reset_registry()
        b = await get_encoder()
    assert a is not b
