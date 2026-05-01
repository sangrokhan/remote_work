from __future__ import annotations

import pytest
from unittest.mock import patch

from spar.encoder.client import SentenceTransformerEncoder
from spar.encoder.config import EncoderSettings
from spar.encoder.factory import EncoderFactory, EncoderProvider


@pytest.fixture
def settings():
    return EncoderSettings(
        encoder_provider="sentence_transformers",
        encoder_model="BAAI/bge-small-en-v1.5",
        encoder_device="cpu",
    )


def test_provider_enum_values():
    assert EncoderProvider.SENTENCE_TRANSFORMERS.value == "sentence_transformers"


def test_create_sentence_transformers(settings):
    with patch("spar.encoder.client.SentenceTransformer"):
        encoder = EncoderFactory.create(settings)
    assert isinstance(encoder, SentenceTransformerEncoder)
    assert encoder.model_name == "BAAI/bge-small-en-v1.5"


def test_unknown_provider_raises():
    bad = EncoderSettings(encoder_provider="unknown_provider")
    with pytest.raises(ValueError, match="Unknown encoder provider"):
        EncoderFactory.create(bad)
