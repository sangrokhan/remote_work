from __future__ import annotations

from enum import Enum

from spar.encoder.base import EncoderClient
from spar.encoder.client import SentenceTransformerEncoder
from spar.encoder.config import EncoderSettings


class EncoderProvider(str, Enum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class EncoderFactory:
    @staticmethod
    def create(settings: EncoderSettings) -> EncoderClient:
        if settings.encoder_provider == EncoderProvider.SENTENCE_TRANSFORMERS:
            return SentenceTransformerEncoder(
                model_name=settings.encoder_model,
                device=settings.encoder_device,
            )
        raise ValueError(f"Unknown encoder provider: {settings.encoder_provider!r}")
