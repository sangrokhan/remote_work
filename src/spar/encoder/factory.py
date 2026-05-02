from __future__ import annotations

from enum import Enum

from spar.encoder.client import CrossEncoderClient
from spar.encoder.config import EncoderSettings


class EncoderRole(str, Enum):
    RERANKER = "reranker"


class EncoderFactory:
    @staticmethod
    def create(role: EncoderRole, settings: EncoderSettings) -> CrossEncoderClient:
        if role is EncoderRole.RERANKER:
            return CrossEncoderClient(
                base_url=settings.encoder_reranker_url,
                model=settings.encoder_reranker_model,
            )
        raise ValueError(f"Unknown encoder role: {role!r}")
