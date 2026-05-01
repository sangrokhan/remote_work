from __future__ import annotations

from enum import Enum

from spar.reranker.client import CrossEncoderClient, LocalCrossEncoderClient
from spar.reranker.config import EncoderSettings


class EncoderRole(str, Enum):
    RERANKER = "reranker"


class EncoderFactory:
    @staticmethod
    def create(role: EncoderRole, settings: EncoderSettings) -> CrossEncoderClient | LocalCrossEncoderClient:
        if role is EncoderRole.RERANKER:
            if settings.encoder_reranker_backend == "local":
                return LocalCrossEncoderClient(
                    model=settings.encoder_reranker_model,
                    device=settings.encoder_reranker_device,
                )
            return CrossEncoderClient(
                base_url=settings.encoder_reranker_url,
                model=settings.encoder_reranker_model,
            )
        raise ValueError(f"Unknown encoder role: {role!r}")
