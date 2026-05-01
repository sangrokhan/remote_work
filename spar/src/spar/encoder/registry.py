from __future__ import annotations

import asyncio
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from spar.encoder.base import EncoderClient

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

_encoder: EncoderClient | None = None
_lock = asyncio.Lock()


class SentenceTransformerEncoder(EncoderClient):
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], *, normalize: bool = True) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=normalize)

    @property
    def model_name(self) -> str:
        return self._model_name


async def get_encoder() -> EncoderClient:
    global _encoder
    if _encoder is not None:
        return _encoder
    async with _lock:
        if _encoder is None:
            model = os.getenv("ENCODER_MODEL", _DEFAULT_MODEL)
            device = os.getenv("ENCODER_DEVICE", "cpu")
            _encoder = SentenceTransformerEncoder(model_name=model, device=device)
    return _encoder


def reset_registry() -> None:
    global _encoder
    _encoder = None
