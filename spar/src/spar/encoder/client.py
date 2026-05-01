from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from spar.encoder.base import EncoderClient


class SentenceTransformerEncoder(EncoderClient):
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], *, normalize: bool = True) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=normalize)

    @property
    def model_name(self) -> str:
        return self._model_name
