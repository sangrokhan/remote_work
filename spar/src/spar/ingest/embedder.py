"""Dense embedder — sentence-transformers 래퍼."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer  # type: ignore

# 런타임 import는 클래스 안에서 — 테스트 monkeypatch 가능하게 모듈 attr로 노출
try:
    from sentence_transformers import SentenceTransformer  # noqa: F811
except ImportError:  # pragma: no cover — runtime 미설치 환경
    SentenceTransformer = None  # type: ignore


DEFAULT_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5")


class Embedder:
    """단일 모델 dense embedder. 코사인 유사도용 정규화 강제."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto") -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers 미설치. requirements.txt 확인."
            )
        kwargs = {} if device == "auto" else {"device": device}
        self._model = SentenceTransformer(model_name, **kwargs)
        self.model_name = model_name

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # numpy → list (Milvus 호환)
        return [list(map(float, v)) for v in vecs]
