"""Dense embedder — sentence-transformers 래퍼."""

from __future__ import annotations

import os
import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer  # type: ignore

# 런타임 import는 클래스 안에서 — 테스트 monkeypatch 가능하게 모듈 attr로 노출
try:
    from sentence_transformers import SentenceTransformer  # noqa: F811
except ImportError:  # pragma: no cover — runtime 미설치 환경
    SentenceTransformer = None  # type: ignore


DEFAULT_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5")


def _normalize_vector(vector: Iterable[float]) -> list[float]:
    """벡터를 L2 정규화한다. norm이 0이면 원본을 그대로 반환한다."""
    values = list(vector)
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 0:
        return values
    inv = 1.0 / norm
    return [float(v * inv) for v in values]


def _remote_embedding_url(raw_url: str) -> str:
    """Embedding API 엔드포인트를 정규화한다."""
    base = raw_url.rstrip("/")
    if base.endswith("/embeddings"):
        return base
    return f"{base}/embeddings"


class Embedder:
    """단일 모델 dense embedder. 코사인 유사도용 정규화 강제."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto") -> None:
        self.model_name = model_name
        self._remote_url = os.getenv("EMBEDDING_URL", "").strip()
        self._remote_api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
        self._http_client = None

        if self._remote_url:
            self._remote_url = _remote_embedding_url(self._remote_url)
            self._http_client = httpx.Client(timeout=30.0)
            return

        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers 미설치. requirements.txt 확인."
            )
        kwargs = {} if device == "auto" else {"device": device}
        self._model = SentenceTransformer(model_name, **kwargs)

    def __del__(self) -> None:  # pragma: no cover — 실행 종료 정리에 대응
        if self._http_client is not None:
            try:
                self._http_client.close()
            except Exception:
                pass

    @property
    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._remote_api_key:
            headers["Authorization"] = f"Bearer {self._remote_api_key}"
        return headers

    def _request_remote(self, texts: list[str]) -> list[list[float]]:
        if self._http_client is None:
            raise RuntimeError("Remote embedding client is not initialized.")
        response = self._http_client.post(
            self._remote_url,
            headers=self._headers,
            json={"model": self.model_name, "input": texts},
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list):
            data = payload.get("embeddings")
        if not isinstance(data, list):
            raise RuntimeError(
                "원격 임베딩 응답 형식이 올바르지 않습니다. "
                "'data' 또는 'embeddings' 필드를 확인하세요."
            )

        vectors: list[list[float]] = []
        if data and isinstance(data[0], dict) and "embedding" in data[0]:
            data = sorted(data, key=lambda x: x.get("index", 0))
            for item in data:
                if not isinstance(item, dict) or "embedding" not in item:
                    raise RuntimeError("원격 임베딩 응답 data 항목이 비정상입니다.")
                vectors.append(_normalize_vector(item["embedding"]))
        else:
            for item in data:
                if not isinstance(item, list):
                    raise RuntimeError("원격 임베딩 응답 벡터 형식이 비정상입니다.")
                vectors.append(_normalize_vector(item))
        if len(vectors) != len(texts):
            raise RuntimeError("원격 임베딩 응답 개수와 입력 개수가 다릅니다.")
        return vectors

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        if self._remote_url:
            return self._request_remote(texts)
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # numpy → list (Milvus 호환)
        return [list(map(float, v)) for v in vecs]
