"""Process-level encoder client registry; lazy-initialises and caches the configured EncoderClient."""
from __future__ import annotations

import asyncio
import os
import httpx
import numpy as np

from spar.encoder.base import EncoderClient

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover — remote mode 우선 동작에서만 허용
    SentenceTransformer = None  # type: ignore

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_encoder: EncoderClient | None = None
_lock = asyncio.Lock()


def _remote_url(raw: str) -> str:
    base = raw.rstrip("/")
    if base.endswith("/embeddings"):
        return base
    return f"{base}/embeddings"


class RemoteSentenceTransformerEncoder(EncoderClient):
    def __init__(self, model_name: str, base_url: str, api_key: str = "") -> None:
        self._model_name = model_name
        self._url = _remote_url(base_url)
        self._api_key = api_key.strip()
        self._client = httpx.Client(timeout=30.0)

    def __del__(self) -> None:  # pragma: no cover — 환경 정리에 상응
        try:
            self._client.close()
        except Exception:
            pass

    @property
    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, texts: list[str], *, normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0))
        resp = self._client.post(
            self._url,
            headers=self._headers,
            json={"model": self._model_name, "input": texts},
        )
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data")
        if not isinstance(data, list):
            data = payload.get("embeddings")
        if not isinstance(data, list):
            raise RuntimeError(
                "원격 임베딩 응답 형식이 올바르지 않습니다. "
                "'data' 또는 'embeddings' 필드가 필요합니다."
            )

        vectors: list[list[float]] = []
        if data and isinstance(data[0], dict) and "embedding" in data[0]:
            data = sorted(data, key=lambda x: x.get("index", 0))
            for item in data:
                if not isinstance(item, dict) or "embedding" not in item:
                    raise RuntimeError("원격 임베딩 응답 data 항목이 비정상입니다.")
                vectors.append(list(item["embedding"]))
        else:
            for item in data:
                if not isinstance(item, list):
                    raise RuntimeError("원격 임베딩 응답 벡터 형식이 비정상입니다.")
                vectors.append(list(item))

        arr = np.array(vectors, dtype=float)
        if normalize:
            if arr.size:
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms = np.where(norms == 0.0, 1.0, norms)
                arr = arr / norms
        return arr


class SentenceTransformerEncoder(EncoderClient):
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._model_name = model_name
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers 미설치. requirements.txt 확인."
            )
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
            remote_url = (
                os.getenv("ENCODER_URL", "").strip()
                or os.getenv("EMBEDDING_URL", "").strip()
            )
            if remote_url:
                api_key = (
                    os.getenv("ENCODER_API_KEY", "").strip()
                    or os.getenv("EMBEDDING_API_KEY", "").strip()
                )
                _encoder = RemoteSentenceTransformerEncoder(
                    model_name=os.getenv("ENCODER_MODEL", _DEFAULT_MODEL),
                    base_url=remote_url,
                    api_key=api_key,
                )
                return _encoder

            model = os.getenv("ENCODER_MODEL", _DEFAULT_MODEL)
            device = os.getenv("ENCODER_DEVICE", "cpu")
            _encoder = SentenceTransformerEncoder(model_name=model, device=device)
    return _encoder


def reset_registry() -> None:
    global _encoder
    _encoder = None
