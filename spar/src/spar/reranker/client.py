"""Async cross-encoder reranker client: calls the reranker HTTP endpoint to score query-chunk pairs."""
from __future__ import annotations

import asyncio

import httpx

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
except ImportError:  # pragma: no cover
    _CrossEncoder = None  # type: ignore


class CrossEncoderClient:
    def __init__(self, base_url: str, model: str) -> None:
        self._url = base_url
        self._model = model
        self._http = httpx.AsyncClient(timeout=30.0)

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        resp = await self._http.post(
            self._url,
            json={"model": self._model, "query": query, "documents": documents},
        )
        resp.raise_for_status()
        data = resp.json()
        results = sorted(data["results"], key=lambda r: r["index"])
        return [r["relevance_score"] for r in results]

    @property
    def model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        await self._http.aclose()


class LocalCrossEncoderClient:
    def __init__(self, model: str, device: str = "cpu") -> None:
        if _CrossEncoder is None:
            raise RuntimeError("sentence-transformers 미설치. pip install sentence-transformers")
        self._model_name = model
        self._encoder = _CrossEncoder(model, device=device)

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        scores = await asyncio.to_thread(self._encoder.predict, pairs)
        return [float(s) for s in scores]

    @property
    def model(self) -> str:
        return self._model_name

    async def aclose(self) -> None:
        pass
