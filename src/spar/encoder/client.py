from __future__ import annotations

import httpx


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
