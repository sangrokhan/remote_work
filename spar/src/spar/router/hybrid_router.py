from __future__ import annotations

from spar.router.embedding_router import EmbeddingRouter
from spar.router.llm_router import LLMRouter
from spar.router.regex_router import RegexRouter
from spar.router.schemas import Route, RouteResult

_REGEX_THRESHOLD = 0.9


class HybridRouter:
    """3-layer query router: Regex → Embedding → LLM."""

    def __init__(
        self,
        embedding_threshold: float = 0.65,
        use_llm: bool = True,
    ) -> None:
        self._regex = RegexRouter()
        self._embedding = EmbeddingRouter(threshold=embedding_threshold)
        self._llm_router = LLMRouter()
        self._use_llm = use_llm

    async def route(self, query: str) -> RouteResult:
        result = self._regex.route(query)
        if result and result.confidence >= _REGEX_THRESHOLD:
            return result

        result = self._embedding.route(query)
        if result is not None:
            return result

        if self._use_llm:
            return await self._llm_router.route(query)

        return RouteResult(route=Route.DEFAULT_RAG, confidence=0.0, layer="fallback")
