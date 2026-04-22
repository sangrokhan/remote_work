"""
BGE3 embedding provider stub.
Implements EmbeddingProvider using the BGE3 model for dense retrieval.
Replace embed() with real BGE3 API/local model call when ready.
"""
from __future__ import annotations

from langgraph_flow.core.base import EmbeddingProvider


class BGE3Provider(EmbeddingProvider):
    ENV_URL_KEY = "BGE3_API_URL"
    ENV_KEY_KEY = "BGE3_API_KEY"

    def embed(self, text: str) -> list[float]:
        # TODO: call BGE3 model and return real embedding vector
        return []
