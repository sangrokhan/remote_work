"""
Embedding provider factory for langgraph_flow.
get_embedding_provider() is the entry point for the retriever node.
"""
from __future__ import annotations

from langgraph_flow.core.base import EmbeddingProvider
from langgraph_flow.core.bge3_provider import BGE3Provider

EMBEDDING_REGISTRY: dict[str, type[EmbeddingProvider]] = {
    "bge3": BGE3Provider,
}

def get_embedding_provider(name: str = "bge3") -> EmbeddingProvider:
    cls = EMBEDDING_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown embedding provider: {name!r}. Available: {list(EMBEDDING_REGISTRY)}")
    return cls()
