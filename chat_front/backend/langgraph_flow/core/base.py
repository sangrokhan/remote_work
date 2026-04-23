"""
Abstract base class for all embedding providers in langgraph_flow.
Used by the retriever node for agentic RAG document retrieval.
Concrete providers declare ENV_URL_KEY / ENV_KEY_KEY to load config from .env.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    ENV_URL_KEY: str = ""
    ENV_KEY_KEY: str = ""

    def __init__(self, api_url: str = "", api_key: str = "") -> None:
        self.api_url = api_url or os.getenv(self.ENV_URL_KEY, "")
        self.api_key = api_key or os.getenv(self.ENV_KEY_KEY, "")

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...
