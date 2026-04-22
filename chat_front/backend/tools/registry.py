from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTool(ABC):
    @abstractmethod
    def run(self, query: str) -> str:
        ...


class RetrieverTool(BaseTool):
    def run(self, query: str) -> str:
        return f"[retriever] {query}"
