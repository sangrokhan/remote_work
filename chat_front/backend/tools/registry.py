from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    @abstractmethod
    async def ainvoke(self, input_data: Dict[str, Any]) -> Any:
        ...


class RetrieverTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="retriever",
            description="Retrieves relevant documents for a given query",
        )

    async def ainvoke(self, input_data: Dict[str, Any]) -> Any:
        query = input_data.get("query", "")
        return f"[retriever] {query}"
