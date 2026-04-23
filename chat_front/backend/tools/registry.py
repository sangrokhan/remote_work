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


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    async def execute(self, name: str, input_data: Dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not registered")
        return await tool.ainvoke(input_data)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())


class RetrieverTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="retriever",
            description="Retrieves relevant documents for a given query",
        )

    async def ainvoke(self, input_data: Dict[str, Any]) -> Any:
        query = input_data.get("query", "")
        return f"[retriever] {query}"
