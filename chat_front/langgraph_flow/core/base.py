"""
Abstract base class for LLM implementations in langgraph_flow.
Nodes import this via config["configurable"]["llm"]; never instantiated directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, context: str = "") -> str:
        ...
