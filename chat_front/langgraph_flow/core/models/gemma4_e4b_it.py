from __future__ import annotations

from langgraph_flow.core.base import BaseLLM


class Gemma4E4BIt(BaseLLM):
    def generate(self, prompt: str, context: str = "") -> str:
        return f"[Gemma4-E4B-it] {prompt[:30]} → {context[:50]}"
