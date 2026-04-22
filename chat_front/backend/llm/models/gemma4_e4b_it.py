from __future__ import annotations

from llm.base import BaseLLM


class Gemma4E4BIt(BaseLLM):
    def generate(self, prompt: str, context: str) -> str:
        return f"[Gemma4-E4B-it] {context[:60]}"
