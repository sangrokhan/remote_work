"""Gemma4-E4B-it langgraph_flow stub. Instruction-tuned 4B model placeholder."""
from __future__ import annotations

from langgraph_flow.core.base import BaseLLM


class Gemma4E4BIt(BaseLLM):
    ENV_URL_KEY = "GEMMA4_E4B_IT_API_URL"
    ENV_KEY_KEY = "GEMMA4_E4B_IT_API_KEY"

    def generate(self, prompt: str, context: str = "") -> str:
        return f"[Gemma4-E4B-it] {prompt[:30]} → {context[:50]}"
