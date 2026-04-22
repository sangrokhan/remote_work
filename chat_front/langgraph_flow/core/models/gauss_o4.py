from __future__ import annotations

from langgraph_flow.core.base import BaseLLM


class GaussO4(BaseLLM):
    def generate(self, prompt: str, context: str = "") -> str:
        return f"[GaussO4] {prompt[:30]} → {context[:50]}"
