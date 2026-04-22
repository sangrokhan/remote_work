"""GaussO4Think langgraph_flow stub. Simulates chain-of-thought output format."""
from __future__ import annotations

from langgraph_flow.core.base import BaseLLM


class GaussO4Think(BaseLLM):
    def generate(self, prompt: str, context: str = "") -> str:
        return f"[GaussO4-think] {prompt[:30]} → {context[:50]}"
