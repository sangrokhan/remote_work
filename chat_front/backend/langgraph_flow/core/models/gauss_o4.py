"""GaussO4 langgraph_flow stub. Replace generate() with real API call when endpoint is ready."""
from __future__ import annotations

from langgraph_flow.core.base import BaseLLM


class GaussO4(BaseLLM):
    ENV_URL_KEY = "GAUSS_O4_API_URL"
    ENV_KEY_KEY = "GAUSS_O4_API_KEY"

    def generate(self, prompt: str, context: str = "") -> str:
        return f"[GaussO4] {prompt[:30]} → {context[:50]}"
