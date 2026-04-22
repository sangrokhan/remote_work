"""GaussO4 backend LLM stub. Returns formatted dummy output for testing."""
from __future__ import annotations

from llm.base import BaseLLM


class GaussO4(BaseLLM):
    ENV_URL_KEY = "GAUSS_O4_API_URL"
    ENV_KEY_KEY = "GAUSS_O4_API_KEY"

    def generate(self, prompt: str, context: str) -> str:
        return f"[GaussO4] {prompt[:30]} → {context[:80]}"
