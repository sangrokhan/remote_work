"""GaussO4Think backend LLM stub. Wraps output in <thinking> tags to simulate chain-of-thought."""
from __future__ import annotations

from llm.base import BaseLLM


class GaussO4Think(BaseLLM):
    ENV_URL_KEY = "GAUSS_O4_THINK_API_URL"
    ENV_KEY_KEY = "GAUSS_O4_THINK_API_KEY"

    def generate(self, prompt: str, context: str) -> str:
        return f"[GaussO4-think] <thinking>{prompt[:30]}</thinking> → {context[:80]}"
