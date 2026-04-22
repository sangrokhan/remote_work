from __future__ import annotations

from llm.base import BaseLLM


class GaussO4Think(BaseLLM):
    def generate(self, prompt: str, context: str) -> str:
        return f"[GaussO4-think] <thinking>{prompt[:30]}</thinking> → {context[:80]}"
