"""Gemma4-E4B-it backend LLM stub. Returns dummy output for testing."""
from __future__ import annotations

from typing import Any, Optional

from llm.base import BaseLLM, LanguageModelInput


class Gemma4E4BIt(BaseLLM):
    ENV_URL_KEY = "GEMMA4_E4B_IT_API_URL"
    ENV_KEY_KEY = "GEMMA4_E4B_IT_API_KEY"

    def invoke(self, input: LanguageModelInput, config: Optional[Any] = None, **kwargs: Any) -> str:
        prompt = self._to_str(input)
        return f"[Gemma4-E4B-it] {prompt[:60]}"

    async def ainvoke(self, input: LanguageModelInput, config: Optional[Any] = None, **kwargs: Any) -> str:
        prompt = self._to_str(input)
        return f"[Gemma4-E4B-it] {prompt[:60]}"
