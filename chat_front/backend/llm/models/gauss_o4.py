"""GaussO4 backend LLM stub. Returns formatted dummy output for testing."""
from __future__ import annotations

from typing import Any, Optional

from llm.base import BaseLLM, LanguageModelInput


class GaussO4(BaseLLM):
    ENV_URL_KEY = "GAUSS_O4_API_URL"
    ENV_KEY_KEY = "GAUSS_O4_API_KEY"

    def invoke(self, input: LanguageModelInput, config: Optional[Any] = None, **kwargs: Any) -> str:
        prompt = self._to_str(input)
        return f"[GaussO4] {prompt[:30]}"

    async def ainvoke(self, input: LanguageModelInput, config: Optional[Any] = None, **kwargs: Any) -> str:
        prompt = self._to_str(input)
        return f"[GaussO4] {prompt[:30]}"
