"""GaussO4Think backend LLM stub. Wraps output in <thinking> tags to simulate chain-of-thought."""
from __future__ import annotations

from typing import Any, Optional

from llm.base import BaseLLM, LanguageModelInput


class GaussO4Think(BaseLLM):
    ENV_URL_KEY = "GAUSS_O4_THINK_API_URL"
    ENV_KEY_KEY = "GAUSS_O4_THINK_API_KEY"

    def invoke(self, input: LanguageModelInput, config: Optional[Any] = None, **kwargs: Any) -> str:
        prompt = self._to_str(input)
        return f"[GaussO4-think] <thinking>{prompt[:30]}</thinking>"

    async def ainvoke(self, input: LanguageModelInput, config: Optional[Any] = None, **kwargs: Any) -> str:
        prompt = self._to_str(input)
        return f"[GaussO4-think] <thinking>{prompt[:30]}</thinking>"
