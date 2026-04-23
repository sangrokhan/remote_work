"""GaussO4 backend LLM."""
from __future__ import annotations

from typing import Any, Optional

from openai import AsyncOpenAI
from pydantic import PrivateAttr

from llm.base import BaseLLM, LanguageModelInput


class GaussO4(BaseLLM):
    ENV_URL_KEY = "GAUSS_O4_API_URL"
    ENV_KEY_KEY = "GAUSS_O4_API_KEY"
    MODEL_NAME = "gauss-o4-instruct"

    _async_client: AsyncOpenAI = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self._async_client = AsyncOpenAI(
            base_url=self.api_url or None,
            api_key=self.api_key or "dummy",
        )

    def invoke(self, input: LanguageModelInput, config: Optional[Any] = None, **kwargs: Any) -> str:
        prompt = self._to_str(input)
        return f"[GaussO4] {prompt[:30]}"

    async def ainvoke(self, input: LanguageModelInput, config: Optional[Any] = None, **kwargs: Any) -> str:
        prompt = self._to_str(input)
        resp = await self._async_client.chat.completions.create(
            model=self.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""
