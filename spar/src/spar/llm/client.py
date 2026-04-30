from __future__ import annotations

from openai import AsyncOpenAI


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "dummy") -> None:
        self._model = model
        self._openai = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        resp = await self._openai.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    @property
    def model(self) -> str:
        return self._model
