from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class LLMBackend(Protocol):
    @property
    def model(self) -> str: ...

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = ...,
        max_tokens: int = ...,
    ) -> str: ...


class FallbackLLMClient:
    """Try clients in order; on exception advance to the next."""

    def __init__(self, primary: LLMBackend, *fallbacks: LLMBackend) -> None:
        if not fallbacks:
            raise ValueError("FallbackLLMClient requires at least one fallback")
        self._chain: tuple[LLMBackend, ...] = (primary, *fallbacks)

    @property
    def model(self) -> str:
        return self._chain[0].model

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        last_exc: BaseException | None = None
        for idx, client in enumerate(self._chain):
            try:
                return await client.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "LLM client #%d (%s) failed: %s", idx, client.model, exc
                )
        assert last_exc is not None
        raise last_exc
