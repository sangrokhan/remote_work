from __future__ import annotations

import asyncio
import logging
from typing import Protocol

logger = logging.getLogger(__name__)

_BACKOFF_BASE = 2.0


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
        # 앞 클라이언트는 1회 시도 후 다음으로 진행
        for idx, client in enumerate(self._chain[:-1]):
            try:
                return await client.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                logger.warning(
                    "LLM client #%d (%s) failed, trying next: %s", idx, client.model, exc
                )

        # 마지막 클라이언트: 지수 백오프로 성공할 때까지 재시도
        last_idx = len(self._chain) - 1
        last = self._chain[last_idx]
        delay = _BACKOFF_BASE
        attempt = 0
        while True:
            try:
                result = await last.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if attempt > 0:
                    logger.info(
                        "LLM client #%d (%s) succeeded after %d retries",
                        last_idx, last.model, attempt,
                    )
                return result
            except Exception as exc:
                attempt += 1
                logger.warning(
                    "LLM client #%d (%s) failed (attempt %d), retrying in %.0fs: %s",
                    last_idx, last.model, attempt, delay, exc,
                )
                await asyncio.sleep(delay)
                delay *= 2
