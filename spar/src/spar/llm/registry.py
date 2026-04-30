from __future__ import annotations

import asyncio

from spar.llm.client import LLMClient
from spar.llm.config import get_settings
from spar.llm.factory import LLMFactory, LLMRole

_registry: dict[LLMRole, LLMClient] = {}
_lock = asyncio.Lock()


async def get_client(role: LLMRole = LLMRole.MAIN) -> LLMClient:
    if role in _registry:
        return _registry[role]
    async with _lock:
        if role not in _registry:
            _registry[role] = LLMFactory.create(role, get_settings())
    return _registry[role]


def reset_registry() -> None:
    _registry.clear()
