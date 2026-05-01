from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from spar.llm.config import get_settings
from spar.llm.factory import LLMFactory, LLMRole

if TYPE_CHECKING:
    from spar.llm.fallback import LLMBackend

_registry: dict[LLMRole, LLMBackend] = {}
_lock = asyncio.Lock()


async def get_client(role: LLMRole = LLMRole.MAIN) -> LLMBackend:
    if role in _registry:
        return _registry[role]
    async with _lock:
        if role not in _registry:
            _registry[role] = LLMFactory.create(role, get_settings())
    return _registry[role]


def reset_registry() -> None:
    _registry.clear()
