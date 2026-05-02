from __future__ import annotations

import asyncio

from spar.encoder.client import CrossEncoderClient
from spar.encoder.config import get_settings
from spar.encoder.factory import EncoderFactory, EncoderRole

_registry: dict[EncoderRole, CrossEncoderClient] = {}
_lock = asyncio.Lock()


async def get_reranker(role: EncoderRole = EncoderRole.RERANKER) -> CrossEncoderClient:
    if role in _registry:
        return _registry[role]
    async with _lock:
        if role not in _registry:
            _registry[role] = EncoderFactory.create(role, get_settings())
    return _registry[role]


def reset_registry() -> None:
    _registry.clear()
