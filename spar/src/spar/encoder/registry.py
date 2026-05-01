from __future__ import annotations

import asyncio

from spar.encoder.base import EncoderClient
from spar.encoder.config import get_settings
from spar.encoder.factory import EncoderFactory

_encoder: EncoderClient | None = None
_lock = asyncio.Lock()


async def get_encoder() -> EncoderClient:
    global _encoder
    if _encoder is not None:
        return _encoder
    async with _lock:
        if _encoder is None:
            _encoder = EncoderFactory.create(get_settings())
    return _encoder


def reset_registry() -> None:
    global _encoder
    _encoder = None
