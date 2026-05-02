import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from spar.llm.fallback import FallbackLLMClient, _BACKOFF_BASE


class _StubBackend:
    def __init__(
        self,
        name: str,
        *,
        fail: bool = False,
        fail_times: int = 0,
        reply: str = "",
    ) -> None:
        self._name = name
        self._fail = fail
        self._fail_times = fail_times
        self._reply = reply
        self.calls = 0

    @property
    def model(self) -> str:
        return self._name

    async def chat(
        self,
        messages,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        self.calls += 1
        if self._fail or self.calls <= self._fail_times:
            raise RuntimeError(f"{self._name} failed")
        return self._reply


def test_requires_at_least_one_fallback():
    primary = _StubBackend("p", reply="ok")
    with pytest.raises(ValueError, match="at least one fallback"):
        FallbackLLMClient(primary)


async def test_primary_success_skips_fallback():
    primary = _StubBackend("p", reply="primary-ok")
    fallback = _StubBackend("f", reply="fallback-ok")
    chain = FallbackLLMClient(primary, fallback)

    out = await chain.chat([{"role": "user", "content": "hi"}])

    assert out == "primary-ok"
    assert primary.calls == 1
    assert fallback.calls == 0


async def test_falls_back_on_primary_exception():
    primary = _StubBackend("p", fail=True)
    fallback = _StubBackend("f", reply="fallback-ok")
    chain = FallbackLLMClient(primary, fallback)

    with patch("spar.llm.fallback.asyncio.sleep", new_callable=AsyncMock):
        out = await chain.chat([{"role": "user", "content": "hi"}])

    assert out == "fallback-ok"
    assert primary.calls == 1
    assert fallback.calls == 1


async def test_last_client_retries_with_exponential_backoff():
    """마지막 클라이언트 2회 실패 후 성공 → sleep 호출 지연값 확인."""
    primary = _StubBackend("p", fail=True)
    fallback = _StubBackend("f", fail_times=2, reply="fallback-ok")
    chain = FallbackLLMClient(primary, fallback)

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    with patch("spar.llm.fallback.asyncio.sleep", side_effect=fake_sleep):
        out = await chain.chat([{"role": "user", "content": "hi"}])

    assert out == "fallback-ok"
    assert primary.calls == 1
    assert fallback.calls == 3  # 2 fail + 1 success
    assert sleep_calls == [_BACKOFF_BASE, _BACKOFF_BASE * 2]


async def test_last_client_backoff_delays_double_each_time():
    """지연값이 2배씩 누적되는지 검증 (3회 실패)."""
    primary = _StubBackend("p", fail=True)
    fallback = _StubBackend("f", fail_times=3, reply="ok")
    chain = FallbackLLMClient(primary, fallback)

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    with patch("spar.llm.fallback.asyncio.sleep", side_effect=fake_sleep):
        await chain.chat([])

    assert sleep_calls == [2.0, 4.0, 8.0]


def test_model_returns_primary_model():
    primary = _StubBackend("primary-model", reply="x")
    fallback = _StubBackend("fallback-model", reply="y")
    chain = FallbackLLMClient(primary, fallback)
    assert chain.model == "primary-model"
