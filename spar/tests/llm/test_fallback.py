import pytest

from spar.llm.fallback import FallbackLLMClient


class _StubBackend:
    def __init__(self, name: str, *, fail: bool = False, reply: str = "") -> None:
        self._name = name
        self._fail = fail
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
        if self._fail:
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

    out = await chain.chat([{"role": "user", "content": "hi"}])

    assert out == "fallback-ok"
    assert primary.calls == 1
    assert fallback.calls == 1


async def test_all_fail_raises_last_exception():
    primary = _StubBackend("p", fail=True)
    fallback = _StubBackend("f", fail=True)
    chain = FallbackLLMClient(primary, fallback)

    with pytest.raises(RuntimeError, match="f failed"):
        await chain.chat([{"role": "user", "content": "hi"}])

    assert primary.calls == 1
    assert fallback.calls == 1


def test_model_returns_primary_model():
    primary = _StubBackend("primary-model", reply="x")
    fallback = _StubBackend("fallback-model", reply="y")
    chain = FallbackLLMClient(primary, fallback)
    assert chain.model == "primary-model"
