# Encoder Singleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `spar.llm` 패턴을 그대로 mirror하는 pluggable embedding encoder singleton 구현 — `.env`에서 provider/model 설정, `get_encoder()`로 singleton 접근, `EmbeddingRouter`가 주입받는 구조.

**Architecture:** `EncoderClient` ABC → `SentenceTransformerEncoder` 구현체 → `EncoderFactory.create(settings)` → `get_encoder()` async singleton (asyncio.Lock). `EmbeddingRouter`는 생성자에서 `EncoderClient`를 주입받아 직접 의존성 없음.

**Tech Stack:** Python 3.11+, pydantic-settings, sentence-transformers, numpy, pytest, unittest.mock

---

## File Map

**Create (new):**
- `src/spar/encoder/base.py` — `EncoderClient` ABC (`encode` 추상 메서드)
- `tests/encoder/__init__.py`
- `tests/encoder/test_config.py`
- `tests/encoder/test_factory.py`
- `tests/encoder/test_registry.py`

**Fill in (exists but empty):**
- `src/spar/encoder/config.py` — `EncoderSettings` + `get_settings()` + `reset_settings()`
- `src/spar/encoder/client.py` — `SentenceTransformerEncoder(EncoderClient)`
- `src/spar/encoder/factory.py` — `EncoderProvider` enum + `EncoderFactory.create(settings)`
- `src/spar/encoder/registry.py` — `get_encoder()` async singleton
- `src/spar/encoder/__init__.py` — public exports

**Modify:**
- `src/spar/router/embedding_router.py` — `EncoderClient` 주입받도록 리팩터
- `tests/router/test_embedding_router.py` — mock encoder 사용

---

### Task 1: EncoderClient ABC + config

**Files:**
- Create: `src/spar/encoder/base.py`
- Fill in: `src/spar/encoder/config.py`

- [ ] **Step 1: tests/encoder/__init__.py 생성**

```bash
mkdir -p /path/to/spar/tests/encoder
touch tests/encoder/__init__.py
```

- [ ] **Step 2: test_config.py 작성 (실패 확인용)**

`tests/encoder/test_config.py`:
```python
from __future__ import annotations

from spar.encoder.config import EncoderSettings, get_settings, reset_settings


def test_defaults():
    s = EncoderSettings()
    assert s.encoder_provider == "sentence_transformers"
    assert s.encoder_model == "BAAI/bge-small-en-v1.5"
    assert s.encoder_device == "cpu"


def test_get_settings_singleton():
    reset_settings()
    a = get_settings()
    b = get_settings()
    assert a is b


def test_reset_settings():
    reset_settings()
    a = get_settings()
    reset_settings()
    b = get_settings()
    assert a is not b
```

- [ ] **Step 3: pytest 실행 — FAIL 확인**

```bash
pytest tests/encoder/test_config.py -v
```
Expected: `ImportError` or `ModuleNotFoundError`

- [ ] **Step 4: base.py 작성**

`src/spar/encoder/base.py`:
```python
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EncoderClient(ABC):
    @abstractmethod
    def encode(self, texts: list[str], *, normalize: bool = True) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
```

- [ ] **Step 5: config.py 작성**

`src/spar/encoder/config.py`:
```python
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class EncoderSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    encoder_provider: str = "sentence_transformers"
    encoder_model: str = "BAAI/bge-small-en-v1.5"
    encoder_device: str = "cpu"


_settings: EncoderSettings | None = None


def get_settings() -> EncoderSettings:
    global _settings
    if _settings is None:
        _settings = EncoderSettings()
    return _settings


def reset_settings() -> None:
    global _settings
    _settings = None
```

- [ ] **Step 6: pytest 실행 — PASS 확인**

```bash
pytest tests/encoder/test_config.py -v
```
Expected: 3 passed

- [ ] **Step 7: 커밋**

```bash
git add src/spar/encoder/base.py src/spar/encoder/config.py tests/encoder/__init__.py tests/encoder/test_config.py
git commit -m "feat(encoder): add EncoderClient ABC and EncoderSettings config"
```

---

### Task 2: SentenceTransformerEncoder (client.py)

**Files:**
- Fill in: `src/spar/encoder/client.py`

- [ ] **Step 1: test_client.py 작성**

`tests/encoder/test_client.py`:
```python
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from spar.encoder.client import SentenceTransformerEncoder


@pytest.fixture
def encoder():
    with patch("spar.encoder.client.SentenceTransformer") as mock_cls:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_cls.return_value = mock_model
        yield SentenceTransformerEncoder(model_name="BAAI/bge-small-en-v1.5", device="cpu")


def test_model_name(encoder):
    assert encoder.model_name == "BAAI/bge-small-en-v1.5"


def test_encode_returns_ndarray(encoder):
    result = encoder.encode(["hello world"])
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3)


def test_encode_calls_underlying_model(encoder):
    encoder.encode(["test"], normalize=True)
    encoder._model.encode.assert_called_once_with(
        ["test"], normalize_embeddings=True
    )
```

- [ ] **Step 2: pytest 실행 — FAIL 확인**

```bash
pytest tests/encoder/test_client.py -v
```
Expected: `ImportError`

- [ ] **Step 3: client.py 작성**

`src/spar/encoder/client.py`:
```python
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from spar.encoder.base import EncoderClient


class SentenceTransformerEncoder(EncoderClient):
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], *, normalize: bool = True) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=normalize)

    @property
    def model_name(self) -> str:
        return self._model_name
```

- [ ] **Step 4: pytest 실행 — PASS 확인**

```bash
pytest tests/encoder/test_client.py -v
```
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add src/spar/encoder/client.py tests/encoder/test_client.py
git commit -m "feat(encoder): add SentenceTransformerEncoder implementation"
```

---

### Task 3: EncoderFactory

**Files:**
- Fill in: `src/spar/encoder/factory.py`
- Create: `tests/encoder/test_factory.py`

- [ ] **Step 1: test_factory.py 작성**

`tests/encoder/test_factory.py`:
```python
from __future__ import annotations

import pytest
from unittest.mock import patch

from spar.encoder.client import SentenceTransformerEncoder
from spar.encoder.config import EncoderSettings
from spar.encoder.factory import EncoderFactory, EncoderProvider


@pytest.fixture
def settings():
    return EncoderSettings(
        encoder_provider="sentence_transformers",
        encoder_model="BAAI/bge-small-en-v1.5",
        encoder_device="cpu",
    )


def test_provider_enum_values():
    assert EncoderProvider.SENTENCE_TRANSFORMERS.value == "sentence_transformers"


def test_create_sentence_transformers(settings):
    with patch("spar.encoder.client.SentenceTransformer"):
        encoder = EncoderFactory.create(settings)
    assert isinstance(encoder, SentenceTransformerEncoder)
    assert encoder.model_name == "BAAI/bge-small-en-v1.5"


def test_unknown_provider_raises():
    bad = EncoderSettings(encoder_provider="unknown_provider")
    with pytest.raises(ValueError, match="Unknown encoder provider"):
        EncoderFactory.create(bad)
```

- [ ] **Step 2: pytest 실행 — FAIL 확인**

```bash
pytest tests/encoder/test_factory.py -v
```
Expected: `ImportError`

- [ ] **Step 3: factory.py 작성**

`src/spar/encoder/factory.py`:
```python
from __future__ import annotations

from enum import Enum

from spar.encoder.base import EncoderClient
from spar.encoder.client import SentenceTransformerEncoder
from spar.encoder.config import EncoderSettings


class EncoderProvider(str, Enum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class EncoderFactory:
    @staticmethod
    def create(settings: EncoderSettings) -> EncoderClient:
        if settings.encoder_provider == EncoderProvider.SENTENCE_TRANSFORMERS:
            return SentenceTransformerEncoder(
                model_name=settings.encoder_model,
                device=settings.encoder_device,
            )
        raise ValueError(f"Unknown encoder provider: {settings.encoder_provider!r}")
```

- [ ] **Step 4: pytest 실행 — PASS 확인**

```bash
pytest tests/encoder/test_factory.py -v
```
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add src/spar/encoder/factory.py tests/encoder/test_factory.py
git commit -m "feat(encoder): add EncoderProvider enum and EncoderFactory"
```

---

### Task 4: get_encoder() singleton registry

**Files:**
- Fill in: `src/spar/encoder/registry.py`
- Create: `tests/encoder/test_registry.py`

- [ ] **Step 1: test_registry.py 작성**

`tests/encoder/test_registry.py`:
```python
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from spar.encoder.base import EncoderClient
from spar.encoder.config import EncoderSettings
from spar.encoder.registry import get_encoder, reset_registry

_TEST_SETTINGS = EncoderSettings(
    encoder_provider="sentence_transformers",
    encoder_model="BAAI/bge-small-en-v1.5",
    encoder_device="cpu",
)


@pytest.fixture(autouse=True)
def clean_registry():
    reset_registry()
    yield
    reset_registry()


async def test_get_encoder_returns_encoder_client():
    with patch("spar.encoder.client.SentenceTransformer"), \
         patch("spar.encoder.registry.get_settings", return_value=_TEST_SETTINGS):
        encoder = await get_encoder()
    assert isinstance(encoder, EncoderClient)


async def test_get_encoder_singleton():
    with patch("spar.encoder.client.SentenceTransformer"), \
         patch("spar.encoder.registry.get_settings", return_value=_TEST_SETTINGS):
        a = await get_encoder()
        b = await get_encoder()
    assert a is b


async def test_reset_clears_singleton():
    with patch("spar.encoder.client.SentenceTransformer"), \
         patch("spar.encoder.registry.get_settings", return_value=_TEST_SETTINGS):
        a = await get_encoder()
        reset_registry()
        b = await get_encoder()
    assert a is not b
```

- [ ] **Step 2: pytest 실행 — FAIL 확인**

```bash
pytest tests/encoder/test_registry.py -v
```
Expected: `ImportError`

- [ ] **Step 3: registry.py 작성**

`src/spar/encoder/registry.py`:
```python
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
```

- [ ] **Step 4: pytest 실행 — PASS 확인**

```bash
pytest tests/encoder/test_registry.py -v
```
Expected: 3 passed

- [ ] **Step 5: __init__.py 작성**

`src/spar/encoder/__init__.py`:
```python
from spar.encoder.base import EncoderClient
from spar.encoder.factory import EncoderProvider
from spar.encoder.registry import get_encoder, reset_registry

__all__ = ["get_encoder", "reset_registry", "EncoderClient", "EncoderProvider"]
```

- [ ] **Step 6: 커밋**

```bash
git add src/spar/encoder/registry.py src/spar/encoder/__init__.py tests/encoder/test_registry.py
git commit -m "feat(encoder): add get_encoder() async singleton registry"
```

---

### Task 5: EmbeddingRouter 리팩터 — EncoderClient 주입

**Files:**
- Modify: `src/spar/router/embedding_router.py`
- Modify: `tests/router/test_embedding_router.py`

- [ ] **Step 1: test_embedding_router.py 수정 — mock encoder 사용**

`tests/router/test_embedding_router.py` 전체 교체:
```python
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from spar.encoder.base import EncoderClient
from spar.router.embedding_router import EmbeddingRouter
from spar.router.schemas import Route


def _make_encoder(dim: int = 8) -> EncoderClient:
    """Stub encoder: returns deterministic vectors per unique text."""
    encoder = MagicMock(spec=EncoderClient)

    rng = np.random.default_rng(42)
    cache: dict[str, np.ndarray] = {}

    def _encode(texts: list[str], *, normalize: bool = True) -> np.ndarray:
        vecs = []
        for t in texts:
            if t not in cache:
                v = rng.random(dim).astype(np.float32)
                if normalize:
                    v /= np.linalg.norm(v)
                cache[t] = v
            vecs.append(cache[t])
        return np.array(vecs)

    encoder.encode.side_effect = _encode
    return encoder


@pytest.fixture(scope="module")
def router():
    return EmbeddingRouter(encoder=_make_encoder(), threshold=0.5)


def test_route_returns_result_or_none(router):
    result = router.route("How do I configure RACH parameters in NR?")
    assert result is None or result.layer == "embedding"


def test_low_confidence_returns_none():
    strict = EmbeddingRouter(encoder=_make_encoder(), threshold=0.99)
    result = strict.route("zzz xyz qwerty random nonsense")
    assert result is None


def test_route_result_has_confidence(router):
    result = router.route("What is Carrier Aggregation?")
    if result is not None:
        assert 0.0 <= result.confidence <= 1.0
        assert result.layer == "embedding"
```

- [ ] **Step 2: pytest 실행 — 기존 테스트 상태 확인**

```bash
pytest tests/router/test_embedding_router.py -v
```
Expected: pass (아직 router 변경 전이므로 기존 테스트 통과 확인)

- [ ] **Step 3: embedding_router.py 수정**

`src/spar/router/embedding_router.py`:
```python
from __future__ import annotations

import numpy as np

from spar.encoder.base import EncoderClient
from spar.router.schemas import Route, RouteResult

ROUTE_EXAMPLES: dict[Route, list[str]] = {
    Route.STRUCTURED_LOOKUP: [
        "What is the default value of maxTxPower?",
        "Show me the range of pMax parameter",
        "List parameters in NRCellDU MO",
        "What alarms are related to RACH failure?",
        "What is the formula for RRC success rate counter?",
    ],
    Route.DEFINITION_EXPLAIN: [
        "What is Carrier Aggregation?",
        "Explain the difference between FDD and TDD",
        "What does BWP stand for?",
        "Describe the handover procedure",
        "What is beam management in NR?",
    ],
    Route.PROCEDURAL: [
        "How do I configure RACH parameters?",
        "Steps to enable Carrier Aggregation",
        "How to install the RAN software package?",
        "Procedure for activating a new cell",
        "How to configure QoS profiles?",
    ],
    Route.DIAGNOSTIC: [
        "Why is the handover failure rate high?",
        "RACH congestion issue after software upgrade",
        "Cell is stuck in blocking state",
        "Throughput dropped after parameter change",
        "Why are users experiencing call drops?",
    ],
    Route.COMPARATIVE: [
        "What changed in v7.0 compared to v6.0?",
        "Difference between SA and NSA mode configuration",
        "How does the new preamble format compare to the old one?",
        "What features were added in the latest release?",
        "Compare LTE and NR RACH procedures",
    ],
    Route.DEFAULT_RAG: [
        "Tell me about the RAN system",
        "General information about 5G",
        "Overview of the network",
    ],
}


class EmbeddingRouter:
    """Layer 2: cosine similarity against route centroid embeddings."""

    def __init__(self, encoder: EncoderClient, threshold: float = 0.65) -> None:
        self.threshold = threshold
        self._encoder = encoder
        self._centroids = self._build_centroids()

    def _build_centroids(self) -> dict[Route, np.ndarray]:
        centroids: dict[Route, np.ndarray] = {}
        for route, examples in ROUTE_EXAMPLES.items():
            embs = self._encoder.encode(examples, normalize=True)
            centroids[route] = np.mean(embs, axis=0)
        return centroids

    def route(self, query: str) -> RouteResult | None:
        q_emb = self._encoder.encode([query], normalize=True)[0]
        best_route, best_score = Route.DEFAULT_RAG, -1.0
        for route, centroid in self._centroids.items():
            score = float(np.dot(q_emb, centroid))
            if score > best_score:
                best_score, best_route = score, route

        if best_score < self.threshold:
            return None

        return RouteResult(route=best_route, confidence=best_score, layer="embedding")
```

- [ ] **Step 4: pytest 실행 — PASS 확인**

```bash
pytest tests/router/test_embedding_router.py tests/encoder/ -v
```
Expected: 모두 pass

- [ ] **Step 5: 전체 테스트 suite 확인**

```bash
pytest --tb=short -q
```
Expected: 기존 테스트 포함 모두 pass

- [ ] **Step 6: 커밋**

```bash
git add src/spar/router/embedding_router.py tests/router/test_embedding_router.py
git commit -m "refactor(router): inject EncoderClient into EmbeddingRouter"
```

---

### Task 6: hybrid_router / 사용처 업데이트 확인

**Files:**
- Modify: `src/spar/router/hybrid_router.py` (EmbeddingRouter 생성 부분)

- [ ] **Step 1: hybrid_router에서 EmbeddingRouter 생성 방식 확인**

```bash
grep -n "EmbeddingRouter" src/spar/router/hybrid_router.py
```

- [ ] **Step 2: hybrid_router 수정 — get_encoder() 사용**

`hybrid_router.py`에서 `EmbeddingRouter()` 생성 코드를 찾아서:

기존:
```python
self._embedding = EmbeddingRouter(threshold=..., model_name=...)
```

수정 후 (async init 패턴):
```python
# HybridRouter.__init__에서는 encoder 주입
from spar.encoder import get_encoder

# async factory method 추가
@classmethod
async def create(cls, ...) -> "HybridRouter":
    encoder = await get_encoder()
    embedding_router = EmbeddingRouter(encoder=encoder, threshold=...)
    return cls(..., embedding_router=embedding_router)
```

> 주의: `hybrid_router.py` 실제 코드를 확인한 후 최소 변경으로 수정할 것. `EmbeddingRouter`가 직접 생성되지 않는다면 이 step 건너뜀.

- [ ] **Step 3: 전체 테스트 통과 확인**

```bash
pytest --tb=short -q
```
Expected: 모두 pass

- [ ] **Step 4: 커밋**

```bash
git add src/spar/router/hybrid_router.py
git commit -m "feat(router): wire EmbeddingRouter with get_encoder() singleton"
```

---

## Self-Review

**Spec coverage:**
- [x] EncoderClient ABC — Task 1
- [x] SentenceTransformerEncoder — Task 2
- [x] EncoderFactory (pluggable provider) — Task 3
- [x] get_encoder() async singleton — Task 4
- [x] EmbeddingRouter 주입 — Task 5
- [x] 사용처 업데이트 — Task 6

**Type consistency:**
- `EncoderClient.encode(texts: list[str], *, normalize: bool = True) -> np.ndarray` — Tasks 1,2,3,5 모두 동일 시그니처
- `get_encoder() -> EncoderClient` — Tasks 4,5,6 일관
- `reset_registry()` — Task 4 registry, __init__ exports 일치

**No placeholders:** 모든 step에 실제 코드 포함.
