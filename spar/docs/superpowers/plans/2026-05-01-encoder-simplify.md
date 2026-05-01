# Encoder Module Simplify Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `encoder/` 5개 파일(base/client/factory/config/registry)을 2개(base/registry)로 통합 — YAGNI, PRD Task 1.4 범위에 맞게.

**Architecture:** `EncoderClient` ABC는 router 주입/테스트 인터페이스로 유지. `SentenceTransformerEncoder` + env 설정 + 싱글톤을 `registry.py` 하나로 합침. `factory.py`/`config.py`/`client.py` 삭제.

**Tech Stack:** Python 3.12, sentence-transformers, numpy, pytest, asyncio

---

## File Map

| Action | Path |
|--------|------|
| Keep (unchanged) | `src/spar/encoder/base.py` |
| Rewrite | `src/spar/encoder/registry.py` |
| Rewrite | `src/spar/encoder/__init__.py` |
| Delete | `src/spar/encoder/client.py` |
| Delete | `src/spar/encoder/factory.py` |
| Delete | `src/spar/encoder/config.py` |
| Rewrite | `tests/encoder/test_registry.py` |
| Delete | `tests/encoder/test_client.py` |
| Delete | `tests/encoder/test_config.py` |
| Delete | `tests/encoder/test_factory.py` |

Router/ingest 파일은 변경 없음 — `EncoderClient` import 경로(`spar.encoder.base`) 유지.

---

## Task 1: 새 registry.py 작성 (TDD)

**Files:**
- Modify: `src/spar/encoder/registry.py`
- Modify: `tests/encoder/test_registry.py`

### Step 1: 기존 테스트 삭제 후 새 테스트 작성

`tests/encoder/test_registry.py` 전체 교체:

```python
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from spar.encoder.base import EncoderClient
from spar.encoder.registry import SentenceTransformerEncoder, get_encoder, reset_registry


@pytest.fixture(autouse=True)
def clean_registry():
    reset_registry()
    yield
    reset_registry()


# --- SentenceTransformerEncoder ---

@pytest.fixture
def encoder():
    with patch("spar.encoder.registry.SentenceTransformer") as mock_cls:
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


def test_encode_normalize_flag(encoder):
    encoder.encode(["test"], normalize=True)
    encoder._model.encode.assert_called_once_with(["test"], normalize_embeddings=True)


def test_encode_normalize_false(encoder):
    encoder.encode(["test"], normalize=False)
    encoder._model.encode.assert_called_once_with(["test"], normalize_embeddings=False)


def test_implements_encoder_client(encoder):
    assert isinstance(encoder, EncoderClient)


# --- get_encoder singleton ---

async def test_get_encoder_returns_encoder_client():
    with patch("spar.encoder.registry.SentenceTransformer"):
        enc = await get_encoder()
    assert isinstance(enc, EncoderClient)


async def test_get_encoder_singleton():
    with patch("spar.encoder.registry.SentenceTransformer"):
        a = await get_encoder()
        b = await get_encoder()
    assert a is b


async def test_reset_registry_breaks_singleton():
    with patch("spar.encoder.registry.SentenceTransformer"):
        a = await get_encoder()
    reset_registry()
    with patch("spar.encoder.registry.SentenceTransformer"):
        b = await get_encoder()
    assert a is not b


async def test_get_encoder_uses_env_model(monkeypatch):
    monkeypatch.setenv("ENCODER_MODEL", "intfloat/e5-large-v2")
    with patch("spar.encoder.registry.SentenceTransformer") as mock_cls:
        mock_cls.return_value = MagicMock()
        enc = await get_encoder()
    assert enc.model_name == "intfloat/e5-large-v2"


async def test_get_encoder_uses_env_device(monkeypatch):
    monkeypatch.setenv("ENCODER_DEVICE", "cuda")
    with patch("spar.encoder.registry.SentenceTransformer") as mock_cls:
        mock_cls.return_value = MagicMock()
        await get_encoder()
    mock_cls.assert_called_once_with("BAAI/bge-small-en-v1.5", device="cuda")
```

### Step 2: 테스트 실행 — FAIL 확인

```bash
cd /home/han/.openclaw/workspace/remote_work/spar
pytest tests/encoder/test_registry.py -v
```

Expected: `ImportError` 또는 `FAILED` — `SentenceTransformerEncoder` 아직 registry에 없음.

### Step 3: 새 registry.py 작성

`src/spar/encoder/registry.py` 전체 교체:

```python
from __future__ import annotations

import asyncio
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from spar.encoder.base import EncoderClient

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

_encoder: EncoderClient | None = None
_lock = asyncio.Lock()


class SentenceTransformerEncoder(EncoderClient):
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], *, normalize: bool = True) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=normalize)

    @property
    def model_name(self) -> str:
        return self._model_name


async def get_encoder() -> EncoderClient:
    global _encoder
    if _encoder is not None:
        return _encoder
    async with _lock:
        if _encoder is None:
            model = os.getenv("ENCODER_MODEL", _DEFAULT_MODEL)
            device = os.getenv("ENCODER_DEVICE", "cpu")
            _encoder = SentenceTransformerEncoder(model_name=model, device=device)
    return _encoder


def reset_registry() -> None:
    global _encoder
    _encoder = None
```

### Step 4: 테스트 실행 — PASS 확인

```bash
pytest tests/encoder/test_registry.py -v
```

Expected: 모든 테스트 PASS.

### Step 5: Commit

```bash
git add src/spar/encoder/registry.py tests/encoder/test_registry.py
git commit -m "refactor(encoder): consolidate SentenceTransformerEncoder into registry"
```

---

## Task 2: __init__.py 업데이트 + 구 파일 삭제

**Files:**
- Modify: `src/spar/encoder/__init__.py`
- Delete: `src/spar/encoder/client.py`, `src/spar/encoder/factory.py`, `src/spar/encoder/config.py`
- Delete: `tests/encoder/test_client.py`, `tests/encoder/test_config.py`, `tests/encoder/test_factory.py`

### Step 1: __init__.py 업데이트

`src/spar/encoder/__init__.py` 전체 교체:

```python
from spar.encoder.base import EncoderClient
from spar.encoder.registry import SentenceTransformerEncoder, get_encoder, reset_registry

__all__ = ["get_encoder", "reset_registry", "EncoderClient", "SentenceTransformerEncoder"]
```

### Step 2: 구 소스 파일 삭제

```bash
rm src/spar/encoder/client.py
rm src/spar/encoder/factory.py
rm src/spar/encoder/config.py
```

### Step 3: 구 테스트 파일 삭제

```bash
rm tests/encoder/test_client.py
rm tests/encoder/test_config.py
rm tests/encoder/test_factory.py
```

### Step 4: 전체 테스트 스위트 실행

```bash
pytest tests/ -v --ignore=tests/ingest/test_run_ingest_smoke.py --ignore=tests/ingest/test_tspec_llm_smoke.py
```

Expected: 모든 테스트 PASS. router 테스트는 `spar.encoder.base.EncoderClient`만 import하므로 영향 없음.

### Step 5: Commit

```bash
git add src/spar/encoder/__init__.py
git rm src/spar/encoder/client.py src/spar/encoder/factory.py src/spar/encoder/config.py
git rm tests/encoder/test_client.py tests/encoder/test_config.py tests/encoder/test_factory.py
git commit -m "refactor(encoder): remove factory/config/client — collapse to base+registry"
```

---

## Task 3: 환경변수 명칭 변경 반영 (EMBED_MODEL → ENCODER_MODEL)

> `ingest/embedder.py`는 `EMBED_MODEL` env var 사용. `encoder/registry.py`는 `ENCODER_MODEL` 사용. 충돌 없음 — 각자 독립. 이 Task는 README/AGENTS.md 문서 동기화.

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/prd.md`

### Step 1: README.md 환경변수 섹션 확인 후 업데이트

```bash
grep -n "EMBED_MODEL\|ENCODER_MODEL\|ENCODER_DEVICE\|encoder" README.md
```

`ENCODER_MODEL`, `ENCODER_DEVICE` env var 문서화. 없으면 환경변수 섹션에 추가:

```
ENCODER_MODEL   sentence-transformers 모델명 (기본: BAAI/bge-small-en-v1.5)
ENCODER_DEVICE  인코더 디바이스 (기본: cpu)
EMBED_MODEL     ingest 전용 임베더 모델 (기본: BAAI/bge-large-en-v1.5)
```

### Step 2: AGENTS.md 디렉토리 맵 업데이트

`encoder/` 항목을 다음으로 교체:

```
src/spar/encoder/
  base.py      — EncoderClient ABC (router 주입 인터페이스)
  registry.py  — SentenceTransformerEncoder + get_encoder() 싱글톤
```

### Step 3: docs/prd.md Task 1.4 체크박스 업데이트

```
- [x] encoder 싱글톤/팩토리 — 완료 (refactor: base+registry로 통합)
```

### Step 4: Commit

```bash
git add README.md AGENTS.md docs/prd.md
git commit -m "docs: update encoder module structure after simplification refactor"
```

---

## Self-Review Checklist

- [x] `EncoderClient` ABC 유지 — router 테스트 `MagicMock(spec=EncoderClient)` 패턴 보존
- [x] `spar.encoder.base.EncoderClient` import 경로 불변 — router 파일 수정 불필요
- [x] env var `ENCODER_MODEL`/`ENCODER_DEVICE` 테스트 커버
- [x] `reset_registry()` 테스트 fixture 유지 — 격리 보장
- [x] `ingest/Embedder` 독립 유지 — 인터페이스 다름 (`list[list[float]]` vs `np.ndarray`)
- [x] smoke 테스트 제외한 전체 pytest PASS 검증 포함
