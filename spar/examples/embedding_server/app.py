from __future__ import annotations

import os
from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - 예제 서버 실행 환경 전용
    raise RuntimeError(
        "sentence-transformers is required. Install requirements.txt first."
    ) from exc


DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
DEFAULT_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")
API_KEY = os.environ.get("EMBEDDING_API_KEY", "").strip()

_model: SentenceTransformer | None = None


class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="Model name or local path")
    input: str | list[str] = Field(..., description="Input text or text list")


class EmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingItem]
    model: str
    usage: EmbeddingUsage


def _require_api_key(auth_header: str | None) -> None:
    if not API_KEY:
        return
    if auth_header != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")


def _ensure_model_loaded() -> SentenceTransformer:
    if _model is None:
        raise RuntimeError("Model is not initialized.")
    return _model


def _normalize_input(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    texts = [item for item in value if isinstance(item, str)]
    if len(texts) != len(value):
        raise HTTPException(status_code=400, detail="All input items must be strings")
    if not texts:
        raise HTTPException(status_code=400, detail="Input must not be empty")
    return texts


def _estimate_tokens(texts: list[str]) -> int:
    # 운영 비용 추정이 아니라 호환성 필드 채우기 목적의 근사치
    return sum(max(1, len(text.split())) for text in texts)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model
    _model = SentenceTransformer(DEFAULT_MODEL, device=DEFAULT_DEVICE)
    yield
    _model = None


app = FastAPI(title="Sentence-Transformers Embedding Server", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embeddings(
    body: EmbeddingRequest,
    authorization: Annotated[str | None, Header()] = None,
) -> EmbeddingResponse:
    _require_api_key(authorization)

    texts = _normalize_input(body.input)
    model = _ensure_model_loaded()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    data = [
        EmbeddingItem(index=index, embedding=[float(v) for v in vector])
        for index, vector in enumerate(vectors)
    ]
    tokens = _estimate_tokens(texts)
    return EmbeddingResponse(
        data=data,
        model=DEFAULT_MODEL,
        usage=EmbeddingUsage(prompt_tokens=tokens, total_tokens=tokens),
    )
