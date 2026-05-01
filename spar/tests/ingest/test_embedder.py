import os
import pytest

from spar.ingest.embedder import Embedder


def test_embedder_dim_matches_milvus_constant():
    """EMBED_DIM(=1024)과 모델 차원 일치."""
    from spar.retrieval.milvus_client import EMBED_DIM
    assert EMBED_DIM == 1024


def test_embedder_normalizes_unit_norm(monkeypatch):
    """encode 결과는 L2-norm == 1 (정규화 보장)."""
    class _StubST:
        def __init__(self, name): self.name = name
        def encode(self, texts, batch_size=32, normalize_embeddings=True,
                   show_progress_bar=False):
            import numpy as np
            arr = np.ones((len(texts), 1024), dtype="float32")
            if normalize_embeddings:
                arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
            return arr

    monkeypatch.setattr("spar.ingest.embedder.SentenceTransformer", _StubST)
    e = Embedder(model_name="stub")
    vecs = e.encode(["a", "b"])
    assert len(vecs) == 2 and len(vecs[0]) == 1024
    import math
    assert all(abs(sum(v * v for v in vec) - 1.0) < 1e-5 for vec in vecs)


def test_embedder_remote_embedding(monkeypatch):
    """EMBEDDING_URL이 있으면 원격 임베딩 API를 호출한다."""
    import types

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "data": [
                    {"index": 0, "embedding": [3.0, 4.0]},
                    {"index": 1, "embedding": [6.0, 8.0]},
                ]
            }

    called = types.SimpleNamespace(posted=False)

    class _Client:
        def post(self, *_, **kwargs):
            called.posted = True
            assert kwargs["json"]["model"] == "BAAI/bge-large-en-v1.5"
            assert kwargs["json"]["input"] == ["a", "b"]
            return _Response()

    monkeypatch.setenv("EMBEDDING_URL", "http://embedder-host:8000/v1")
    monkeypatch.setattr("spar.ingest.embedder.httpx.Client", _Client)

    e = Embedder()
    vecs = e.encode(["a", "b"])
    assert called.posted
    assert vecs == [
        [0.6, 0.8],
        [0.6, 0.8],
    ]


@pytest.mark.skipif(
    os.environ.get("RUN_HEAVY_TESTS") != "1",
    reason="실 모델 다운로드 필요 — RUN_HEAVY_TESTS=1로 활성화",
)
def test_embedder_real_model_smoke():
    e = Embedder(model_name=os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5"))
    vecs = e.encode(["hello world", "3GPP NR"])
    assert len(vecs) == 2 and len(vecs[0]) == 1024
