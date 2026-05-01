# BM25 Hybrid Search (Milvus Built-in) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Milvus 내장 BM25 함수를 사용해 기존 dense 검색에 sparse(BM25) 필드를 추가하고, RRF 기반 hybrid search API를 구현한다.

**Architecture:** `milvus_client.py`의 `_build_schema()`에 `sparse_vec` 필드와 BM25 Function을 추가해 인덱스 생성 시 자동으로 text→sparse 변환이 이루어지게 한다. 새 `hybrid_search()` 메서드는 `AnnSearchRequest` 두 개(dense + sparse)를 `RRFRanker`로 결합한다. 기존 `search()` (dense-only)는 하위 호환 유지.

**Tech Stack:** pymilvus 2.6.12, `Function` / `FunctionType.BM25`, `AnnSearchRequest`, `RRFRanker`, pytest + unittest.mock

---

## File Map

| 파일 | 변경 내용 |
|------|-----------|
| `src/spar/retrieval/milvus_client.py` | 수정 — sparse 필드, BM25 Function, sparse 인덱스, `hybrid_search()` |
| `tests/retrieval/__init__.py` | 새로 생성 — 빈 파일 |
| `tests/retrieval/test_milvus_client.py` | 새로 생성 — 스키마·hybrid search 유닛 테스트 |

---

### Task 1: 테스트 디렉토리 생성 + 스키마 변경 실패 테스트

**Files:**
- Create: `tests/retrieval/__init__.py`
- Create: `tests/retrieval/test_milvus_client.py`

- [ ] **Step 1: `tests/retrieval/__init__.py` 생성**

```bash
mkdir -p tests/retrieval
touch tests/retrieval/__init__.py
```

- [ ] **Step 2: 실패하는 스키마 테스트 작성**

`tests/retrieval/test_milvus_client.py`:

```python
"""Unit tests for SparMilvusClient schema and hybrid search."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from spar.retrieval.milvus_client import _build_schema, SPARSE_INDEX_PARAMS


class TestSchema:
    def test_sparse_vec_field_exists(self):
        schema = _build_schema()
        field_names = [f.name for f in schema.fields]
        assert "sparse_vec" in field_names

    def test_text_field_has_analyzer_enabled(self):
        schema = _build_schema()
        text_field = next(f for f in schema.fields if f.name == "text")
        assert text_field.params.get("enable_analyzer") is True

    def test_bm25_function_registered(self):
        schema = _build_schema()
        # CollectionSchema.functions 는 list[Function]
        fn_names = [fn.name for fn in schema.functions]
        assert "bm25" in fn_names

    def test_bm25_function_maps_text_to_sparse(self):
        schema = _build_schema()
        bm25_fn = next(fn for fn in schema.functions if fn.name == "bm25")
        assert bm25_fn.input_field_names == ["text"]
        assert bm25_fn.output_field_names == ["sparse_vec"]

    def test_sparse_index_params_metric_type(self):
        assert SPARSE_INDEX_PARAMS["metric_type"] == "BM25"
        assert SPARSE_INDEX_PARAMS["index_type"] == "SPARSE_INVERTED_INDEX"
```

- [ ] **Step 3: 실패 확인**

```bash
pytest tests/retrieval/test_milvus_client.py::TestSchema -v 2>&1 | tail -20
```

Expected: `ImportError` 또는 `AssertionError` — `sparse_vec`, `SPARSE_INDEX_PARAMS` 아직 없음.

---

### Task 2: 스키마에 BM25 sparse 필드 + Function 추가

**Files:**
- Modify: `src/spar/retrieval/milvus_client.py`

- [ ] **Step 1: import에 `Function`, `FunctionType` 추가**

현재 import 블록 (`from pymilvus import ...`) 에 두 심볼 추가:

```python
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusClient,
    connections,
    utility,
)
```

- [ ] **Step 2: `SPARSE_INDEX_PARAMS` 상수 추가**

`SEARCH_PARAMS` 정의 바로 아래에 추가:

```python
SPARSE_INDEX_PARAMS = {
    "index_type": "SPARSE_INVERTED_INDEX",
    "metric_type": "BM25",
}

SPARSE_SEARCH_PARAMS = {"metric_type": "BM25"}
```

- [ ] **Step 3: `_build_schema()` 수정**

`text` 필드에 `enable_analyzer=True` 추가, `sparse_vec` 필드 추가, BM25 Function 등록:

```python
def _build_schema(description: str = "") -> CollectionSchema:
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
        FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="product", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="release", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="deployment_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="mo_name", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source_doc", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="page", dtype=DataType.INT32),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=65_535,
            enable_analyzer=True,  # BM25 토크나이저 활성화
        ),
        FieldSchema(name="sparse_vec", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]
    schema = CollectionSchema(fields=fields, description=description, enable_dynamic_field=True)

    bm25_fn = Function(
        name="bm25",
        input_field_names=["text"],
        output_field_names=["sparse_vec"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_fn)
    return schema
```

- [ ] **Step 4: `create_collection()`에 sparse 인덱스 생성 추가**

기존 `col.create_index(field_name="embedding", ...)` 바로 아래에:

```python
col.create_index(field_name="embedding", index_params=HNSW_INDEX_PARAMS)
col.create_index(field_name="sparse_vec", index_params=SPARSE_INDEX_PARAMS)
col.load()
```

- [ ] **Step 5: 스키마 테스트 통과 확인**

```bash
pytest tests/retrieval/test_milvus_client.py::TestSchema -v 2>&1 | tail -20
```

Expected: 5개 PASS

- [ ] **Step 6: 커밋**

```bash
git add src/spar/retrieval/milvus_client.py tests/retrieval/__init__.py tests/retrieval/test_milvus_client.py
git commit -m "feat(retrieval): Milvus 스키마에 BM25 sparse 필드 + Function 추가"
```

---

### Task 3: `hybrid_search()` 실패 테스트 작성

**Files:**
- Modify: `tests/retrieval/test_milvus_client.py`

- [ ] **Step 1: hybrid search 테스트 추가**

`test_milvus_client.py` 파일 끝에 추가:

```python
class TestHybridSearch:
    def _make_mock_hit(self, chunk_id: str, score: float) -> MagicMock:
        hit = MagicMock()
        hit.id = chunk_id
        hit.score = score
        hit.fields = {"text": "sample text", "doc_type": "parameter_ref", "source_doc": "doc.md",
                      "section": "sec1", "page": 1, "product": "NR", "release": "v7"}
        return hit

    @patch("spar.retrieval.milvus_client.Collection")
    @patch("spar.retrieval.milvus_client.connections")
    @patch("spar.retrieval.milvus_client.utility")
    def test_hybrid_search_returns_ranked_results(self, mock_utility, mock_conn, mock_col_cls):
        mock_utility.has_collection.return_value = True
        mock_col = MagicMock()
        mock_col_cls.return_value = mock_col

        hit1 = self._make_mock_hit("c1", 0.9)
        hit2 = self._make_mock_hit("c2", 0.7)
        mock_col.hybrid_search.return_value = [[hit1, hit2]]

        from spar.retrieval.milvus_client import SparMilvusClient
        client = SparMilvusClient()
        results = client.hybrid_search(
            doc_type="parameter_ref",
            query_text="전압 임계값",
            query_vector=[0.1] * 1024,
            top_k=5,
        )

        assert len(results) == 2
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["score"] == 0.9
        mock_col.hybrid_search.assert_called_once()

    @patch("spar.retrieval.milvus_client.Collection")
    @patch("spar.retrieval.milvus_client.connections")
    @patch("spar.retrieval.milvus_client.utility")
    def test_hybrid_search_uses_rrf_ranker(self, mock_utility, mock_conn, mock_col_cls):
        from pymilvus import RRFRanker
        mock_utility.has_collection.return_value = True
        mock_col = MagicMock()
        mock_col_cls.return_value = mock_col
        mock_col.hybrid_search.return_value = [[]]

        from spar.retrieval.milvus_client import SparMilvusClient
        client = SparMilvusClient()
        client.hybrid_search("parameter_ref", "test query", [0.0] * 1024, top_k=3)

        call_kwargs = mock_col.hybrid_search.call_args
        ranker = call_kwargs.kwargs.get("ranker") or call_kwargs.args[1]
        assert isinstance(ranker, RRFRanker)

    @patch("spar.retrieval.milvus_client.Collection")
    @patch("spar.retrieval.milvus_client.connections")
    @patch("spar.retrieval.milvus_client.utility")
    def test_hybrid_search_passes_two_ann_requests(self, mock_utility, mock_conn, mock_col_cls):
        from pymilvus import AnnSearchRequest
        mock_utility.has_collection.return_value = True
        mock_col = MagicMock()
        mock_col_cls.return_value = mock_col
        mock_col.hybrid_search.return_value = [[]]

        from spar.retrieval.milvus_client import SparMilvusClient
        client = SparMilvusClient()
        client.hybrid_search("parameter_ref", "test query", [0.0] * 1024)

        reqs = mock_col.hybrid_search.call_args.kwargs.get("reqs") or mock_col.hybrid_search.call_args.args[0]
        assert len(reqs) == 2
        fields = [r.anns_field for r in reqs]
        assert "embedding" in fields
        assert "sparse_vec" in fields
```

- [ ] **Step 2: 실패 확인**

```bash
pytest tests/retrieval/test_milvus_client.py::TestHybridSearch -v 2>&1 | tail -20
```

Expected: `AttributeError` — `SparMilvusClient` 에 `hybrid_search` 없음.

---

### Task 4: `hybrid_search()` 구현

**Files:**
- Modify: `src/spar/retrieval/milvus_client.py`

- [ ] **Step 1: import에 `AnnSearchRequest`, `RRFRanker` 추가**

```python
from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
    connections,
    utility,
)
```

- [ ] **Step 2: `SparMilvusClient`에 `hybrid_search()` 추가**

기존 `search()` 메서드 바로 아래에 추가:

```python
def hybrid_search(
    self,
    doc_type: str,
    query_text: str,
    query_vector: list[float],
    top_k: int = 10,
    output_fields: list[str] | None = None,
    expr: str | None = None,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Dense + BM25 sparse hybrid search using RRF ranking.

    Args:
        doc_type: 컬렉션 종류 (DOC_TYPES 중 하나).
        query_text: BM25 검색용 원문 텍스트.
        query_vector: dense embedding 벡터 (dim=EMBED_DIM).
        top_k: 반환할 결과 수.
        output_fields: 반환할 필드 목록.
        expr: Milvus 필터 표현식 (예: 'product == "NR"').
        rrf_k: RRF 공식의 k 파라미터 (기본 60).
    """
    col = self.get_collection(doc_type)
    if output_fields is None:
        output_fields = [
            "chunk_id", "doc_type", "product", "release",
            "source_doc", "section", "page", "text",
        ]

    dense_req = AnnSearchRequest(
        data=[query_vector],
        anns_field="embedding",
        param=SEARCH_PARAMS,
        limit=top_k * 2,
        expr=expr,
    )
    sparse_req = AnnSearchRequest(
        data=[query_text],
        anns_field="sparse_vec",
        param=SPARSE_SEARCH_PARAMS,
        limit=top_k * 2,
        expr=expr,
    )

    results = col.hybrid_search(
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=rrf_k),
        limit=top_k,
        output_fields=output_fields,
    )
    return [
        {"chunk_id": hit.id, "score": hit.score, **hit.fields}
        for hit in results[0]
    ]
```

- [ ] **Step 3: 전체 테스트 통과 확인**

```bash
pytest tests/retrieval/test_milvus_client.py -v 2>&1 | tail -30
```

Expected: 8개 PASS

- [ ] **Step 4: 커밋**

```bash
git add src/spar/retrieval/milvus_client.py tests/retrieval/test_milvus_client.py
git commit -m "feat(retrieval): Milvus hybrid_search() — BM25 sparse + dense RRF 결합"
```

---

### Task 5: 전체 테스트 회귀 확인

**Files:**
- 없음 (기존 테스트 실행만)

- [ ] **Step 1: 전체 테스트 실행**

```bash
pytest --tb=short -q 2>&1 | tail -20
```

Expected: 기존 테스트 포함 전부 PASS, 새 테스트 8개 추가.

- [ ] **Step 2: 최종 커밋 (필요 시)**

회귀 없으면 추가 커밋 불필요.

---

## Self-Review

**Spec coverage:**
- ✅ BM25 인덱스 구현 (Milvus 내장 FunctionType.BM25)
- ✅ Hybrid 결합 RRF (RRFRanker)
- ✅ 파라미터/카운터 검색 BM25 가중치 조정 가능 (`rrf_k` 파라미터로 제어)
- ✅ 기존 dense `search()` 하위 호환 유지

**주의사항:**
- 기존 컬렉션(sparse_vec 필드 없음)에서 `hybrid_search()` 호출 시 오류 발생. `create_collection(drop_if_exists=True)`로 재생성 필요.
- Milvus 서버 버전 2.5+ 필요 (BM25 Function 서버 측 지원). `docker pull milvusdb/milvus:v2.5.0` 이상 사용.
