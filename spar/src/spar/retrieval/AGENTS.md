# retrieval/ — 검색 및 쿼리 처리 모듈

## 역할

Milvus 하이브리드 검색, 쿼리 재작성/분해, Route→doc_type 매핑, 알람 직접 조회.

## 파일 맵

| 파일 | 역할 |
|---|---|
| `milvus_client.py` | `SparMilvusClient` — hybrid_search (dense+BM25), 컬렉션 관리 |
| `routing.py` | `doc_types_for_route()`, `build_expr()` — Route→Milvus 필터 변환 |
| `query_rewriter.py` | `rewrite_query()`, `build_context()` — LLM 기반 질의 재작성 + 복잡도 분류 |
| `query_decomposer.py` | `QueryDecomposer.decompose()` — 복합 질의를 최대 4개 서브질의로 분해 |
| `alarm_index.py` | `AlarmIndex` 싱글톤 — alarm_code 직접 lookup (벡터 검색 우회) |

## doc_type → Milvus 컬렉션 매핑

| doc_type | 내용 |
|---|---|
| `parameter_ref` | 파라미터 레퍼런스 엑셀 |
| `counter_ref` | 카운터 레퍼런스 엑셀 |
| `alarm_ref` | 알람 레퍼런스 엑셀 |
| `feature_desc` | Feature Description (DOCX/PDF) |
| `spec` | 3GPP TS 스펙 (MD) |
| `mop` | Method of Procedure |
| `install_guide` | 설치/운영 가이드 |
| `release_notes` | 릴리스 노트 |

## 쿼리 재작성 출력 스키마

```python
@dataclass
class QueryRewriteResult:
    rewritten: str          # 재작성된 질의
    complexity: str         # "simple" | "complex"
```

## 규약

- `milvus_client.py`는 동기 API — pipeline/nodes.py에서 `asyncio.run_in_executor`로 호출
- `routing.py`는 순수 함수 — 상태 없음, 테스트 용이
- `alarm_index.py` 싱글톤은 `get_alarm_index()`로만 접근
- 새 doc_type 추가 시 `routing.py`의 `_ROUTE_DOC_TYPES`와 `_ENTITY_PRIORITY` 동시 갱신
