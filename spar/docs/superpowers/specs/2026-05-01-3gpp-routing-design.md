# 3GPP Spec Number-Aware Routing & Retrieval

**Date:** 2026-05-01  
**Status:** Approved  
**Scope:** Task 2.2 (HybridRouter) + Task 1.4 (Hybrid Search) 부분 구현

---

## 1. 목표

3GPP 문서번호(`TS 29.502` 등)를 포함한 질의에서:
1. RegexRouter가 spec_number를 fast-path로 추출
2. Milvus expr 필터로 해당 문서 그룹 내 검색 범위 한정
3. BM25(키워드) + Dense(의미) + Reranker로 최종 청크 선정

---

## 2. 데이터 흐름

```
[3GPP .md files] data/tspec-llm/3GPP-clean/Rel-18/**/*.md
      ↓ scripts/slice_3gpp_intros.py (각 파일 앞 1000줄)
      → /tmp/3gpp_intros/<series>/<filename>
      ↓ scripts/run_ingest.py --input-dir /tmp/3gpp_intros --doc-type spec
      ↓ 파일명 파싱: 29502-i40.md → spec_number="29.502", series=29
[Milvus: spar_spec 컬렉션]
      ↓ (query 시)
[Query: "TS 29.502 session management 설명"]
      ↓ HybridRouter.route()
      ├─ Layer 1 RegexRouter: TS \d{2}[\.\s]?\d{3} → entities={spec_number: "29.502"}, route=DEFINITION_EXPLAIN
      └─ (regex hit → Layer 2/3 skip)
      ↓ retrieval.hybrid_search(query, expr="spec_number == '29.502'")
           ├─ BM25: query text (spec_number 토큰 포함)
           └─ Dense: query embedding
      ↓ RRF merge → Reranker
[최종 청크 반환]
```

---

## 3. 컴포넌트

### 3.1 scripts/slice_3gpp_intros.py (신규)

- `data/tspec-llm/3GPP-clean/Rel-18/**/*.md` 순회
- 각 파일 앞 1000줄 → `/tmp/3gpp_intros/<series>/<filename>`
- 파일명 파싱 규칙: `29502` → `"29.502"`, `38300` → `"38.300"` (앞 2자리 + `.` + 뒤 3자리)
- 파싱 실패 시 `spec_number=""` (무시하지 않고 ingest, 필터만 skip)

### 3.2 run_ingest.py 확장

- `--intro-only` 플래그 추가 (spec doc_type 전용)
- spec_number, series 메타데이터 파일명에서 자동 파싱 후 청크에 부착

### 3.3 Milvus 스키마 변경

`src/spar/retrieval/milvus_client.py` — `_build_schema()` 수정:

```python
FieldSchema(name="spec_number", dtype=DataType.VARCHAR, max_length=16),
# e.g. "29.502", "38.300", "" (unknown)
```

### 3.4 RegexRouter 확장

`src/spar/router/regex_router.py`:

```python
_SPEC_NUM_RE = re.compile(
    r"\b(?:3GPP\s+)?TS\s*(\d{2})[\.\s]?(\d{3})\b", re.IGNORECASE
)
```

매칭 시:
- `route = Route.DEFINITION_EXPLAIN`
- `entities = {"spec_number": "<xx>.<yyy>"}`
- `confidence = 1.0`, `layer = "regex"`

### 3.5 hybrid_search() 확장

`src/spar/retrieval/` — `expr` 파라미터 추가:

```python
async def hybrid_search(
    query: str,
    collection: str,
    top_k: int = 10,
    expr: str | None = None,   # 추가
) -> list[dict]: ...
```

spec_number entity 존재 시 caller(HybridRouter 또는 API layer)가 `expr="spec_number == '29.502'"` 전달.

**Fallback**: expr 결과 0건 → expr=None으로 재시도.

### 3.6 테스트 (tests/test_routing_3gpp.py)

**단위 테스트** (Milvus 불필요):

| 질의 | 기대 layer | 기대 route | 기대 spec_number |
|------|-----------|-----------|-----------------|
| `"TS 29.502 session management"` | regex | definition_explain | 29.502 |
| `"3GPP TS 38.300 NR architecture"` | regex | definition_explain | 38.300 |
| `"TS29502 what is SMF?"` | regex | definition_explain | 29.502 |
| `"session management in 5G core"` | — | — | (no match) |
| `"ALM-1234 alarm"` | regex | structured_lookup | (no spec_number) |

**통합 테스트** (Milvus mock):
- spec_number entity 있을 때 `hybrid_search` expr 파라미터 전달 검증
- expr 결과 0건 시 fallback 재시도 검증

---

## 4. 에러 처리

| 상황 | 처리 |
|------|------|
| 파일명 파싱 불가 | spec_number="" 로 ingest, 필터 skip |
| expr 결과 0건 | expr=None으로 fallback 재시도 |
| TS 번호 인식했으나 Milvus에 해당 spec 없음 | fallback 동일 |

---

## 5. 변경 파일 목록

| 파일 | 변경 유형 |
|------|---------|
| `scripts/slice_3gpp_intros.py` | 신규 |
| `scripts/run_ingest.py` | 수정 (`--intro-only`, spec 메타 파싱) |
| `src/spar/router/regex_router.py` | 수정 (TS 패턴 추가) |
| `src/spar/retrieval/milvus_client.py` | 수정 (`spec_number` 필드) |
| `src/spar/retrieval/hybrid_search.py` | 수정 (`expr` 파라미터) |
| `tests/test_routing_3gpp.py` | 신규 |

---

## 6. 미포함 (이번 스코프 외)

- HybridRouter → retrieval layer 직접 연결 (현재 API layer에서 entities 소비)
- Reranker 실제 연동 (cross-encoder 모델 필요, Task 1.5)
- 전체 문서 ingest (intro 1000줄만 임시 작업)
