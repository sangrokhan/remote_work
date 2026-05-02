# router/ — 3-Layer 쿼리 라우터

## 역할

사용자 질의를 6개 Route 중 하나로 분류.  
Regex → Embedding → LLM 순서로 시도하여 비용/지연 최소화.

## 파일 맵

| 파일 | 역할 |
|---|---|
| `schemas.py` | `Route` enum (6개), `RouteResult` dataclass |
| `regex_router.py` | Layer 1 — MO명/알람코드/파라미터명/카운터/스펙번호 정규식 fast-path |
| `embedding_router.py` | Layer 2 — 각 Route별 예시 문장 centroid 임베딩 코사인 유사도 |
| `llm_router.py` | Layer 3 — `router_system.txt` 프롬프트로 LLM 분류, JSON 응답 파싱 |
| `hybrid_router.py` | `HybridRouter` — 세 레이어 조합, threshold 설정 |
| `__init__.py` | `HybridRouter`, `Route`, `RouteResult` 재노출 |

## Route 목록

| Route | 대상 doc_type | 예시 |
|---|---|---|
| `structured_lookup` | parameter_ref, counter_ref, alarm_ref | "maxTxPower 기본값은?" |
| `definition_explain` | feature_desc, spec | "Carrier Aggregation이란?" |
| `procedural` | mop, install_guide | "RACH 파라미터 설정 방법" |
| `diagnostic` | alarm_ref, feature_desc | "HO 실패율이 높은 이유" |
| `comparative` | release_notes, feature_desc | "v7.0 vs v6.0 변경사항" |
| `default_rag` | feature_desc, spec, mop, ... | 기타 일반 질의 |

## RouteResult 스키마

```python
@dataclass
class RouteResult:
    route: Route
    confidence: float        # 0.0~1.0
    layer: str               # "regex" | "embedding" | "llm" | "llm_fallback" | "fallback"
    entities: dict[str, Any] # alarm_code, param_name, mo_name, counter_name, spec_number 등
    product: str | None      # 제품 필터 (Milvus expr)
    release: str | None      # 릴리스 필터 (Milvus expr)
    needs_decomposition: bool  # True → pipeline이 decompose 노드로 분기
```

## 임계값

| 레이어 | 파라미터 | 기본값 |
|---|---|---|
| Regex → 다음 레이어 | `_REGEX_THRESHOLD` | 0.9 |
| Embedding → LLM | `embedding_threshold` | 0.65 |

## 규약

- 새 Route 추가 시: `schemas.py` → `embedding_router.py` 예시 문장 → `routing.py` doc_type 매핑 순서로 수정
- regex_router는 부수효과 없음 — 매칭 실패 시 반드시 `None` 반환
