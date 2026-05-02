# pipeline/ — LangGraph RAG 파이프라인

## 역할

LangGraph `StateGraph`로 RAG 파이프라인을 구성.  
`GraphConfig` 플래그로 기능 계층(baseline → verify_loop)을 선택적으로 활성화.

## 파일 맵

| 파일 | 역할 |
|---|---|
| `state.py` | `SparState` TypedDict — 모든 노드가 공유하는 불변 패싱 상태 |
| `config.py` | `GraphConfig` dataclass + `PRESET_CONFIGS` 목록 |
| `nodes.py` | `Nodes` dataclass — 노드 구현체 전체 (preprocess~tool_call) |
| `graph.py` | `build_graph()` — StateGraph 조립 및 컴파일 |
| `__init__.py` | `build_graph`, `SparState`, `GraphConfig` 재노출 |

## 파이프라인 흐름

```
preprocess → rewrite_query → prepare_context → route
    ↓ (route_selector)
decompose → decomposed_retrieve ─┐
rag_retrieve ───────────────────┤
structured_retrieve ────────────┤→ [tool_call] → rerank → generate → [verify → tool_call 루프]
multi_hop_retrieve ─────────────┘
```

## SparState 주요 필드

| 필드 | 방향 | 설명 |
|---|---|---|
| `query` | in | 원본 사용자 질의 |
| `expanded_query` | preprocess→ | 약어 확장 질의 |
| `rewritten_query` | rewrite→ | LLM 재작성 질의 |
| `route_result` | route→ | `RouteResult` |
| `raw_chunks` | retrieve→ | 검색 결과 청크 |
| `reranked_chunks` | rerank→ | 리랭크 후 청크 |
| `answer` | generate→ | 최종 답변 |
| `verify_score` | verify→ | 1~5점 자기 평가 |
| `node_trace` | 누적 | 실행된 노드 이름 목록 |
| `node_timings` | 누적 | 노드별 실행 시간(ms) |

## GraphConfig 프리셋

| 이름 | 활성 기능 |
|---|---|
| `baseline` | route + retrieve + generate |
| `+reranker` | + rerank |
| `full_retrieval` | + query_expansion + prepare_context + rerank |
| `e2e` | full_retrieval + real LLM generate |
| `verify_loop` | e2e + verify + tool_call retry |

## 규약

- 노드 함수는 `state: SparState → SparState` 시그니처 유지 (LangGraph 불변성)
- 새 노드 추가 시 `nodes.py` 구현 → `graph.py` 등록 → `state.py` 필드 추가 순서
- `node_timings`에 반드시 ms 단위로 기록
