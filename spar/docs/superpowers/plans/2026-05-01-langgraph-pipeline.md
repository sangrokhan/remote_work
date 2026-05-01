# Plan: LangGraph 기반 SPAR 파이프라인 재구성

**작성일**: 2026-05-01  
**대상 브랜치**: `feat/langgraph-pipeline`  
**배경**: 현재 `app.py`의 flat 순차 파이프라인(`route → retrieve → generate`)을 LangGraph StateGraph로 전환. Route별 분기 추적, 노드별 재시도, iterative retrieval(Phase 5 대비)을 가능하게 한다.

---

## Requirements Summary

- 현재 `query_endpoint` 내 선형 파이프라인을 LangGraph graph로 교체
- 6개 Route 별 conditional edge로 분기 (`STRUCTURED_LOOKUP`, `DEFINITION_EXPLAIN`, `PROCEDURAL`, `DIAGNOSTIC`, `COMPARATIVE`, `DEFAULT_RAG`)
- 모든 노드 입출력이 `SparState` TypedDict를 통해 흐르므로 LangSmith/stdout에서 추적 가능
- 기존 컴포넌트(`HybridRouter`, `MilvusClient`, `CrossEncoderClient`, `LLMClient`)는 node wrapper만 추가, 내부 수정 없음
- `app.py`는 graph compile + invoke만 담당, 비즈니스 로직 없음

---

## Acceptance Criteria

- [ ] `SparState` TypedDict 정의 (`src/spar/pipeline/state.py`)
- [ ] 5개 node 함수 각각 `SparState → SparState` 시그니처 (async)
- [ ] `StateGraph` compile 후 `app.py`에서 `await graph.ainvoke(state)` 1줄로 실행
- [ ] LangSmith 없이도 `LANGGRAPH_TRACING=false` 환경에서 정상 동작
- [ ] 기존 `/health`, `/query` endpoint 스펙 변경 없음 (`QueryResponse` 그대로)
- [ ] `pytest` 통과 — node별 unit test + graph 통합 smoke test
- [ ] `mypy --strict` 통과

---

## 현재 구조 vs 목표 구조

### 현재 (`app.py`)
```
QueryRequest
  → HybridRouter.route()          # RouteResult
  → retrieve() [stub]             # list[dict]  ← route 무시
  → generate() [stub]             # str
  → QueryResponse
```

### 목표 (LangGraph)
```
QueryRequest → initial SparState
  → preprocess_node               # abbrev mapping
  → route_node                    # HybridRouter → state.route_result
  → [conditional edge by Route]
      ├─ STRUCTURED_LOOKUP → structured_retrieve_node  (Phase 3 KG/DB 대비 hook)
      ├─ DIAGNOSTIC        → multi_hop_retrieve_node   (iterative retrieval 확장 포인트)
      └─ 나머지 4개        → rag_retrieve_node         # Milvus hybrid_search
  → rerank_node                   # CrossEncoderClient (route 무관 공통)
  → generate_node                 # LLMClient
  → SparState → QueryResponse
```

---

## Implementation Steps

### Step 1 — 의존성 추가 (`pyproject.toml`)

추가할 패키지:
- `langgraph>=0.2` — StateGraph, conditional edges, ainvoke
- `langchain-core>=0.3` — LangSmith callback 연동 (선택)

`langgraph`는 `langchain` 전체를 끌어오지 않으므로 overhead 낮음.

---

### Step 2 — `SparState` 정의

**파일**: `src/spar/pipeline/state.py` (신규)

```python
from __future__ import annotations
from typing import TypedDict, Any
from spar.router.schemas import RouteResult

class SparState(TypedDict, total=False):
    # input
    query: str
    product: str | None
    release: str | None
    top_k: int
    request_id: str

    # preprocess
    expanded_query: str              # abbrev_mapper 결과

    # routing
    route_result: RouteResult

    # retrieval
    raw_chunks: list[dict[str, Any]]        # Milvus 결과
    reranked_chunks: list[dict[str, Any]]   # CrossEncoder 정렬 후

    # generation
    answer: str

    # observability
    error: str | None
    node_trace: list[str]            # 거친 노드 이름 순서
```

`total=False` → 노드가 점진적으로 채움. 각 노드 내부에서 키 존재 assert.

---

### Step 3 — Node 구현

**파일**: `src/spar/pipeline/nodes.py` (신규)

**3a. preprocess_node**
- `AbbrevMapper().expand_query(state["query"])` 호출
- 결과를 `expanded_query`에 저장

**3b. route_node**
- `HybridRouter.route(expanded_query)` 호출
- `route_result`에 `RouteResult` 저장

**3c. rag_retrieve_node**
- `MilvusClient.hybrid_search(expanded_query, product, release, top_k)` 호출
- 결과를 `raw_chunks`에 저장
- **현재 `retrieve()` stub을 이 노드로 완전 대체**

**3d. structured_retrieve_node** (Phase 3 대비 stub)
- 현재는 `rag_retrieve_node` fallback
- Phase 3 KG/DB 구현 후 swap

**3e. multi_hop_retrieve_node** (Phase 5 대비 stub)
- 현재는 `rag_retrieve_node` fallback
- Phase 5에서 LangGraph `Send` API 또는 SubGraph로 iterative retrieval 확장

**3f. rerank_node**
- `CrossEncoderClient.rerank(query, [c["text"] for c in raw_chunks])` 호출
- score 내림차순 정렬 후 `reranked_chunks`에 저장
- **`src/spar/reranker/client.py`가 구현됐지만 파이프라인 미연결 → 이 노드에서 첫 연결**

**3g. generate_node**
- `reranked_chunks` (없으면 `raw_chunks`) + query → LLMClient 호출
- `answer`에 저장

---

### Step 4 — Graph 조립

**파일**: `src/spar/pipeline/graph.py` (신규)

```python
from langgraph.graph import StateGraph, END
from spar.pipeline.state import SparState
from spar.router.schemas import Route

def route_selector(state: SparState) -> str:
    route = state["route_result"].route
    if route == Route.STRUCTURED_LOOKUP:
        return "structured_retrieve"
    if route == Route.DIAGNOSTIC:
        return "multi_hop_retrieve"
    return "rag_retrieve"

def build_graph(router, milvus, reranker, llm):
    nodes = make_nodes(router, milvus, reranker, llm)  # 클로저로 의존성 주입

    g = StateGraph(SparState)
    g.add_node("preprocess", nodes.preprocess)
    g.add_node("route", nodes.route)
    g.add_node("rag_retrieve", nodes.rag_retrieve)
    g.add_node("structured_retrieve", nodes.structured_retrieve)
    g.add_node("multi_hop_retrieve", nodes.multi_hop_retrieve)
    g.add_node("rerank", nodes.rerank)
    g.add_node("generate", nodes.generate)

    g.set_entry_point("preprocess")
    g.add_edge("preprocess", "route")
    g.add_conditional_edges("route", route_selector, {
        "rag_retrieve": "rag_retrieve",
        "structured_retrieve": "structured_retrieve",
        "multi_hop_retrieve": "multi_hop_retrieve",
    })
    g.add_edge("rag_retrieve", "rerank")
    g.add_edge("structured_retrieve", "rerank")
    g.add_edge("multi_hop_retrieve", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", END)

    return g.compile()
```

---

### Step 5 — `app.py` 교체

**파일**: `src/spar/api/app.py` (수정)

lifespan에서 `build_graph(router, milvus, reranker, llm)` 호출로 graph 초기화.

```python
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    t0 = time.monotonic()
    initial_state: SparState = {
        "query": req.query, "product": req.product,
        "release": req.release, "top_k": req.top_k,
        "request_id": str(uuid.uuid4()),
    }
    final_state = await _graph.ainvoke(initial_state)
    return QueryResponse(
        request_id=final_state["request_id"],
        query=final_state["query"],
        route=final_state["route_result"].route.value,
        answer=final_state["answer"],
        sources=final_state.get("reranked_chunks", []),
        latency_ms=round((time.monotonic() - t0) * 1000, 1),
    )
```

`retrieve()`, `generate()` stub 함수 삭제.

---

### Step 6 — 테스트

**파일**: `tests/pipeline/test_nodes.py`, `tests/pipeline/test_graph.py` (신규)

- 각 node: mock 의존성으로 unit test
- graph smoke test: `STRUCTURED_LOOKUP` / `DIAGNOSTIC` / `DEFAULT_RAG` 경로 각각 end-to-end 확인
- `node_trace` 리스트로 실제 분기 경로 assertion

---

## Refactoring Points (우선순위 순)

| # | 위치 | 현재 문제 | 변경 내용 |
|---|------|----------|----------|
| R1 | `app.py:retrieve()` | stub, route 무시 | 삭제 → `rag_retrieve_node`로 |
| R2 | `app.py:generate()` | stub | 삭제 → `generate_node`로 |
| R3 | `app.py:_router` | global 싱글톤 | `build_graph()` 인자로 DI |
| R4 | `src/spar/reranker/` | 구현됐지만 파이프라인 미연결 | `rerank_node`에서 첫 연결 |
| R5 | `src/spar/retrieval/milvus_client.py` | schema만 있고 `hybrid_search()` stub | `rag_retrieve_node`에서 실 호출 위해 구현 |
| R6 | `src/spar/llm/` | `LLMFactory` 존재, 실제 generate 없음 | `generate_node`에서 vLLM 호출 추가 |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| `langgraph` async + FastAPI 이벤트루프 충돌 | `asyncio.get_running_loop()` 사용. LangGraph `ainvoke`는 FastAPI 내에서 직접 await 가능 |
| `SparState` total=False → 런타임 KeyError | 각 노드 진입 시 필수 키 assert, typing_extensions `Required` 활용 |
| Phase 3 KG/DB 없는 상태에서 STRUCTURED_LOOKUP | `structured_retrieve_node`가 `rag_retrieve_node` fallback — Phase 3 때 swap |
| LangSmith API key 없는 환경 | `LANGCHAIN_TRACING_V2=false` 기본값, key 없어도 동작 |

---

## Verification Steps

```bash
# 의존성 설치
pip install -e ".[dev]"

# 타입 체크
mypy src/spar/pipeline/ --strict

# 테스트
pytest tests/pipeline/ -v

# 서버 기동 후 smoke
uvicorn spar.api.app:app --reload
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is HO TTT parameter?"}'
```

---

## 파일 목록 (신규/수정)

| 파일 | 상태 |
|------|------|
| `pyproject.toml` | 수정 — `langgraph`, `langchain-core` 추가 |
| `src/spar/pipeline/__init__.py` | 신규 |
| `src/spar/pipeline/state.py` | 신규 — `SparState` TypedDict |
| `src/spar/pipeline/nodes.py` | 신규 — 7개 node 함수 |
| `src/spar/pipeline/graph.py` | 신규 — `build_graph()`, `route_selector()` |
| `src/spar/api/app.py` | 수정 — graph DI, stub 함수 제거 |
| `tests/pipeline/test_nodes.py` | 신규 |
| `tests/pipeline/test_graph.py` | 신규 |
