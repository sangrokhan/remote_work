# Milvus Retrieval Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `rag_retrieve` / `structured_retrieve` 노드를 실제 `SparMilvusClient.hybrid_search()`에 연결해 stub 제거

**Architecture:** Route 결과(`route`, `entities`, `product`, `release`)를 기반으로 검색 대상 doc_type 컬렉션과 Milvus filter expr을 결정하는 매핑 레이어를 추가한다. `Nodes`에 `EncoderClient`와 `SparMilvusClient`를 주입해 각 retrieve 노드에서 실제 검색을 수행한다.

**Tech Stack:** pymilvus, `SparMilvusClient` (기존), `EncoderClient` (기존), LangGraph `SparState`

---

## File Map

| 파일 | 역할 |
|---|---|
| `src/spar/retrieval/routing.py` (신규) | Route → doc_type 목록 + expr 빌더 |
| `src/spar/pipeline/nodes.py` (수정) | encoder+milvus 주입, retrieve stub 제거 |
| `src/spar/pipeline/graph.py` (수정) | `build_graph()` 시그니처에 encoder+milvus 추가 |
| `src/spar/api/app.py` (수정) | 앱 시작 시 `SparMilvusClient` 초기화 + 주입 |
| `tests/retrieval/test_routing.py` (신규) | 매핑 로직 단위 테스트 |
| `tests/pipeline/test_nodes_retrieval.py` (신규) | Milvus/encoder mock 통합 테스트 |

---

### Task 1: Route → doc_type 매핑 + expr 빌더

**Files:**
- Create: `src/spar/retrieval/routing.py`
- Create: `tests/retrieval/test_routing.py`

Route별 검색 대상 컬렉션과 Milvus scalar filter expr을 결정하는 순수 함수 레이어.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/retrieval/test_routing.py
import pytest
from spar.router.schemas import Route, RouteResult
from spar.retrieval.routing import doc_types_for_route, build_expr


def _result(route, entities=None, product=None, release=None):
    return RouteResult(
        route=route, confidence=1.0, layer="test",
        entities=entities or {}, product=product, release=release,
    )


def test_structured_lookup_alarm():
    types = doc_types_for_route(_result(Route.STRUCTURED_LOOKUP, entities={"alarm_code": "ALM-123"}))
    assert types == ["alarm_ref"]


def test_structured_lookup_param():
    types = doc_types_for_route(_result(Route.STRUCTURED_LOOKUP, entities={"param_name": "maxTxPower"}))
    assert types == ["parameter_ref"]


def test_structured_lookup_mo():
    types = doc_types_for_route(_result(Route.STRUCTURED_LOOKUP, entities={"mo_name": "NRCellDU"}))
    assert types == ["parameter_ref"]


def test_structured_lookup_no_entities_fallback():
    types = doc_types_for_route(_result(Route.STRUCTURED_LOOKUP))
    assert set(types) == {"parameter_ref", "counter_ref", "alarm_ref"}


def test_definition_explain():
    types = doc_types_for_route(_result(Route.DEFINITION_EXPLAIN))
    assert set(types) == {"feature_desc", "spec"}


def test_procedural():
    types = doc_types_for_route(_result(Route.PROCEDURAL))
    assert set(types) == {"mop", "install_guide"}


def test_diagnostic():
    types = doc_types_for_route(_result(Route.DIAGNOSTIC))
    assert set(types) == {"alarm_ref", "feature_desc"}


def test_comparative():
    types = doc_types_for_route(_result(Route.COMPARATIVE))
    assert set(types) == {"release_notes", "feature_desc"}


def test_default_rag():
    types = doc_types_for_route(_result(Route.DEFAULT_RAG))
    assert "feature_desc" in types


def test_build_expr_alarm():
    expr = build_expr(_result(Route.STRUCTURED_LOOKUP, entities={"alarm_code": "ALM-123"}))
    assert expr == 'mo_name == "ALM-123"'


def test_build_expr_param():
    expr = build_expr(_result(Route.STRUCTURED_LOOKUP, entities={"param_name": "maxTxPower"}))
    assert expr is None  # param_name은 텍스트 검색으로 충분


def test_build_expr_product_filter():
    expr = build_expr(_result(Route.DEFAULT_RAG, product="NR"))
    assert expr == 'product == "NR"'


def test_build_expr_product_and_release():
    expr = build_expr(_result(Route.DEFAULT_RAG, product="LTE", release="v6.0"))
    assert 'product == "LTE"' in expr
    assert 'release == "v6.0"' in expr


def test_build_expr_none_when_no_filters():
    expr = build_expr(_result(Route.DEFAULT_RAG))
    assert expr is None
```

- [ ] **Step 2: 실패 확인**

```bash
cd /home/han/.openclaw/workspace/remote_work/spar
python -m pytest tests/retrieval/test_routing.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError` 또는 `ImportError`

- [ ] **Step 3: `routing.py` 구현**

```python
# src/spar/retrieval/routing.py
from __future__ import annotations

from spar.router.schemas import Route, RouteResult

_ROUTE_DOC_TYPES: dict[Route, list[str]] = {
    Route.STRUCTURED_LOOKUP: ["parameter_ref", "counter_ref", "alarm_ref"],
    Route.DEFINITION_EXPLAIN: ["feature_desc", "spec"],
    Route.PROCEDURAL: ["mop", "install_guide"],
    Route.DIAGNOSTIC: ["alarm_ref", "feature_desc"],
    Route.COMPARATIVE: ["release_notes", "feature_desc"],
    Route.DEFAULT_RAG: ["feature_desc", "spec", "mop", "install_guide", "release_notes"],
}

_ENTITY_TO_DOC_TYPE: dict[str, str] = {
    "alarm_code": "alarm_ref",
    "param_name": "parameter_ref",
    "mo_name": "parameter_ref",
}


def doc_types_for_route(result: RouteResult) -> list[str]:
    """Route + entities → 검색 대상 doc_type 목록."""
    if result.route == Route.STRUCTURED_LOOKUP and result.entities:
        for key, doc_type in _ENTITY_TO_DOC_TYPE.items():
            if key in result.entities:
                return [doc_type]
    return _ROUTE_DOC_TYPES.get(result.route, ["feature_desc"])


def build_expr(result: RouteResult) -> str | None:
    """RouteResult → Milvus scalar filter expr (없으면 None)."""
    clauses: list[str] = []

    entities = result.entities or {}
    if "alarm_code" in entities:
        clauses.append(f'mo_name == "{entities["alarm_code"]}"')

    if result.product and result.product != "both":
        clauses.append(f'product == "{result.product}"')
    if result.release:
        clauses.append(f'release == "{result.release}"')

    return " && ".join(clauses) if clauses else None
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/retrieval/test_routing.py -v
```

Expected: 전체 PASS

- [ ] **Step 5: 커밋**

```bash
git add src/spar/retrieval/routing.py tests/retrieval/test_routing.py
git commit -m "feat(retrieval): Route→doc_type 매핑 + Milvus expr 빌더"
```

---

### Task 2: `Nodes`에 encoder + milvus 주입 및 `rag_retrieve` 실구현

**Files:**
- Modify: `src/spar/pipeline/nodes.py`
- Create: `tests/pipeline/test_nodes_retrieval.py`

`Nodes` dataclass에 `EncoderClient`와 `SparMilvusClient`를 추가하고, `rag_retrieve`에서 query를 임베딩한 뒤 `hybrid_search`를 호출한다. 여러 doc_type이 대상이면 병렬 검색 후 score 기준으로 합산한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/pipeline/test_nodes_retrieval.py
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spar.pipeline.nodes import Nodes
from spar.pipeline.state import SparState
from spar.router.schemas import Route, RouteResult


def _make_nodes():
    router = MagicMock()
    reranker = MagicMock()
    encoder = MagicMock()
    encoder.embed.return_value = [[0.1] * 1024]
    milvus = MagicMock()
    milvus.hybrid_search.return_value = [
        {"chunk_id": "c1", "score": 0.9, "text": "real chunk"}
    ]
    return Nodes.create(
        router=router,
        reranker=reranker,
        encoder=encoder,
        milvus=milvus,
    ), encoder, milvus


def _state(route=Route.DEFAULT_RAG, entities=None, product=None, release=None):
    return SparState(
        query="test query",
        expanded_query="test query expanded",
        route_result=RouteResult(
            route=route, confidence=0.9, layer="test",
            entities=entities or {}, product=product, release=release,
        ),
    )


@pytest.mark.asyncio
async def test_rag_retrieve_calls_hybrid_search():
    nodes, encoder, milvus = _make_nodes()
    state = _state()
    result = await nodes.rag_retrieve(state)

    encoder.embed.assert_called_once()
    assert milvus.hybrid_search.called
    assert len(result["raw_chunks"]) > 0
    assert result["raw_chunks"][0]["text"] == "real chunk"


@pytest.mark.asyncio
async def test_rag_retrieve_not_stub():
    nodes, _, _ = _make_nodes()
    state = _state()
    result = await nodes.rag_retrieve(state)
    assert "[stub]" not in result["raw_chunks"][0]["text"]


@pytest.mark.asyncio
async def test_structured_retrieve_alarm_targets_alarm_ref():
    nodes, encoder, milvus = _make_nodes()
    state = _state(
        route=Route.STRUCTURED_LOOKUP,
        entities={"alarm_code": "ALM-1234"},
    )
    await nodes.structured_retrieve(state)
    call_args = milvus.hybrid_search.call_args
    assert call_args[1]["doc_type"] == "alarm_ref" or call_args[0][0] == "alarm_ref"


@pytest.mark.asyncio
async def test_rag_retrieve_passes_product_filter():
    nodes, encoder, milvus = _make_nodes()
    state = _state(product="NR")
    await nodes.rag_retrieve(state)
    call_kwargs = milvus.hybrid_search.call_args[1]
    assert call_kwargs.get("expr") == 'product == "NR"'
```

- [ ] **Step 2: 실패 확인**

```bash
python -m pytest tests/pipeline/test_nodes_retrieval.py -v 2>&1 | head -40
```

Expected: `TypeError` (Nodes.create 시그니처 불일치) 또는 `AssertionError`

- [ ] **Step 3: `nodes.py` 수정**

```python
# src/spar/pipeline/nodes.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spar.encoder.base import EncoderClient
from spar.pipeline.state import SparState
from spar.preprocessing.abbrev_mapper import (
    build_reverse_index,
    expand_query,
    load_acronyms,
)
from spar.reranker.client import CrossEncoderClient
from spar.retrieval.milvus_client import SparMilvusClient
from spar.retrieval.routing import build_expr, doc_types_for_route
from spar.router.hybrid_router import HybridRouter

_ACRONYMS_PATH = Path(__file__).parent.parent.parent.parent.parent / "dictionary" / "acronyms.json"


def _append_trace(state: SparState, node: str) -> list[str]:
    return [*state.get("node_trace", []), node]


@dataclass
class Nodes:
    router: HybridRouter
    reranker: CrossEncoderClient
    encoder: EncoderClient
    milvus: SparMilvusClient
    _acronyms: dict
    _reverse_index: dict[str, str]

    @classmethod
    def create(
        cls,
        router: HybridRouter,
        reranker: CrossEncoderClient,
        encoder: EncoderClient,
        milvus: SparMilvusClient,
        acronyms_path: Path | None = None,
    ) -> "Nodes":
        path = acronyms_path or _ACRONYMS_PATH
        if path.exists():
            acronyms = load_acronyms(path)
            reverse_index = build_reverse_index(acronyms)
        else:
            acronyms, reverse_index = {}, {}
        return cls(
            router=router,
            reranker=reranker,
            encoder=encoder,
            milvus=milvus,
            _acronyms=acronyms,
            _reverse_index=reverse_index,
        )

    async def preprocess(self, state: SparState) -> SparState:
        query = state["query"]
        expanded = expand_query(query, self._acronyms, self._reverse_index)
        return {**state, "expanded_query": expanded, "node_trace": _append_trace(state, "preprocess")}

    async def route(self, state: SparState) -> SparState:
        query = state.get("expanded_query") or state["query"]
        result = await self.router.route(query)
        return {**state, "route_result": result, "node_trace": _append_trace(state, "route")}

    async def _hybrid_search_multi(
        self,
        doc_types: list[str],
        query_text: str,
        query_vector: list[float],
        top_k: int,
        expr: str | None,
    ) -> list[dict[str, Any]]:
        """여러 doc_type 병렬 검색 후 score 내림차순 합산."""
        async def _search_one(doc_type: str) -> list[dict[str, Any]]:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.milvus.hybrid_search(
                    doc_type=doc_type,
                    query_text=query_text,
                    query_vector=query_vector,
                    top_k=top_k,
                    expr=expr,
                ),
            )

        results = await asyncio.gather(*[_search_one(dt) for dt in doc_types])
        merged = [chunk for chunks in results for chunk in chunks]
        merged.sort(key=lambda c: c["score"], reverse=True)
        return merged[:top_k]

    async def rag_retrieve(self, state: SparState) -> SparState:
        query = state.get("expanded_query") or state["query"]
        route_result = state["route_result"]
        top_k = state.get("top_k", 10)

        query_vector: list[float] = self.encoder.embed([query])[0]
        doc_types = doc_types_for_route(route_result)
        expr = build_expr(route_result)

        chunks = await self._hybrid_search_multi(doc_types, query, query_vector, top_k, expr)
        return {**state, "raw_chunks": chunks, "node_trace": _append_trace(state, "rag_retrieve")}

    async def structured_retrieve(self, state: SparState) -> SparState:
        """entity 기반 정밀 검색 — STRUCTURED_LOOKUP 전용."""
        result = await self.rag_retrieve(state)
        return {**result, "node_trace": _append_trace(result, "structured_retrieve")}

    async def multi_hop_retrieve(self, state: SparState) -> SparState:
        # Phase 5: iterative retrieval via LangGraph Send — fallback to RAG
        result = await self.rag_retrieve(state)
        return {**result, "node_trace": _append_trace(result, "multi_hop_retrieve")}

    async def rerank(self, state: SparState) -> SparState:
        chunks = state.get("raw_chunks", [])
        if not chunks:
            return {**state, "reranked_chunks": [], "node_trace": _append_trace(state, "rerank")}
        query = state.get("expanded_query") or state["query"]
        scores = await self.reranker.rerank(query, [c["text"] for c in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        reranked = [{"score": s, **c} for c, s in ranked]
        return {**state, "reranked_chunks": reranked, "node_trace": _append_trace(state, "rerank")}

    async def generate(self, state: SparState) -> SparState:
        chunks = state.get("reranked_chunks") or state.get("raw_chunks", [])
        context = "\n\n".join(c["text"] for c in chunks[:5])
        query = state["query"]
        answer = f"[stub] context={len(chunks)} chunks\nquery={query}\n{context[:200]}"
        return {**state, "answer": answer, "node_trace": _append_trace(state, "generate")}
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/pipeline/test_nodes_retrieval.py -v
```

Expected: 전체 PASS

- [ ] **Step 5: 커밋**

```bash
git add src/spar/pipeline/nodes.py tests/pipeline/test_nodes_retrieval.py
git commit -m "feat(pipeline): rag_retrieve Milvus hybrid_search 연결, encoder+milvus 주입"
```

---

### Task 3: `build_graph()` 시그니처 + `app.py` 의존성 주입

**Files:**
- Modify: `src/spar/pipeline/graph.py`
- Modify: `src/spar/api/app.py`

`build_graph()`에 `encoder`와 `milvus` 파라미터를 추가하고, FastAPI lifespan에서 `SparMilvusClient`를 초기화해 주입한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/pipeline/test_graph.py 기존 파일에 추가
from unittest.mock import MagicMock
from spar.pipeline.graph import build_graph


def test_build_graph_accepts_encoder_and_milvus():
    router = MagicMock()
    reranker = MagicMock()
    encoder = MagicMock()
    milvus = MagicMock()
    graph = build_graph(router=router, reranker=reranker, encoder=encoder, milvus=milvus)
    assert graph is not None
```

- [ ] **Step 2: 실패 확인**

```bash
python -m pytest tests/pipeline/test_graph.py::test_build_graph_accepts_encoder_and_milvus -v
```

Expected: `TypeError` (파라미터 없음)

- [ ] **Step 3: `graph.py` 수정**

```python
# src/spar/pipeline/graph.py
from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, StateGraph

from spar.encoder.base import EncoderClient
from spar.pipeline.nodes import Nodes
from spar.pipeline.state import SparState
from spar.reranker.client import CrossEncoderClient
from spar.retrieval.milvus_client import SparMilvusClient
from spar.router.hybrid_router import HybridRouter
from spar.router.schemas import Route


def _route_selector(state: SparState) -> str:
    route = state["route_result"].route
    if route == Route.STRUCTURED_LOOKUP:
        return "structured_retrieve"
    if route == Route.DIAGNOSTIC:
        return "multi_hop_retrieve"
    return "rag_retrieve"


def build_graph(
    router: HybridRouter,
    reranker: CrossEncoderClient,
    encoder: EncoderClient,
    milvus: SparMilvusClient,
    acronyms_path: Path | None = None,
):
    nodes = Nodes.create(
        router=router,
        reranker=reranker,
        encoder=encoder,
        milvus=milvus,
        acronyms_path=acronyms_path,
    )

    g: StateGraph = StateGraph(SparState)
    g.add_node("preprocess", nodes.preprocess)
    g.add_node("route", nodes.route)
    g.add_node("rag_retrieve", nodes.rag_retrieve)
    g.add_node("structured_retrieve", nodes.structured_retrieve)
    g.add_node("multi_hop_retrieve", nodes.multi_hop_retrieve)
    g.add_node("rerank", nodes.rerank)
    g.add_node("generate", nodes.generate)

    g.set_entry_point("preprocess")
    g.add_edge("preprocess", "route")
    g.add_conditional_edges(
        "route",
        _route_selector,
        {
            "rag_retrieve": "rag_retrieve",
            "structured_retrieve": "structured_retrieve",
            "multi_hop_retrieve": "multi_hop_retrieve",
        },
    )
    g.add_edge("rag_retrieve", "rerank")
    g.add_edge("structured_retrieve", "rerank")
    g.add_edge("multi_hop_retrieve", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", END)

    return g.compile()
```

- [ ] **Step 4: `app.py` 확인 후 lifespan에 milvus 추가**

```bash
cat src/spar/api/app.py
```

`app.py`의 lifespan / 의존성 주입 패턴 확인 후, 아래 패턴으로 `SparMilvusClient` 초기화 추가:

```python
# src/spar/api/app.py — lifespan 또는 startup 핸들러 내부
from spar.retrieval.milvus_client import SparMilvusClient, MilvusConfig

# 기존 encoder, router, reranker 초기화 코드 아래에 추가
milvus_client = SparMilvusClient(MilvusConfig())

graph = build_graph(
    router=router,
    reranker=reranker,
    encoder=encoder,       # 기존 encoder 객체
    milvus=milvus_client,
)
```

> **주의:** app.py 실제 패턴을 보고 맞게 수정. lifespan context manager면 `yield` 전에 초기화, `yield` 후에 `milvus_client.close()` 호출.

- [ ] **Step 5: 기존 테스트 전체 통과 확인**

```bash
python -m pytest tests/ -v --ignore=tests/retrieval/test_routing.py --ignore=tests/pipeline/test_nodes_retrieval.py
```

Expected: 기존 테스트 전부 PASS (새 시그니처로 깨진 테스트 있으면 mock 파라미터 추가)

- [ ] **Step 6: 커밋**

```bash
git add src/spar/pipeline/graph.py src/spar/api/app.py
git commit -m "feat(pipeline): build_graph에 encoder+milvus 파라미터 추가, app 주입"
```

---

### Task 4: 전체 테스트 + 통합 확인

**Files:**
- 없음 (기존 파일 검증)

- [ ] **Step 1: 전체 테스트 실행**

```bash
python -m pytest tests/ -v
```

Expected: 전부 PASS

- [ ] **Step 2: 타입 체크**

```bash
python -m mypy src/spar/retrieval/routing.py src/spar/pipeline/nodes.py src/spar/pipeline/graph.py --ignore-missing-imports
```

Expected: 에러 없음

- [ ] **Step 3: 최종 커밋**

```bash
git add -A
git commit -m "chore: Milvus retrieval wiring 완료 — stub 제거"
```

---

## 구현 후 남는 한계

| 항목 | 상태 |
|---|---|
| `multi_hop_retrieve` 실구현 | Phase 5 — 현재 RAG 위임 유지 |
| `generate` 실구현 (LLM 호출) | 별도 Task |
| Milvus 연결 없는 환경 테스트 | mock으로 커버 완료 |
| `encoder.embed()` 비동기화 | 현재 `run_in_executor` 래핑으로 처리 |
