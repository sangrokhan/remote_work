# Hybrid Verify Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LangGraph 파이프라인에 `tool_call` + `verify` 노드를 추가해, 생성 결과가 불충분하면 미시도 retrieval 전략 + 개선된 쿼리로 최대 3회 재시도한다.

**Architecture:** 기존 파이프라인(`route → retrieve → rerank → generate`)을 보존하면서 `tool_call` 노드(retrieve 뒤)와 `verify` 노드(generate 뒤)를 추가한다. `use_verify_loop=True` 일 때만 활성화. `verify` 실패 시 결정적 전략 순서(decomposed → multi_hop → structured → rag)로 다음 미시도 전략을 선택하고 LLM으로 쿼리를 개선해 재검색한다.

**Tech Stack:** Python 3.11+, LangGraph `StateGraph`, `AsyncOpenAI` (LLMClient), `pytest`, `unittest.mock`

---

## 파일 맵

| 파일 | 역할 |
|------|------|
| `src/spar/pipeline/state.py` | 5개 필드 추가 |
| `src/spar/pipeline/config.py` | `use_verify_loop` 플래그 + `verify_loop` preset 추가 |
| `src/spar/prompts/verify.txt` | verify LLM 프롬프트 |
| `src/spar/prompts/tool_call_rewrite.txt` | 쿼리 재작성 LLM 프롬프트 |
| `src/spar/pipeline/nodes.py` | `verify`, `tool_call` 노드 메서드 추가 |
| `src/spar/pipeline/graph.py` | `_verify_selector`, verify loop 엣지 추가 |
| `tests/pipeline/test_nodes_verify_toolcall.py` | 신규 노드 단위 테스트 |
| `tests/pipeline/test_graph_verify_loop.py` | 그래프 구조 테스트 |

---

### Task 1: State 필드 추가

**Files:**
- Modify: `src/spar/pipeline/state.py`
- Test: `tests/pipeline/test_config.py` (기존 파일에 테스트 추가)

- [ ] **Step 1: 실패 테스트 작성**

`tests/pipeline/test_config.py` 에 추가:

```python
def test_sparstate_has_verify_loop_fields():
    s: SparState = {
        "query": "test",
        "retry_count": 0,
        "tried_strategies": ["rag"],
        "verify_score": 2.5,
        "verify_reason": "missing parameter details",
        "improved_query": "what is the default value of maxUE parameter",
    }
    assert s["retry_count"] == 0
    assert s["tried_strategies"] == ["rag"]
    assert s["verify_score"] == 2.5
    assert s["verify_reason"] == "missing parameter details"
    assert s["improved_query"] == "what is the default value of maxUE parameter"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/pipeline/test_config.py::test_sparstate_has_verify_loop_fields -v
```

Expected: `KeyError` 또는 타입 오류로 FAIL

- [ ] **Step 3: state.py에 필드 추가**

`src/spar/pipeline/state.py` 의 `eval_metrics` 필드 아래에 추가:

```python
    # verify loop
    retry_count: int
    tried_strategies: list[str]
    verify_score: float | None
    verify_reason: str | None
    improved_query: str | None
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/pipeline/test_config.py -v
```

Expected: 모든 테스트 PASS

- [ ] **Step 5: 커밋**

```bash
git add src/spar/pipeline/state.py tests/pipeline/test_config.py
git commit -m "feat(pipeline): add verify loop fields to SparState"
```

---

### Task 2: GraphConfig 플래그 + preset 추가

**Files:**
- Modify: `src/spar/pipeline/config.py`
- Test: `tests/pipeline/test_config.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/pipeline/test_config.py` 에 추가:

```python
def test_graphconfig_has_verify_loop_flag():
    cfg = GraphConfig(name="x")
    assert cfg.use_verify_loop is False


def test_preset_verify_loop_exists():
    cfg = next((c for c in PRESET_CONFIGS if c.name == "verify_loop"), None)
    assert cfg is not None
    assert cfg.use_query_expansion is True
    assert cfg.use_prepare_context is True
    assert cfg.use_reranker is True
    assert cfg.use_real_generate is True
    assert cfg.use_verify_loop is True
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/pipeline/test_config.py::test_graphconfig_has_verify_loop_flag tests/pipeline/test_config.py::test_preset_verify_loop_exists -v
```

Expected: FAIL (`AttributeError: 'GraphConfig' object has no attribute 'use_verify_loop'`)

- [ ] **Step 3: config.py 수정**

`src/spar/pipeline/config.py` 의 `GraphConfig` 에 필드 추가:

```python
@dataclass
class GraphConfig:
    name: str
    use_query_expansion: bool = False
    use_prepare_context: bool = False
    use_reranker: bool = False
    use_real_generate: bool = False
    use_verify_loop: bool = False
```

`PRESET_CONFIGS` 리스트 끝에 추가:

```python
    GraphConfig(
        name="verify_loop",
        use_query_expansion=True,
        use_prepare_context=True,
        use_reranker=True,
        use_real_generate=True,
        use_verify_loop=True,
    ),
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/pipeline/test_config.py -v
```

Expected: 모든 테스트 PASS

- [ ] **Step 5: 커밋**

```bash
git add src/spar/pipeline/config.py tests/pipeline/test_config.py
git commit -m "feat(pipeline): add use_verify_loop flag and verify_loop preset"
```

---

### Task 3: 프롬프트 파일 추가

**Files:**
- Create: `src/spar/prompts/verify.txt`
- Create: `src/spar/prompts/tool_call_rewrite.txt`
- Test: `tests/prompts/test___init__.py` (기존 파일에 추가)

- [ ] **Step 1: 실패 테스트 작성**

`tests/prompts/test___init__.py` 에 추가:

```python
def test_load_verify_prompt():
    from spar.prompts import load_prompt
    text = load_prompt("verify.txt")
    assert "{query}" in text
    assert "{answer}" in text
    assert "{contexts_summary}" in text


def test_load_tool_call_rewrite_prompt():
    from spar.prompts import load_prompt
    text = load_prompt("tool_call_rewrite.txt")
    assert "{query}" in text
    assert "{reason}" in text
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/prompts/test___init__.py -v
```

Expected: FAIL (`FileNotFoundError`)

- [ ] **Step 3: verify.txt 작성**

`src/spar/prompts/verify.txt`:

```
You are evaluating whether an answer sufficiently addresses a question.

Question: {query}

Answer: {answer}

Contexts used (summarized):
{contexts_summary}

Rate the answer on a scale of 1-5:
1 = completely insufficient (missing key information, wrong, or irrelevant)
2 = mostly insufficient (touches on topic but lacks critical details)
3 = adequate (answers the question but could be more complete)
4 = good (answers clearly with relevant detail)
5 = excellent (complete, accurate, well-supported by context)

Respond ONLY with valid JSON (no markdown, no explanation):
{{"score": <integer 1-5>, "reason": "<one sentence explaining the score>"}}
```

- [ ] **Step 4: tool_call_rewrite.txt 작성**

`src/spar/prompts/tool_call_rewrite.txt`:

```
You are improving a search query based on feedback about why a previous answer was insufficient.

Original query: {query}

Reason the previous answer was insufficient: {reason}

Write an improved search query that addresses the gap identified in the reason.
The improved query should be specific, focused, and likely to retrieve the missing information.

Respond with ONLY the improved query text (no explanation, no quotes).
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
pytest tests/prompts/test___init__.py -v
```

Expected: 모든 테스트 PASS

- [ ] **Step 6: 커밋**

```bash
git add src/spar/prompts/verify.txt src/spar/prompts/tool_call_rewrite.txt tests/prompts/test___init__.py
git commit -m "feat(prompts): add verify and tool_call_rewrite prompt templates"
```

---

### Task 4: `verify` 노드 구현

**Files:**
- Modify: `src/spar/pipeline/nodes.py`
- Create: `tests/pipeline/test_nodes_verify_toolcall.py`

- [ ] **Step 1: 테스트 파일 생성 및 실패 테스트 작성**

`tests/pipeline/test_nodes_verify_toolcall.py` 생성:

```python
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from spar.pipeline.nodes import Nodes
from spar.pipeline.state import SparState


def _make_nodes(llm_response: str) -> Nodes:
    router = MagicMock()
    router.route = AsyncMock(return_value=MagicMock(route=MagicMock(), needs_decomposition=False))
    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=[0.9])
    encoder = MagicMock()
    encoder.encode = MagicMock(return_value=np.zeros((2, 3)))
    milvus = MagicMock()
    milvus.hybrid_search = MagicMock(return_value=[])
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=llm_response)
    return Nodes.create(
        router=router,
        reranker=reranker,
        encoder=encoder,
        milvus=milvus,
        llm=llm,
    )


@pytest.mark.asyncio
async def test_verify_sufficient_sets_score_gte_3():
    nodes = _make_nodes(json.dumps({"score": 4, "reason": "answer is complete"}))
    state: SparState = {
        "query": "what is maxUE?",
        "answer": "maxUE controls the maximum number of UEs.",
        "reranked_chunks": [{"text": "maxUE is a parameter", "score": 0.9}],
        "retry_count": 0,
        "tried_strategies": ["rag"],
    }
    result = await nodes.verify(state)
    assert result["verify_score"] == 4
    assert result["verify_reason"] == "answer is complete"
    assert "verify" in result["node_trace"]


@pytest.mark.asyncio
async def test_verify_insufficient_sets_score_lt_3():
    nodes = _make_nodes(json.dumps({"score": 1, "reason": "missing parameter range"}))
    state: SparState = {
        "query": "what is the range of maxUE?",
        "answer": "I don't have enough information.",
        "reranked_chunks": [],
        "retry_count": 0,
        "tried_strategies": ["rag"],
    }
    result = await nodes.verify(state)
    assert result["verify_score"] == 1
    assert result["verify_reason"] == "missing parameter range"


@pytest.mark.asyncio
async def test_verify_malformed_json_defaults_sufficient():
    nodes = _make_nodes("not valid json at all")
    state: SparState = {
        "query": "test",
        "answer": "answer",
        "reranked_chunks": [],
        "retry_count": 0,
        "tried_strategies": [],
    }
    result = await nodes.verify(state)
    assert result["verify_score"] == 5.0
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/pipeline/test_nodes_verify_toolcall.py::test_verify_sufficient_sets_score_gte_3 -v
```

Expected: FAIL (`AttributeError: 'Nodes' object has no attribute 'verify'`)

- [ ] **Step 3: `verify` 노드 메서드 구현**

`src/spar/pipeline/nodes.py` 의 `generate` 메서드 아래에 추가:

```python
    async def verify(self, state: SparState) -> SparState:
        t0 = time.monotonic()
        if self.llm is None:
            elapsed = (time.monotonic() - t0) * 1000
            return {
                **state,
                "verify_score": 5.0,
                "verify_reason": "no llm — skipping verify",
                "node_trace": _append_trace(state, "verify"),
                "node_timings": _record_timing(state, "verify", elapsed),
            }

        query = state.get("improved_query") or state.get("rewritten_query") or state.get("expanded_query") or state["query"]
        answer = state.get("answer", "")
        chunks = state.get("reranked_chunks") or state.get("raw_chunks", [])
        contexts_summary = "\n---\n".join(c["text"][:300] for c in chunks[:5])

        prompt = load_prompt("verify.txt").format(
            query=query,
            answer=answer,
            contexts_summary=contexts_summary or "(no context)",
        )
        raw = await self.llm.chat([{"role": "user", "content": prompt}], max_tokens=128)

        try:
            parsed = json.loads(raw.strip())
            score = float(parsed["score"])
            reason = str(parsed.get("reason", ""))
        except Exception:
            score = 5.0
            reason = "parse error — treating as sufficient"

        elapsed = (time.monotonic() - t0) * 1000
        return {
            **state,
            "verify_score": score,
            "verify_reason": reason,
            "node_trace": _append_trace(state, "verify"),
            "node_timings": _record_timing(state, "verify", elapsed),
        }
```

`nodes.py` 상단 import에 추가:

```python
import json

from spar.prompts import load_prompt
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/pipeline/test_nodes_verify_toolcall.py::test_verify_sufficient_sets_score_gte_3 tests/pipeline/test_nodes_verify_toolcall.py::test_verify_insufficient_sets_score_lt_3 tests/pipeline/test_nodes_verify_toolcall.py::test_verify_malformed_json_defaults_sufficient -v
```

Expected: 3개 모두 PASS

- [ ] **Step 5: 커밋**

```bash
git add src/spar/pipeline/nodes.py tests/pipeline/test_nodes_verify_toolcall.py
git commit -m "feat(pipeline): add verify node with LLM self-eval"
```

---

### Task 5: `tool_call` 노드 구현

**Files:**
- Modify: `src/spar/pipeline/nodes.py`
- Modify: `tests/pipeline/test_nodes_verify_toolcall.py`

전략 fallback 순서: `["decomposed", "multi_hop", "structured", "rag"]`

`tool_call` 진입 시 `tried_strategies`에 없는 첫 번째 전략을 선택. 선택된 전략에 따라 `decomposed_retrieve` / `multi_hop_retrieve` / `structured_retrieve` / `rag_retrieve` 메서드를 직접 호출 (router 우회).

- [ ] **Step 1: 실패 테스트 추가**

`tests/pipeline/test_nodes_verify_toolcall.py` 에 추가:

```python
@pytest.mark.asyncio
async def test_tool_call_picks_next_untried_strategy():
    nodes = _make_nodes("what is maxUE default value")
    state: SparState = {
        "query": "what is maxUE?",
        "tried_strategies": ["rag"],
        "retry_count": 0,
        "raw_chunks": [],
        "verify_reason": "missing default value info",
        "route_result": MagicMock(needs_decomposition=False, route=MagicMock()),
        "top_k": 5,
        "matched_terms": [],
    }
    result = await nodes.tool_call(state)
    assert "decomposed" in result["tried_strategies"]
    assert result["retry_count"] == 1
    assert result["improved_query"] == "what is maxUE default value"
    assert "tool_call" in result["node_trace"]


@pytest.mark.asyncio
async def test_tool_call_skips_already_tried():
    nodes = _make_nodes("retry query")
    state: SparState = {
        "query": "test",
        "tried_strategies": ["rag", "decomposed", "multi_hop"],
        "retry_count": 2,
        "raw_chunks": [],
        "verify_reason": "still missing info",
        "route_result": MagicMock(needs_decomposition=False, route=MagicMock()),
        "top_k": 5,
        "matched_terms": [],
    }
    result = await nodes.tool_call(state)
    assert "structured" in result["tried_strategies"]
    assert result["retry_count"] == 3


@pytest.mark.asyncio
async def test_tool_call_no_untried_strategy_returns_state_unchanged():
    nodes = _make_nodes("query")
    state: SparState = {
        "query": "test",
        "tried_strategies": ["rag", "decomposed", "multi_hop", "structured"],
        "retry_count": 3,
        "raw_chunks": [],
        "verify_reason": "exhausted",
        "route_result": MagicMock(needs_decomposition=False, route=MagicMock()),
        "top_k": 5,
        "matched_terms": [],
    }
    result = await nodes.tool_call(state)
    assert result["retry_count"] == 3
    assert "tool_call" in result["node_trace"]
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/pipeline/test_nodes_verify_toolcall.py::test_tool_call_picks_next_untried_strategy -v
```

Expected: FAIL (`AttributeError: 'Nodes' object has no attribute 'tool_call'`)

- [ ] **Step 3: `tool_call` 노드 메서드 구현**

`src/spar/pipeline/nodes.py` 상단(`_ACRONYMS_PATH` 상수 근처)에 추가:

```python
_FALLBACK_ORDER = ["decomposed", "multi_hop", "structured", "rag"]
```

그 다음 `verify` 메서드 아래에 추가:

```python
    async def tool_call(self, state: SparState) -> SparState:
        t0 = time.monotonic()
        tried = list(state.get("tried_strategies") or [])
        retry_count = state.get("retry_count", 0)

        next_strategy = next(
            (s for s in _FALLBACK_ORDER if s not in tried),
            None,
        )

        if next_strategy is None:
            elapsed = (time.monotonic() - t0) * 1000
            return {
                **state,
                "node_trace": _append_trace(state, "tool_call"),
                "node_timings": _record_timing(state, "tool_call", elapsed),
            }

        # rewrite query using verify_reason
        original_query = state.get("rewritten_query") or state.get("expanded_query") or state["query"]
        verify_reason = state.get("verify_reason") or ""

        if self.llm is not None and verify_reason:
            rewrite_prompt = load_prompt("tool_call_rewrite.txt").format(
                query=original_query,
                reason=verify_reason,
            )
            improved_query = await self.llm.chat(
                [{"role": "user", "content": rewrite_prompt}],
                max_tokens=128,
            )
            improved_query = improved_query.strip()
        else:
            improved_query = original_query

        # execute chosen strategy with improved query (router bypass)
        sub_state: SparState = {
            **state,
            "query": improved_query,
            "improved_query": improved_query,
        }

        strategy_map = {
            "decomposed": self.decomposed_retrieve,
            "multi_hop": self.multi_hop_retrieve,
            "structured": self.structured_retrieve,
            "rag": self.rag_retrieve,
        }
        retrieve_fn = strategy_map[next_strategy]
        retrieved = await retrieve_fn(sub_state)

        # merge chunks (dedup by id or text prefix)
        existing = state.get("raw_chunks") or []
        new_chunks = retrieved.get("raw_chunks") or []
        seen: set[str] = {c.get("id") or c.get("text", "")[:120] for c in existing}
        merged = list(existing)
        for chunk in new_chunks:
            key = chunk.get("id") or chunk.get("text", "")[:120]
            if key not in seen:
                seen.add(key)
                merged.append(chunk)

        elapsed = (time.monotonic() - t0) * 1000
        return {
            **state,
            "raw_chunks": merged,
            "improved_query": improved_query,
            "tried_strategies": [*tried, next_strategy],
            "retry_count": retry_count + 1,
            "node_trace": _append_trace(state, "tool_call"),
            "node_timings": _record_timing(state, "tool_call", elapsed),
        }
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/pipeline/test_nodes_verify_toolcall.py -v
```

Expected: 모든 테스트 PASS

- [ ] **Step 5: 커밋**

```bash
git add src/spar/pipeline/nodes.py tests/pipeline/test_nodes_verify_toolcall.py
git commit -m "feat(pipeline): add tool_call node with deterministic strategy fallback"
```

---

### Task 6: 그래프 연결 (`graph.py`)

**Files:**
- Modify: `src/spar/pipeline/graph.py`
- Create: `tests/pipeline/test_graph_verify_loop.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/pipeline/test_graph_verify_loop.py` 생성:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from spar.pipeline.config import GraphConfig, PRESET_CONFIGS
from spar.pipeline.graph import build_graph


def _deps(llm=None):
    router = MagicMock()
    router.route = AsyncMock(return_value=MagicMock(route=MagicMock(), needs_decomposition=False))
    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=[])
    encoder = MagicMock()
    encoder.encode = MagicMock(return_value=np.zeros((1, 3)))
    milvus = MagicMock()
    milvus.hybrid_search = MagicMock(return_value=[])
    return router, reranker, encoder, milvus


def test_verify_loop_config_has_both_nodes():
    router, reranker, encoder, milvus = _deps()
    cfg = GraphConfig(
        name="test_verify",
        use_reranker=True,
        use_real_generate=True,
        use_verify_loop=True,
    )
    graph = build_graph(router, reranker, encoder, milvus, config=cfg)
    node_names = set(graph.get_graph().nodes.keys())
    assert "tool_call" in node_names
    assert "verify" in node_names


def test_verify_loop_false_has_no_verify_nodes():
    router, reranker, encoder, milvus = _deps()
    cfg = GraphConfig(name="no_verify", use_reranker=True)
    graph = build_graph(router, reranker, encoder, milvus, config=cfg)
    node_names = set(graph.get_graph().nodes.keys())
    assert "tool_call" not in node_names
    assert "verify" not in node_names


def test_preset_verify_loop_has_correct_nodes():
    router, reranker, encoder, milvus = _deps()
    cfg = next(c for c in PRESET_CONFIGS if c.name == "verify_loop")
    graph = build_graph(router, reranker, encoder, milvus, config=cfg)
    node_names = set(graph.get_graph().nodes.keys())
    assert "tool_call" in node_names
    assert "verify" in node_names
    assert "rerank" in node_names
    assert "preprocess" in node_names


def test_default_graph_unchanged_no_verify_nodes():
    """기존 동작 보존 — default config(full_retrieval)에는 verify 노드 없음."""
    router, reranker, encoder, milvus = _deps()
    graph = build_graph(router, reranker, encoder, milvus)
    node_names = set(graph.get_graph().nodes.keys())
    assert "tool_call" not in node_names
    assert "verify" not in node_names
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/pipeline/test_graph_verify_loop.py -v
```

Expected: FAIL (`AssertionError: 'tool_call' not in node_names`)

- [ ] **Step 3: `_verify_selector` 함수 추가 및 `build_graph` 수정**

`src/spar/pipeline/graph.py` 에서 `_route_selector` 아래에 추가:

```python
def _verify_selector(state: SparState) -> str:
    score = state.get("verify_score", 5.0)
    retry_count = state.get("retry_count", 0)
    tried = set(state.get("tried_strategies") or [])
    all_strategies = {"rag", "decomposed", "multi_hop", "structured"}
    remaining = all_strategies - tried
    if score < 3 and retry_count < 3 and remaining:
        return "tool_call"
    return END
```

`build_graph` 함수에서 `g.add_node("generate", ...)` 이후 블록을 다음으로 교체:

```python
    g.add_node("generate", nodes.generate)

    if cfg.use_verify_loop:
        # verify loop: all_retrieve → tool_call → rerank/generate → verify ⟲
        g.add_node("tool_call", nodes.tool_call)
        g.add_node("verify", nodes.verify)

        for name in _all_retrieve:
            g.add_edge(name, "tool_call")

        if cfg.use_reranker:
            g.add_node("rerank", nodes.rerank)
            g.add_edge("tool_call", "rerank")
            g.add_edge("rerank", "generate")
        else:
            g.add_edge("tool_call", "generate")

        g.add_edge("generate", "verify")
        g.add_conditional_edges(
            "verify",
            _verify_selector,
            {"tool_call": "tool_call", END: END},
        )
    else:
        if cfg.use_reranker:
            g.add_node("rerank", nodes.rerank)
            for name in _all_retrieve:
                g.add_edge(name, "rerank")
            g.add_edge("rerank", "generate")
        else:
            for name in _all_retrieve:
                g.add_edge(name, "generate")

        g.add_edge("generate", END)
```

> **주의:** 기존 `build_graph` 의 reranker 블록과 `g.add_edge("generate", END)` 를 위 코드로 완전 교체. 나머지 부분은 그대로 유지.

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/pipeline/test_graph_verify_loop.py -v
```

Expected: 모든 테스트 PASS

- [ ] **Step 5: 기존 그래프 테스트 회귀 확인**

```bash
pytest tests/pipeline/ -v
```

Expected: 모든 기존 테스트 PASS (특히 `test_graph_config.py`)

- [ ] **Step 6: 커밋**

```bash
git add src/spar/pipeline/graph.py tests/pipeline/test_graph_verify_loop.py
git commit -m "feat(pipeline): wire tool_call and verify nodes into LangGraph"
```

---

### Task 7: 전체 테스트 + 마무리

- [ ] **Step 1: 전체 테스트 실행**

```bash
pytest tests/ -v --tb=short
```

Expected: 모든 테스트 PASS (실패 있으면 해결 후 진행)

- [ ] **Step 2: 커밋 (필요 시)**

실패 수정 후:

```bash
git add -p
git commit -m "fix(pipeline): resolve test failures from verify loop integration"
```

- [ ] **Step 3: prd.md 업데이트**

`docs/prd.md` 에서 관련 Task 체크박스 갱신. verify loop이 Phase 2 Task로 등록되어 있지 않다면 신규 항목 추가.

- [ ] **Step 4: 최종 커밋**

```bash
git add docs/prd.md
git commit -m "docs: mark hybrid verify loop task complete in prd"
```
