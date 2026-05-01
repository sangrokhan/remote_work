# HyDE + Multi-Query Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two opt-in retrieval expansion strategies — HyDE (hypothetical document embedding) and Multi-Query (alternative phrasing search) — that improve recall for definition and complex queries without changing the existing pipeline graph structure.

**Architecture:** HyDE generates a hypothetical answer passage and uses its embedding as the dense search vector; Multi-Query generates 3 alternative phrasings and merges their results via deduplication. Both are controlled by `HYDE_ENABLED` and `MULTI_QUERY_ENABLED` env vars, read at call time so they can be toggled per-test with `patch.dict(os.environ)`. Integration point is `rag_retrieve()` in `nodes.py` — no new graph nodes needed.

**Tech Stack:** Python 3.11, LangGraph, sentence-transformers (encoder singleton), existing `get_client(LLMRole.ROUTER)` LLM pattern, `pytest` + `unittest.mock`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/spar/retrieval/hyde.py` | `HyDEExpander.expand(query) → str` |
| Create | `src/spar/retrieval/multi_query.py` | `MultiQueryRewriter.rewrite(query) → list[str]` |
| Create | `src/spar/prompts/hyde_system.txt` | System prompt for hypothetical doc generation |
| Create | `src/spar/prompts/multi_query_system.txt` | System prompt for alternative phrasing generation |
| Create | `tests/retrieval/test_hyde.py` | Unit tests for HyDEExpander |
| Create | `tests/retrieval/test_multi_query.py` | Unit tests for MultiQueryRewriter |
| Modify | `src/spar/pipeline/state.py` | Add `hyde_doc: str \| None`, `alternative_queries: list[str]` |
| Modify | `src/spar/pipeline/nodes.py` | Add expanders to `Nodes`, modify `rag_retrieve` |

---

## Task 1: HyDE Prompt + Expander

**Files:**
- Create: `src/spar/prompts/hyde_system.txt`
- Create: `src/spar/retrieval/hyde.py`
- Create: `tests/retrieval/test_hyde.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_hyde.py
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from spar.retrieval.hyde import HyDEExpander


@pytest.fixture(autouse=True)
def clear_registry():
    from spar.llm import registry
    registry._clients.clear()
    yield
    registry._clients.clear()


@pytest.mark.asyncio
async def test_expand_returns_llm_response():
    expander = HyDEExpander()
    with patch("spar.retrieval.hyde.get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.chat.return_value = "The handover threshold TTT defines the time-to-trigger value."
        mock_get.return_value = mock_client

        result = await expander.expand("What is TTT in LTE handover?")

    assert result == "The handover threshold TTT defines the time-to-trigger value."
    mock_client.chat.assert_awaited_once()
    call_args = mock_client.chat.call_args
    messages = call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "TTT" in messages[1]["content"]


@pytest.mark.asyncio
async def test_expand_falls_back_to_query_on_llm_error():
    expander = HyDEExpander()
    with patch("spar.retrieval.hyde.get_client") as mock_get:
        mock_get.side_effect = RuntimeError("LLM unavailable")

        result = await expander.expand("What is TTT in LTE handover?")

    assert result == "What is TTT in LTE handover?"


@pytest.mark.asyncio
async def test_expand_falls_back_to_query_on_empty_response():
    expander = HyDEExpander()
    with patch("spar.retrieval.hyde.get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.chat.return_value = "   "
        mock_get.return_value = mock_client

        result = await expander.expand("What is TTT?")

    assert result == "What is TTT?"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/han/.openclaw/workspace/remote_work/spar
pytest tests/retrieval/test_hyde.py -v
```

Expected: `ModuleNotFoundError: No module named 'spar.retrieval.hyde'`

- [ ] **Step 3: Create the system prompt**

```
# src/spar/prompts/hyde_system.txt
You are a technical documentation assistant for Samsung RAN systems (LTE/NR).
Given a user query, write a concise hypothetical passage (3–5 sentences) that would appear in a Samsung RAN technical reference document and directly answer the query.
Use precise RAN terminology. Write only the passage text — no preamble, no heading, no explanation.
```

- [ ] **Step 4: Implement HyDEExpander**

```python
# src/spar/retrieval/hyde.py
from __future__ import annotations

import logging

from spar.llm import LLMRole, get_client
from spar.prompts import load_prompt

_log = logging.getLogger(__name__)
_SYSTEM_PROMPT = load_prompt("hyde_system.txt")


class HyDEExpander:
    async def expand(self, query: str) -> str:
        try:
            client = await get_client(LLMRole.ROUTER)
            doc = await client.chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                max_tokens=200,
            )
            return doc.strip() or query
        except Exception as exc:
            _log.warning("HyDEExpander fallback — %s: %s", type(exc).__name__, exc)
            return query
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/retrieval/test_hyde.py -v
```

Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/spar/prompts/hyde_system.txt src/spar/retrieval/hyde.py tests/retrieval/test_hyde.py
git commit -m "feat(retrieval): add HyDEExpander for hypothetical document embedding (Task 2.6)"
```

---

## Task 2: Multi-Query Prompt + Rewriter

**Files:**
- Create: `src/spar/prompts/multi_query_system.txt`
- Create: `src/spar/retrieval/multi_query.py`
- Create: `tests/retrieval/test_multi_query.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_multi_query.py
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, patch

from spar.retrieval.multi_query import MultiQueryRewriter


@pytest.fixture(autouse=True)
def clear_registry():
    from spar.llm import registry
    registry._clients.clear()
    yield
    registry._clients.clear()


@pytest.mark.asyncio
async def test_rewrite_returns_three_alternatives():
    rewriter = MultiQueryRewriter()
    alternatives = [
        "What is the time-to-trigger parameter for LTE handover?",
        "How does TTT affect handover decisions in LTE?",
        "Define TTT handover threshold in Samsung LTE configuration.",
    ]
    with patch("spar.retrieval.multi_query.get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.chat.return_value = json.dumps(alternatives)
        mock_get.return_value = mock_client

        result = await rewriter.rewrite("What is TTT in LTE handover?")

    assert result == alternatives
    mock_client.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_rewrite_caps_at_three_alternatives():
    rewriter = MultiQueryRewriter()
    four_alts = ["q1", "q2", "q3", "q4"]
    with patch("spar.retrieval.multi_query.get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.chat.return_value = json.dumps(four_alts)
        mock_get.return_value = mock_client

        result = await rewriter.rewrite("some query")

    assert result == ["q1", "q2", "q3"]


@pytest.mark.asyncio
async def test_rewrite_returns_empty_on_llm_error():
    rewriter = MultiQueryRewriter()
    with patch("spar.retrieval.multi_query.get_client") as mock_get:
        mock_get.side_effect = RuntimeError("LLM unavailable")

        result = await rewriter.rewrite("What is TTT?")

    assert result == []


@pytest.mark.asyncio
async def test_rewrite_returns_empty_on_invalid_json():
    rewriter = MultiQueryRewriter()
    with patch("spar.retrieval.multi_query.get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.chat.return_value = "not json"
        mock_get.return_value = mock_client

        result = await rewriter.rewrite("What is TTT?")

    assert result == []


@pytest.mark.asyncio
async def test_rewrite_filters_blank_strings():
    rewriter = MultiQueryRewriter()
    with patch("spar.retrieval.multi_query.get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.chat.return_value = json.dumps(["q1", "  ", "q3"])
        mock_get.return_value = mock_client

        result = await rewriter.rewrite("What is TTT?")

    assert result == ["q1", "q3"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/retrieval/test_multi_query.py -v
```

Expected: `ModuleNotFoundError: No module named 'spar.retrieval.multi_query'`

- [ ] **Step 3: Create the system prompt**

```
# src/spar/prompts/multi_query_system.txt
You are a query expansion assistant for a Samsung RAN technical document retrieval system.
Given a user query, generate exactly 3 alternative phrasings that preserve the original intent but use different vocabulary, technical synonyms, or sentence structure. These alternatives improve retrieval coverage across different document styles.
Return a JSON array of exactly 3 strings. No explanation, no extra text.
Example output: ["alternative phrasing 1", "alternative phrasing 2", "alternative phrasing 3"]
```

- [ ] **Step 4: Implement MultiQueryRewriter**

```python
# src/spar/retrieval/multi_query.py
from __future__ import annotations

import json
import logging

from spar.llm import LLMRole, get_client
from spar.prompts import load_prompt

_log = logging.getLogger(__name__)
_SYSTEM_PROMPT = load_prompt("multi_query_system.txt")
_MAX_ALTERNATIVES = 3


class MultiQueryRewriter:
    async def rewrite(self, query: str) -> list[str]:
        try:
            client = await get_client(LLMRole.ROUTER)
            raw = await client.chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                max_tokens=256,
            )
            alternatives: list[str] = json.loads(raw)
            if not isinstance(alternatives, list) or not alternatives:
                return []
            return [
                q for q in alternatives[:_MAX_ALTERNATIVES]
                if isinstance(q, str) and q.strip()
            ]
        except Exception as exc:
            _log.warning("MultiQueryRewriter fallback — %s: %s", type(exc).__name__, exc)
            return []
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/retrieval/test_multi_query.py -v
```

Expected: 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/spar/prompts/multi_query_system.txt src/spar/retrieval/multi_query.py tests/retrieval/test_multi_query.py
git commit -m "feat(retrieval): add MultiQueryRewriter for alternative phrasing search (Task 2.6)"
```

---

## Task 3: Pipeline Integration — State + Nodes

**Files:**
- Modify: `src/spar/pipeline/state.py` (lines 29–30 — after `reranked_chunks`)
- Modify: `src/spar/pipeline/nodes.py` (imports, `Nodes` dataclass, `rag_retrieve`)
- Create: `tests/pipeline/test_rag_retrieve_expansion.py`

### 3A: Extend SparState

- [ ] **Step 1: Add expansion fields to SparState**

In `src/spar/pipeline/state.py`, after `reranked_chunks: list[dict[str, Any]]`, add:

```python
    # retrieval expansion (Task 2.6)
    hyde_doc: str | None
    alternative_queries: list[str]
```

Full file after edit:

```python
from __future__ import annotations

from typing import Any, TypedDict

from spar.router.schemas import RouteResult


class SparState(TypedDict, total=False):
    # input
    query: str
    product: str | None
    release: str | None
    top_k: int
    request_id: str
    history: list[dict[str, str]]

    # preprocess
    expanded_query: str
    history_context: str

    # routing
    route_result: RouteResult

    # decomposition
    sub_questions: list[str]

    # retrieval
    raw_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]

    # retrieval expansion (Task 2.6)
    hyde_doc: str | None
    alternative_queries: list[str]

    # generation
    answer: str

    # observability
    error: str | None
    node_trace: list[str]
```

- [ ] **Step 2: Run existing tests to confirm no regression**

```bash
pytest tests/ -v --tb=short -q
```

Expected: all existing tests PASS (SparState is a TypedDict — adding fields is non-breaking)

### 3B: Wire Expanders into Nodes + rag_retrieve

- [ ] **Step 3: Write the failing integration tests**

```python
# tests/pipeline/test_rag_retrieve_expansion.py
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spar.pipeline.nodes import Nodes
from spar.pipeline.state import SparState
from spar.router.schemas import Route, RouteResult


def _make_nodes() -> Nodes:
    router = MagicMock()
    reranker = MagicMock()
    encoder = MagicMock()
    encoder.encode.return_value = [[0.1] * 768]
    milvus = MagicMock()
    milvus.hybrid_search.return_value = [
        {"id": "c1", "text": "chunk one", "score": 0.9},
        {"id": "c2", "text": "chunk two", "score": 0.8},
    ]
    return Nodes.create(router=router, reranker=reranker, encoder=encoder, milvus=milvus)


def _make_state(**kwargs) -> SparState:
    return SparState(
        query="What is TTT?",
        expanded_query="What is TTT(Time-To-Trigger)?",
        top_k=5,
        route_result=RouteResult(route=Route.DEFAULT_RAG, confidence=0.8, needs_decomposition=False),
        node_trace=[],
        **kwargs,
    )


@pytest.mark.asyncio
async def test_rag_retrieve_baseline_no_expansion():
    nodes = _make_nodes()
    state = _make_state()

    with patch.dict("os.environ", {"HYDE_ENABLED": "false", "MULTI_QUERY_ENABLED": "false"}):
        result = await nodes.rag_retrieve(state)

    assert "raw_chunks" in result
    assert len(result["raw_chunks"]) > 0
    assert "hyde_doc" not in result
    assert "alternative_queries" not in result


@pytest.mark.asyncio
async def test_rag_retrieve_hyde_enabled_stores_hyde_doc():
    nodes = _make_nodes()
    state = _make_state()

    with patch("spar.pipeline.nodes.HyDEExpander") as MockExpander, \
         patch.dict("os.environ", {"HYDE_ENABLED": "true", "MULTI_QUERY_ENABLED": "false"}):
        mock_instance = AsyncMock()
        mock_instance.expand.return_value = "TTT is the time-to-trigger threshold used in handover."
        MockExpander.return_value = mock_instance
        # Re-create nodes so __post_init__ picks up the mock
        nodes2 = _make_nodes()
        nodes2._hyde_expander = mock_instance

        result = await nodes2.rag_retrieve(state)

    assert result.get("hyde_doc") == "TTT is the time-to-trigger threshold used in handover."
    assert "raw_chunks" in result


@pytest.mark.asyncio
async def test_rag_retrieve_multi_query_merges_results():
    nodes = _make_nodes()
    # Make milvus return different chunks per call to simulate multiple queries
    call_count = 0
    def _search_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        return [{"id": f"c{call_count}", "text": f"chunk {call_count}", "score": 0.9 - call_count * 0.1}]
    nodes.milvus.hybrid_search.side_effect = _search_side_effect

    state = _make_state()

    with patch.dict("os.environ", {"HYDE_ENABLED": "false", "MULTI_QUERY_ENABLED": "true"}):
        nodes._multi_query_rewriter = AsyncMock()
        nodes._multi_query_rewriter.rewrite.return_value = ["alt q1", "alt q2"]

        result = await nodes.rag_retrieve(state)

    assert result.get("alternative_queries") == ["alt q1", "alt q2"]
    # Results from original + 2 alternatives should be merged (deduplicated)
    assert len(result["raw_chunks"]) >= 1


@pytest.mark.asyncio
async def test_rag_retrieve_multi_query_deduplicates():
    nodes = _make_nodes()
    # All searches return same chunk id
    nodes.milvus.hybrid_search.return_value = [{"id": "c1", "text": "same chunk", "score": 0.9}]

    state = _make_state()

    with patch.dict("os.environ", {"HYDE_ENABLED": "false", "MULTI_QUERY_ENABLED": "true"}):
        nodes._multi_query_rewriter = AsyncMock()
        nodes._multi_query_rewriter.rewrite.return_value = ["alt q1", "alt q2"]

        result = await nodes.rag_retrieve(state)

    # Deduplication: same chunk appears only once
    chunk_ids = [c["id"] for c in result["raw_chunks"]]
    assert len(chunk_ids) == len(set(chunk_ids))
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
pytest tests/pipeline/test_rag_retrieve_expansion.py -v
```

Expected: `ImportError` or `AttributeError` — `_hyde_expander` / `_multi_query_rewriter` not yet in Nodes

- [ ] **Step 5: Modify Nodes dataclass in `src/spar/pipeline/nodes.py`**

Add imports at the top (after existing imports):

```python
import os

from spar.retrieval.hyde import HyDEExpander
from spar.retrieval.multi_query import MultiQueryRewriter
```

Add two new optional fields to the `Nodes` dataclass (after `_decomposer`):

```python
    _hyde_expander: HyDEExpander = None  # type: ignore[assignment]
    _multi_query_rewriter: MultiQueryRewriter = None  # type: ignore[assignment]
```

Extend `__post_init__` to initialise them:

```python
    def __post_init__(self) -> None:
        if self._decomposer is None:
            self._decomposer = QueryDecomposer()
        if self._hyde_expander is None:
            self._hyde_expander = HyDEExpander()
        if self._multi_query_rewriter is None:
            self._multi_query_rewriter = MultiQueryRewriter()
```

Replace the entire `rag_retrieve` method:

```python
    async def rag_retrieve(self, state: SparState) -> SparState:
        query = state.get("expanded_query") or state["query"]
        route_result = state["route_result"]
        top_k = state.get("top_k", 10)
        doc_types = doc_types_for_route(route_result)
        expr = build_expr(route_result)

        # Dense vector: HyDE replaces query embedding when enabled
        if os.getenv("HYDE_ENABLED", "false").lower() == "true":
            hyde_doc = await self._hyde_expander.expand(query)
            query_vector: list[float] = self.encoder.encode([hyde_doc])[0].tolist()
        else:
            hyde_doc = None
            query_vector = self.encoder.encode([query])[0].tolist()

        # Alternative phrasings when multi-query is enabled
        if os.getenv("MULTI_QUERY_ENABLED", "false").lower() == "true":
            alternatives = await self._multi_query_rewriter.rewrite(query)
        else:
            alternatives = []

        # Search: original query + alternatives, dedup by chunk id
        seen: set[str] = set()
        merged: list[dict[str, Any]] = []

        for chunk in await self._hybrid_search_multi(doc_types, query, query_vector, top_k, expr):
            key = chunk.get("id") or chunk.get("text", "")[:120]
            if key not in seen:
                seen.add(key)
                merged.append(chunk)

        if alternatives:
            async def _retrieve_alt(alt: str) -> list[dict[str, Any]]:
                vec: list[float] = self.encoder.encode([alt])[0].tolist()
                return await self._hybrid_search_multi(doc_types, alt, vec, top_k, expr)

            for alt_chunks in await asyncio.gather(*[_retrieve_alt(a) for a in alternatives]):
                for chunk in alt_chunks:
                    key = chunk.get("id") or chunk.get("text", "")[:120]
                    if key not in seen:
                        seen.add(key)
                        merged.append(chunk)

        merged.sort(key=lambda c: c["score"], reverse=True)

        update: dict = {
            "raw_chunks": merged[: top_k * 2],
            "node_trace": _append_trace(state, "rag_retrieve"),
        }
        if hyde_doc and hyde_doc != query:
            update["hyde_doc"] = hyde_doc
        if alternatives:
            update["alternative_queries"] = alternatives
        return {**state, **update}
```

- [ ] **Step 6: Run integration tests**

```bash
pytest tests/pipeline/test_rag_retrieve_expansion.py -v
```

Expected: 4 tests PASS

- [ ] **Step 7: Run full test suite to confirm no regression**

```bash
pytest tests/ -v --tb=short -q
```

Expected: all tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/spar/pipeline/state.py src/spar/pipeline/nodes.py tests/pipeline/test_rag_retrieve_expansion.py
git commit -m "feat(pipeline): integrate HyDE + multi-query expansion into rag_retrieve (Task 2.6)"
```

---

## Verification

After all tasks complete:

- [ ] `pytest tests/retrieval/test_hyde.py tests/retrieval/test_multi_query.py tests/pipeline/test_rag_retrieve_expansion.py -v` — all green
- [ ] `pytest tests/ -q` — no regressions
- [ ] Manual smoke: `HYDE_ENABLED=true python -c "import asyncio; from spar.retrieval.hyde import HyDEExpander; print(asyncio.run(HyDEExpander().expand('What is TTT?')))"`
  - Without a real LLM: returns the query string (fallback path)
- [ ] Manual smoke: `MULTI_QUERY_ENABLED=true python -c "import asyncio; from spar.retrieval.multi_query import MultiQueryRewriter; print(asyncio.run(MultiQueryRewriter().rewrite('What is TTT?')))"`
  - Without a real LLM: returns `[]` (fallback path)

---

## Doc Update (post-implementation)

After merging, update:
- `docs/prd.md` Task 2.6 checkboxes → all checked, add completion date + merge commit
- `README.md` + `AGENTS.md` — add `hyde.py`, `multi_query.py` to directory map
