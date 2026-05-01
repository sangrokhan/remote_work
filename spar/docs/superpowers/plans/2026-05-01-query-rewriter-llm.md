# LLM Query Rewriting & Complexity Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a LangGraph `rewrite_query` node that rewrites ambiguous queries into self-contained form and classifies them as `simple` or `complex` using a small LLM (one call, JSON response).

**Architecture:** A new `rewrite_query()` async function in `query_rewriter.py` calls `get_client(LLMRole.ROUTER)` exactly like `QueryDecomposer`. Returns a `QueryRewriteResult` dataclass. A new `rewrite_query` LangGraph node inserts between `preprocess` and `prepare_context`; removing it from the graph bypasses rewriting entirely without breaking anything else.

**Tech Stack:** Python 3.12, LangGraph, `spar.llm.get_client(LLMRole.ROUTER)`, `spar.prompts.load_prompt`, pytest-asyncio

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/spar/retrieval/query_rewriter.py` | Modify | Add `QueryRewriteResult` dataclass + `rewrite_query()` async function |
| `src/spar/prompts/query_rewrite_system.txt` | Create | System prompt: rewrite + classify JSON output |
| `src/spar/pipeline/state.py` | Modify | Add `rewritten_query`, `query_complexity` fields |
| `src/spar/pipeline/nodes.py` | Modify | Add `rewrite_query` node method to `Nodes` class |
| `src/spar/pipeline/graph.py` | Modify | Insert `rewrite_query` node between `preprocess` and `prepare_context` |
| `tests/unit/retrieval/test_query_rewriter.py` | Create | Unit tests for `rewrite_query()` with mock LLM |

---

## Task 1: System Prompt

**Files:**
- Create: `src/spar/prompts/query_rewrite_system.txt`

- [ ] **Step 1: Create prompt file**

```
You are a query rewriting assistant for a 3GPP/RAN technical knowledge base.

Given a user query and optional conversation history, do two things:
1. Rewrite the query to be self-contained (resolve co-references like "그것", "해당 파라미터", "위에서", "거기서" to their actual referents from history).
2. Classify the query as "simple" or "complex":
   - simple: single concept lookup, one-hop factual question, glossary/definition request
   - complex: multi-hop reasoning, comparison across specifications, troubleshooting sequences, or questions requiring integration of multiple sources

Return ONLY valid JSON with no markdown, no explanation:
{"rewritten": "<rewritten query in same language as original>", "complexity": "simple|complex", "rationale": "<one sentence why>"}

If there is no history or no co-references to resolve, "rewritten" may equal the original query.
```

- [ ] **Step 2: Verify file exists**

```bash
cat src/spar/prompts/query_rewrite_system.txt
```

Expected: file content printed with no error.

- [ ] **Step 3: Commit**

```bash
git add src/spar/prompts/query_rewrite_system.txt
git commit -m "feat(prompts): add query rewrite system prompt"
```

---

## Task 2: `QueryRewriteResult` dataclass + `rewrite_query()` function

**Files:**
- Modify: `src/spar/retrieval/query_rewriter.py`
- Create: `tests/unit/retrieval/test_query_rewriter.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/retrieval/test_query_rewriter.py`:

```python
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from spar.retrieval.query_rewriter import QueryRewriteResult, rewrite_query


def _make_client(raw_response: str) -> Any:
    mock = AsyncMock()
    mock.chat = AsyncMock(return_value=raw_response)
    return mock


@pytest.mark.asyncio
async def test_rewrite_query_basic():
    payload = {"rewritten": "What is the handover procedure?", "complexity": "simple", "rationale": "single concept"}
    with patch("spar.retrieval.query_rewriter.get_client", return_value=_make_client(json.dumps(payload))):
        result = await rewrite_query("What is it?", history=[], acronyms={})
    assert isinstance(result, QueryRewriteResult)
    assert result.rewritten == "What is the handover procedure?"
    assert result.complexity == "simple"
    assert result.original == "What is it?"


@pytest.mark.asyncio
async def test_rewrite_query_complex_classification():
    payload = {"rewritten": "Compare X2 and Xn handover latency", "complexity": "complex", "rationale": "comparison"}
    with patch("spar.retrieval.query_rewriter.get_client", return_value=_make_client(json.dumps(payload))):
        result = await rewrite_query("Compare them", history=[], acronyms={})
    assert result.complexity == "complex"


@pytest.mark.asyncio
async def test_rewrite_query_json_parse_failure_fallback():
    with patch("spar.retrieval.query_rewriter.get_client", return_value=_make_client("not json at all")):
        result = await rewrite_query("original query", history=[], acronyms={})
    assert result.rewritten == "original query"
    assert result.complexity == "simple"
    assert result.original == "original query"


@pytest.mark.asyncio
async def test_rewrite_query_llm_exception_fallback():
    bad_client = AsyncMock()
    bad_client.chat = AsyncMock(side_effect=RuntimeError("LLM down"))
    with patch("spar.retrieval.query_rewriter.get_client", return_value=bad_client):
        result = await rewrite_query("original query", history=[], acronyms={})
    assert result.rewritten == "original query"
    assert result.complexity == "simple"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/han/.openclaw/workspace/remote_work/spar
python -m pytest tests/unit/retrieval/test_query_rewriter.py -v 2>&1 | tail -20
```

Expected: ImportError or AttributeError — `rewrite_query` and `QueryRewriteResult` not defined yet.

- [ ] **Step 3: Implement `QueryRewriteResult` and `rewrite_query()` in `query_rewriter.py`**

Add to the TOP of `src/spar/retrieval/query_rewriter.py` (after existing imports):

```python
import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from spar.llm import LLMRole, get_client
from spar.prompts import load_prompt

_log = logging.getLogger(__name__)
_REWRITE_SYSTEM_PROMPT = load_prompt("query_rewrite_system.txt")


@dataclass
class QueryRewriteResult:
    original: str
    rewritten: str
    complexity: Literal["simple", "complex"]
    rationale: str


async def rewrite_query(
    query: str,
    history: list[dict[str, str]],
    acronyms: dict[str, Any],
    max_turns: int = MAX_HISTORY_TURNS,
) -> QueryRewriteResult:
    history_str = format_history(history, max_turns)
    user_content = f"Conversation history:\n{history_str}\n\nQuery: {query}" if history_str else f"Query: {query}"
    try:
        client = await get_client(LLMRole.ROUTER)
        raw = await client.chat(
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=256,
        )
        parsed = json.loads(raw)
        return QueryRewriteResult(
            original=query,
            rewritten=parsed.get("rewritten", query),
            complexity=parsed.get("complexity", "simple"),
            rationale=parsed.get("rationale", ""),
        )
    except Exception as exc:
        _log.warning("rewrite_query fallback — %s: %s", type(exc).__name__, exc)
        return QueryRewriteResult(original=query, rewritten=query, complexity="simple", rationale="fallback")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/unit/retrieval/test_query_rewriter.py -v 2>&1 | tail -20
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/spar/retrieval/query_rewriter.py tests/unit/retrieval/test_query_rewriter.py
git commit -m "feat(retrieval): add QueryRewriteResult + rewrite_query with LLM + fallback"
```

---

## Task 3: State fields

**Files:**
- Modify: `src/spar/pipeline/state.py`

- [ ] **Step 1: Add fields to `SparState`**

In `src/spar/pipeline/state.py`, add two fields under `# preprocess` section:

```python
    # query rewriting
    rewritten_query: str
    query_complexity: str  # "simple" | "complex"
```

Full updated file:

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

    # query rewriting
    rewritten_query: str
    query_complexity: str  # "simple" | "complex"

    # routing
    route_result: RouteResult

    # decomposition
    sub_questions: list[str]

    # retrieval
    raw_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]

    # generation
    answer: str

    # observability
    error: str | None
    node_trace: list[str]
```

- [ ] **Step 2: Verify no import errors**

```bash
python -c "from spar.pipeline.state import SparState; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/spar/pipeline/state.py
git commit -m "feat(pipeline): add rewritten_query and query_complexity to SparState"
```

---

## Task 4: `rewrite_query` node in `Nodes`

**Files:**
- Modify: `src/spar/pipeline/nodes.py`

- [ ] **Step 1: Add import + node method**

In `nodes.py`, add `rewrite_query` to the import from `query_rewriter`:

```python
from spar.retrieval.query_rewriter import build_context, rewrite_query
```

Then add the node method to the `Nodes` class (after `preprocess`, before `prepare_context`):

```python
    async def rewrite_query(self, state: SparState) -> SparState:
        query = state.get("expanded_query") or state["query"]
        history = state.get("history", [])
        result = await rewrite_query(query, history, self._acronyms)
        return {
            **state,
            "rewritten_query": result.rewritten,
            "query_complexity": result.complexity,
            "node_trace": _append_trace(state, f"rewrite_query:{result.complexity}"),
        }
```

- [ ] **Step 2: Verify import**

```bash
python -c "from spar.pipeline.nodes import Nodes; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/spar/pipeline/nodes.py
git commit -m "feat(pipeline): add rewrite_query node to Nodes"
```

---

## Task 5: Wire node into LangGraph

**Files:**
- Modify: `src/spar/pipeline/graph.py`

- [ ] **Step 1: Insert `rewrite_query` node**

In `build_graph()`, add after `g.add_node("preprocess", nodes.preprocess)`:

```python
    g.add_node("rewrite_query", nodes.rewrite_query)
```

Replace edge:
```python
    # BEFORE:
    g.add_edge("preprocess", "prepare_context")

    # AFTER:
    g.add_edge("preprocess", "rewrite_query")
    g.add_edge("rewrite_query", "prepare_context")
```

Full updated `build_graph()` node/edge section:

```python
    g.add_node("preprocess", nodes.preprocess)
    g.add_node("rewrite_query", nodes.rewrite_query)
    g.add_node("prepare_context", nodes.prepare_context)
    g.add_node("route", nodes.route)
    g.add_node("rag_retrieve", nodes.rag_retrieve)
    g.add_node("structured_retrieve", nodes.structured_retrieve)
    g.add_node("multi_hop_retrieve", nodes.multi_hop_retrieve)
    g.add_node("decompose", nodes.decompose)
    g.add_node("decomposed_retrieve", nodes.decomposed_retrieve)
    g.add_node("rerank", nodes.rerank)
    g.add_node("generate", nodes.generate)

    g.set_entry_point("preprocess")
    g.add_edge("preprocess", "rewrite_query")
    g.add_edge("rewrite_query", "prepare_context")
    g.add_edge("prepare_context", "route")
```

Also update downstream nodes that read query to prefer `rewritten_query`. In `nodes.py`, update `prepare_context`, `route`, `decompose`, `rag_retrieve`, `rerank` to prefer `rewritten_query`:

```python
# In prepare_context:
query = state.get("rewritten_query") or state.get("expanded_query") or state["query"]

# In route:
query = state.get("rewritten_query") or state.get("expanded_query") or state["query"]

# In decompose:
query = state.get("rewritten_query") or state.get("expanded_query") or state["query"]

# In rag_retrieve:
query = state.get("rewritten_query") or state.get("expanded_query") or state["query"]

# In rerank:
query = state.get("rewritten_query") or state.get("expanded_query") or state["query"]
```

- [ ] **Step 2: Verify graph compiles**

```bash
python -c "
from unittest.mock import MagicMock
from spar.pipeline.graph import build_graph
g = build_graph(MagicMock(), MagicMock(), MagicMock(), MagicMock())
print('graph nodes:', list(g.nodes))
"
```

Expected: output includes `rewrite_query` in node list, no errors.

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all existing tests pass + 4 new tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/spar/pipeline/graph.py src/spar/pipeline/nodes.py
git commit -m "feat(pipeline): wire rewrite_query node between preprocess and prepare_context"
```

---

## Task 6: Doc updates

**Files:**
- Modify: `docs/prd.md`, `AGENTS.md`, `README.md`

- [ ] **Step 1: Update `docs/prd.md`**

In Task 2.5, check off the implemented items:

```markdown
- [x] 멀티턴 대화에서 질의를 self-contained 형태로 rewrite
- [x] 모호한 지시어 명시화 ('그 파라미터' → 실제 이름)
```

Add completion note under Task 2.5:
```
완료: 2026-05-01 — feat/query-rewriter-llm branch → main
```

Add to Task 2.5 산출물:
```
- [x] `src/spar/retrieval/query_rewriter.py` — `QueryRewriteResult` + `rewrite_query()`
- [x] `src/spar/prompts/query_rewrite_system.txt`
- [x] `src/spar/pipeline/nodes.py` — `rewrite_query` node
- [x] `tests/unit/retrieval/test_query_rewriter.py`
```

- [ ] **Step 2: Update `AGENTS.md` directory map**

Add to retrieval section:
```
spar/retrieval/query_rewriter.py  — build_context(), rewrite_query(), QueryRewriteResult
spar/prompts/query_rewrite_system.txt  — rewrite + classify prompt
```

- [ ] **Step 3: Update `README.md` current state**

Update current state section to reflect Task 2.5 LLM rewriting complete.

- [ ] **Step 4: Commit**

```bash
git add docs/prd.md AGENTS.md README.md
git commit -m "docs: mark Task 2.5 LLM rewrite complete, update dir maps"
```
