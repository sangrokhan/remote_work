# LangGraph Config-Based Eval Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire all eval queries through `graph.ainvoke()` with `GraphConfig`-driven ablation, enabling Recall@K + faithfulness comparison across feature variants.

**Architecture:** `GraphConfig` dataclass controls node inclusion at `build_graph()` time. `eval_suite.py` builds N graphs from preset configs, runs the same goldset through each, and emits a comparison table. `SparState` gains `node_timings` + eval fields. `generate` node gains a real LLM path. Faithfulness scoring uses an inline LLM judge via `spar.llm`.

**Tech Stack:** LangGraph `StateGraph`, `spar.llm.client.LLMClient`, `pytest`, existing `spar.eval.metrics`

---

## File Map

| Path | Action | Responsibility |
|---|---|---|
| `src/spar/pipeline/config.py` | Create | `GraphConfig` dataclass + `PRESET_CONFIGS` |
| `src/spar/pipeline/state.py` | Modify | Add `node_timings`, `gold_chunks`, `gold_answer`, `eval_metrics` |
| `src/spar/pipeline/nodes.py` | Modify | Timing wrapper on every node; real LLM in `generate`; `llm` field |
| `src/spar/pipeline/graph.py` | Modify | Accept `GraphConfig`; conditional node inclusion |
| `src/spar/eval/metrics.py` | Modify | Add `compute_faithfulness`, `compute_suite_metrics` |
| `src/spar/prompts/faithfulness_judge.txt` | Create | LLM judge prompt |
| `src/spar/eval/eval_suite.py` | Create | Multi-config runner + comparison table printer |
| `src/spar/eval/run_eval.py` | Modify | Delegate to graph via `graph.ainvoke()` |
| `tests/pipeline/__init__.py` | Create | pytest package marker |
| `tests/pipeline/test_config.py` | Create | `GraphConfig` + preset tests |
| `tests/pipeline/test_graph_config.py` | Create | `build_graph(config)` conditional node tests |
| `tests/eval/test_metrics_faithfulness.py` | Create | `compute_faithfulness` + `compute_suite_metrics` tests |
| `tests/eval/test_eval_suite.py` | Create | `run_suite` integration tests (mocked graph) |

---

## Task 1: GraphConfig dataclass

**Files:**
- Create: `src/spar/pipeline/config.py`
- Create: `tests/pipeline/__init__.py`
- Create: `tests/pipeline/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/pipeline/test_config.py
from __future__ import annotations

from spar.pipeline.config import GraphConfig, PRESET_CONFIGS


def test_graphconfig_defaults():
    cfg = GraphConfig(name="x")
    assert cfg.use_query_expansion is False
    assert cfg.use_prepare_context is False
    assert cfg.use_reranker is False
    assert cfg.use_real_generate is False


def test_preset_names_unique():
    names = [c.name for c in PRESET_CONFIGS]
    assert len(names) == len(set(names))


def test_preset_baseline_all_false():
    baseline = next(c for c in PRESET_CONFIGS if c.name == "baseline")
    assert not baseline.use_query_expansion
    assert not baseline.use_prepare_context
    assert not baseline.use_reranker
    assert not baseline.use_real_generate


def test_preset_full_retrieval():
    cfg = next(c for c in PRESET_CONFIGS if c.name == "full_retrieval")
    assert cfg.use_query_expansion
    assert cfg.use_prepare_context
    assert cfg.use_reranker
    assert not cfg.use_real_generate


def test_preset_e2e_all_true():
    cfg = next(c for c in PRESET_CONFIGS if c.name == "e2e")
    assert cfg.use_query_expansion
    assert cfg.use_prepare_context
    assert cfg.use_reranker
    assert cfg.use_real_generate
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /home/han/.openclaw/workspace/remote_work/spar
pytest tests/pipeline/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'spar.pipeline.config'`

- [ ] **Step 3: Create `pipeline/config.py`**

```python
# src/spar/pipeline/config.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphConfig:
    name: str
    use_query_expansion: bool = False
    use_prepare_context: bool = False
    use_reranker: bool = False
    use_real_generate: bool = False


PRESET_CONFIGS: list[GraphConfig] = [
    GraphConfig(name="baseline"),
    GraphConfig(name="+reranker", use_reranker=True),
    GraphConfig(name="+qexpand", use_query_expansion=True),
    GraphConfig(name="+context", use_prepare_context=True),
    GraphConfig(
        name="full_retrieval",
        use_query_expansion=True,
        use_prepare_context=True,
        use_reranker=True,
    ),
    GraphConfig(
        name="e2e",
        use_query_expansion=True,
        use_prepare_context=True,
        use_reranker=True,
        use_real_generate=True,
    ),
]
```

- [ ] **Step 4: Create `tests/pipeline/__init__.py`** (empty file)

- [ ] **Step 5: Run tests to verify pass**

```bash
pytest tests/pipeline/test_config.py -v
```

Expected: 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/spar/pipeline/config.py tests/pipeline/__init__.py tests/pipeline/test_config.py
git commit -m "feat(pipeline): add GraphConfig dataclass + PRESET_CONFIGS"
```

---

## Task 2: SparState extensions

**Files:**
- Modify: `src/spar/pipeline/state.py`

- [ ] **Step 1: Write failing test** (append to `tests/pipeline/test_config.py`)

```python
from spar.pipeline.state import SparState


def test_sparstate_has_timing_field():
    s: SparState = {"query": "test", "node_timings": {"preprocess": 12.5}}
    assert s["node_timings"]["preprocess"] == 12.5


def test_sparstate_has_eval_fields():
    s: SparState = {
        "query": "test",
        "gold_chunks": ["section_4.1"],
        "gold_answer": "The answer is 5.",
        "eval_metrics": {"recall_at_5": 1.0},
    }
    assert s["gold_chunks"] == ["section_4.1"]
    assert s["gold_answer"] == "The answer is 5."
    assert s["eval_metrics"]["recall_at_5"] == 1.0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/pipeline/test_config.py::test_sparstate_has_timing_field -v
```

Expected: `KeyError` or missing field.

- [ ] **Step 3: Update `state.py`** — full replacement:

```python
# src/spar/pipeline/state.py
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

    # retrieval
    raw_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]

    # generation
    answer: str

    # observability
    error: str | None
    node_trace: list[str]
    node_timings: dict[str, float]   # node_name -> execution time ms

    # performance eval inputs (populated by eval_suite; ignored in production)
    gold_chunks: list[str] | None
    gold_answer: str | None
    eval_metrics: dict[str, Any]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/pipeline/test_config.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/spar/pipeline/state.py tests/pipeline/test_config.py
git commit -m "feat(pipeline): extend SparState with node_timings and eval fields"
```

---

## Task 3: Node timing wrapper

**Files:**
- Modify: `src/spar/pipeline/nodes.py`
- Create: `tests/pipeline/test_nodes_timing.py`

- [ ] **Step 1: Write failing test**

```python
# tests/pipeline/test_nodes_timing.py
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from spar.pipeline.nodes import Nodes
from spar.pipeline.state import SparState


def _make_nodes() -> Nodes:
    router = MagicMock()
    router.route = AsyncMock(return_value=MagicMock(route=MagicMock()))
    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=[0.9, 0.8])
    encoder = MagicMock()
    encoder.encode = MagicMock(return_value=np.zeros((1, 3)))
    milvus = MagicMock()
    return Nodes.create(
        router=router, reranker=reranker, encoder=encoder,
        milvus=milvus, acronyms_path=None,
    )


@pytest.mark.asyncio
async def test_preprocess_records_timing():
    nodes = _make_nodes()
    state: SparState = {"query": "test query", "node_timings": {}}
    result = await nodes.preprocess(state)
    assert "preprocess" in result["node_timings"]
    assert result["node_timings"]["preprocess"] >= 0.0


@pytest.mark.asyncio
async def test_timing_accumulates_across_nodes():
    nodes = _make_nodes()
    state: SparState = {"query": "test", "node_timings": {}}
    s1 = await nodes.preprocess(state)
    s2 = await nodes.generate(s1)
    assert "preprocess" in s2["node_timings"]
    assert "generate" in s2["node_timings"]
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/pipeline/test_nodes_timing.py -v
```

Expected: FAIL — `node_timings` not in result.

- [ ] **Step 3: Add timing helper and update nodes in `nodes.py`**

Add after `_append_trace`:

```python
import time

def _record_timing(state: SparState, node: str, elapsed_ms: float) -> dict[str, float]:
    timings = dict(state.get("node_timings") or {})
    timings[node] = elapsed_ms
    return timings
```

Update each node to capture timing. Pattern for `preprocess`:

```python
async def preprocess(self, state: SparState) -> SparState:
    t0 = time.monotonic()
    query = state["query"]
    expanded = expand_query(query, self._acronyms, self._reverse_index)
    elapsed = (time.monotonic() - t0) * 1000
    return {
        **state,
        "expanded_query": expanded,
        "node_trace": _append_trace(state, "preprocess"),
        "node_timings": _record_timing(state, "preprocess", elapsed),
    }
```

Apply same pattern (`t0` / `elapsed` / `_record_timing`) to: `prepare_context`, `route`, `rag_retrieve`, `structured_retrieve`, `multi_hop_retrieve`, `rerank`, `generate`.

- [ ] **Step 4: Verify `pytest-asyncio` config**

Check `pyproject.toml` has:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

If missing, add it. Install if needed: `pip install pytest-asyncio`

- [ ] **Step 5: Run tests**

```bash
pytest tests/pipeline/test_nodes_timing.py -v
```

Expected: 2 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/spar/pipeline/nodes.py tests/pipeline/test_nodes_timing.py
git commit -m "feat(pipeline): add per-node timing to SparState.node_timings"
```

---

## Task 4: build_graph with GraphConfig

**Files:**
- Modify: `src/spar/pipeline/graph.py`
- Create: `tests/pipeline/test_graph_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_graph_config.py
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from spar.pipeline.config import GraphConfig, PRESET_CONFIGS
from spar.pipeline.graph import build_graph


def _deps():
    router = MagicMock()
    router.route = AsyncMock(return_value=MagicMock(route=MagicMock()))
    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=[])
    encoder = MagicMock()
    encoder.encode = MagicMock(return_value=np.zeros((1, 3)))
    milvus = MagicMock()
    milvus.hybrid_search = AsyncMock(return_value=[])
    return router, reranker, encoder, milvus


def test_build_graph_baseline_has_no_rerank_node():
    router, reranker, encoder, milvus = _deps()
    cfg = GraphConfig(name="baseline")
    graph = build_graph(router, reranker, encoder, milvus, config=cfg)
    node_names = set(graph.get_graph().nodes.keys())
    assert "rerank" not in node_names
    assert "route" in node_names


def test_build_graph_full_retrieval_has_rerank():
    router, reranker, encoder, milvus = _deps()
    cfg = next(c for c in PRESET_CONFIGS if c.name == "full_retrieval")
    graph = build_graph(router, reranker, encoder, milvus, config=cfg)
    node_names = set(graph.get_graph().nodes.keys())
    assert "rerank" in node_names
    assert "preprocess" in node_names
    assert "prepare_context" in node_names


def test_build_graph_default_preserves_existing_behavior():
    router, reranker, encoder, milvus = _deps()
    graph = build_graph(router, reranker, encoder, milvus)
    node_names = set(graph.get_graph().nodes.keys())
    assert "rerank" in node_names
    assert "preprocess" in node_names
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/pipeline/test_graph_config.py -v
```

Expected: FAIL — `build_graph` does not accept `config` kwarg.

- [ ] **Step 3: Rewrite `graph.py`**

```python
# src/spar/pipeline/graph.py
from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, StateGraph

from spar.encoder.base import EncoderClient
from spar.llm.client import LLMClient
from spar.pipeline.config import GraphConfig, PRESET_CONFIGS
from spar.pipeline.nodes import Nodes
from spar.pipeline.state import SparState
from spar.reranker.client import CrossEncoderClient
from spar.retrieval.milvus_client import SparMilvusClient
from spar.router.hybrid_router import HybridRouter
from spar.router.schemas import Route

_RETRIEVE_NODES = ("rag_retrieve", "structured_retrieve", "multi_hop_retrieve")
_DEFAULT_CONFIG = next(c for c in PRESET_CONFIGS if c.name == "full_retrieval")


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
    config: GraphConfig | None = None,
    acronyms_path: Path | None = None,
    llm: LLMClient | None = None,
):
    cfg = config if config is not None else _DEFAULT_CONFIG

    nodes = Nodes.create(
        router=router,
        reranker=reranker,
        encoder=encoder,
        milvus=milvus,
        acronyms_path=acronyms_path,
        llm=llm,
    )

    g: StateGraph = StateGraph(SparState)

    if cfg.use_query_expansion:
        entry = "preprocess"
    elif cfg.use_prepare_context:
        entry = "prepare_context"
    else:
        entry = "route"
    g.set_entry_point(entry)

    if cfg.use_query_expansion:
        g.add_node("preprocess", nodes.preprocess)
        next_node = "prepare_context" if cfg.use_prepare_context else "route"
        g.add_edge("preprocess", next_node)

    if cfg.use_prepare_context:
        g.add_node("prepare_context", nodes.prepare_context)
        g.add_edge("prepare_context", "route")

    g.add_node("route", nodes.route)
    for name in _RETRIEVE_NODES:
        g.add_node(name, getattr(nodes, name))

    g.add_conditional_edges(
        "route",
        _route_selector,
        {n: n for n in _RETRIEVE_NODES},
    )

    if cfg.use_reranker:
        g.add_node("rerank", nodes.rerank)
        for name in _RETRIEVE_NODES:
            g.add_edge(name, "rerank")
        g.add_edge("rerank", "generate")
    else:
        for name in _RETRIEVE_NODES:
            g.add_edge(name, "generate")

    g.add_node("generate", nodes.generate)
    g.add_edge("generate", END)

    return g.compile()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/pipeline/test_graph_config.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Run all pipeline tests**

```bash
pytest tests/ -k "pipeline" -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/spar/pipeline/graph.py tests/pipeline/test_graph_config.py
git commit -m "feat(pipeline): build_graph accepts GraphConfig for conditional node inclusion"
```

---

## Task 5: Real generate node + LLM wiring

**Files:**
- Modify: `src/spar/pipeline/nodes.py`

- [ ] **Step 1: Write failing test** (append to `tests/pipeline/test_nodes_timing.py`)

```python
from spar.llm.client import LLMClient


@pytest.mark.asyncio
async def test_generate_stub_when_no_llm():
    nodes = _make_nodes()
    state: SparState = {
        "query": "What is maxHARQTx?",
        "raw_chunks": [{"text": "maxHARQTx default is 5.", "score": 0.9, "source_doc": "ref.md", "section_num": "4.1"}],
        "node_timings": {},
    }
    result = await nodes.generate(state)
    assert result["answer"].startswith("[stub]")
    assert "generate" in result["node_timings"]


@pytest.mark.asyncio
async def test_generate_calls_llm_when_provided():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat = AsyncMock(return_value="maxHARQTx default value is 5.")

    nodes = Nodes.create(
        router=MagicMock(), reranker=MagicMock(),
        encoder=MagicMock(encode=MagicMock(return_value=np.zeros((1, 3)))),
        milvus=MagicMock(), acronyms_path=None, llm=mock_llm,
    )
    state: SparState = {
        "query": "What is maxHARQTx?",
        "raw_chunks": [{"text": "maxHARQTx default is 5.", "score": 0.9, "source_doc": "ref.md", "section_num": "4.1"}],
        "node_timings": {},
    }
    result = await nodes.generate(state)
    assert result["answer"] == "maxHARQTx default value is 5."
    mock_llm.chat.assert_called_once()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/pipeline/test_nodes_timing.py::test_generate_calls_llm_when_provided -v
```

Expected: FAIL — `Nodes.create` has no `llm` parameter.

- [ ] **Step 3: Update `Nodes` in `nodes.py`**

Add import at top:
```python
from spar.llm.client import LLMClient
```

Add field to `Nodes` dataclass (after `_reverse_index`):
```python
llm: LLMClient | None = None
```

Update `Nodes.create` signature and return:
```python
@classmethod
def create(
    cls,
    router: HybridRouter,
    reranker: CrossEncoderClient,
    encoder: EncoderClient,
    milvus: SparMilvusClient,
    acronyms_path: Path | None = None,
    llm: LLMClient | None = None,
) -> "Nodes":
    path = acronyms_path or _ACRONYMS_PATH
    if path.exists():
        acronyms = load_acronyms(path)
        reverse_index = build_reverse_index(acronyms)
    else:
        acronyms, reverse_index = {}, {}
    return cls(
        router=router, reranker=reranker, encoder=encoder, milvus=milvus,
        _acronyms=acronyms, _reverse_index=reverse_index, llm=llm,
    )
```

Replace `generate` method:
```python
async def generate(self, state: SparState) -> SparState:
    t0 = time.monotonic()
    chunks = state.get("reranked_chunks") or state.get("raw_chunks", [])
    query = state["query"]

    if self.llm is None:
        answer = f"[stub] context={len(chunks)} chunks\nquery={query}"
    else:
        context = "\n\n".join(c["text"] for c in chunks[:5])
        history_ctx = state.get("history_context", "")
        user_content = f"{history_ctx}\n\nContext:\n{context}\n\nQuestion: {query}".strip()
        messages = [
            {"role": "system", "content": "You are a Samsung RAN expert. Answer using only the provided context."},
            {"role": "user", "content": user_content},
        ]
        answer = await self.llm.chat(messages)

    elapsed = (time.monotonic() - t0) * 1000
    return {
        **state,
        "answer": answer,
        "node_trace": _append_trace(state, "generate"),
        "node_timings": _record_timing(state, "generate", elapsed),
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/pipeline/test_nodes_timing.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/spar/pipeline/nodes.py
git commit -m "feat(pipeline): wire real LLM into generate node; llm=None keeps stub"
```

---

## Task 6: Faithfulness judge prompt + metrics extensions

**Files:**
- Create: `src/spar/prompts/faithfulness_judge.txt`
- Modify: `src/spar/eval/metrics.py`
- Create: `tests/eval/test_metrics_faithfulness.py`

- [ ] **Step 1: Create `faithfulness_judge.txt`**

Content of `src/spar/prompts/faithfulness_judge.txt`:

```
You are a factual grounding evaluator for a RAG system.

Given an answer, the retrieved context passages, and a reference answer, score how well
the answer is grounded in the context.

Score 1.0: Every claim in the answer is directly supported by the context.
Score 0.5: Most claims are supported; minor unsupported additions present.
Score 0.0: Answer contains claims not found in the context, or contradicts it.

Respond with ONLY a decimal score between 0.0 and 1.0. No explanation.
```

- [ ] **Step 2: Write failing tests**

```python
# tests/eval/test_metrics_faithfulness.py
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from spar.eval.metrics import compute_faithfulness, compute_suite_metrics


@pytest.mark.asyncio
async def test_compute_faithfulness_returns_float():
    llm = MagicMock()
    llm.chat = AsyncMock(return_value="0.85")
    chunks = [{"text": "maxHARQTx default is 5.", "score": 0.9}]
    score = await compute_faithfulness(
        answer="The default value is 5.",
        context_chunks=chunks,
        gold_answer="maxHARQTx default value is 5.",
        llm_client=llm,
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_compute_faithfulness_handles_malformed_response():
    llm = MagicMock()
    llm.chat = AsyncMock(return_value="I cannot determine the score.")
    score = await compute_faithfulness(
        answer="some answer",
        context_chunks=[{"text": "some context", "score": 0.5}],
        gold_answer="reference",
        llm_client=llm,
    )
    assert score == 0.0


def test_compute_suite_metrics_aggregates():
    results = [
        {
            "config_name": "baseline",
            "per_query": [
                {"recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0, "latency_ms": 100.0, "faithfulness": None},
                {"recall_at_5": 0.0, "recall_at_10": 1.0, "mrr": 0.5, "latency_ms": 120.0, "faithfulness": None},
            ],
        }
    ]
    table = compute_suite_metrics(results)
    row = table[0]
    assert row["config"] == "baseline"
    assert row["recall_at_5"] == pytest.approx(0.5)
    assert row["recall_at_10"] == pytest.approx(1.0)
    assert row["mrr"] == pytest.approx(0.75)
    assert row["faithfulness"] is None


def test_compute_suite_metrics_faithfulness_average():
    results = [
        {
            "config_name": "+reranker",
            "per_query": [
                {"recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0, "latency_ms": 200.0, "faithfulness": 0.9},
                {"recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0, "latency_ms": 220.0, "faithfulness": 0.7},
            ],
        }
    ]
    table = compute_suite_metrics(results)
    assert table[0]["faithfulness"] == pytest.approx(0.8)
```

- [ ] **Step 3: Run to verify failure**

```bash
pytest tests/eval/test_metrics_faithfulness.py -v
```

Expected: FAIL — `compute_faithfulness` not defined.

- [ ] **Step 4: Append to `src/spar/eval/metrics.py`**

```python
# --- faithfulness and suite aggregation ---

import re
import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spar.llm.client import LLMClient


async def compute_faithfulness(
    answer: str,
    context_chunks: list[dict],
    gold_answer: str,
    llm_client: "LLMClient",
) -> float:
    from spar.prompts import load_prompt
    prompt = load_prompt("faithfulness_judge")
    context_text = "\n\n".join(c["text"] for c in context_chunks[:5])
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                f"Answer: {answer}\n\n"
                f"Context:\n{context_text}\n\n"
                f"Reference answer: {gold_answer}"
            ),
        },
    ]
    response = await llm_client.chat(messages, max_tokens=16)
    match = re.search(r"([01]?\.\d+|[01])", response.strip())
    return float(match.group(1)) if match else 0.0


def compute_suite_metrics(results: list[dict]) -> list[dict]:
    """
    results: list of {config_name, per_query: [{recall_at_5, recall_at_10, mrr, latency_ms, faithfulness}]}
    Returns one summary row per config.
    """
    rows = []
    for r in results:
        pq = r["per_query"]
        if not pq:
            continue

        def avg(key: str) -> float:
            vals = [x[key] for x in pq if x.get(key) is not None]
            return sum(vals) / len(vals) if vals else 0.0

        latencies = sorted(x["latency_ms"] for x in pq if x.get("latency_ms") is not None)
        p50 = statistics.median(latencies) if latencies else 0.0

        faith_vals = [x["faithfulness"] for x in pq if x.get("faithfulness") is not None]
        faith_avg = sum(faith_vals) / len(faith_vals) if faith_vals else None

        rows.append({
            "config": r["config_name"],
            "recall_at_5": avg("recall_at_5"),
            "recall_at_10": avg("recall_at_10"),
            "mrr": avg("mrr"),
            "p50_ms": p50,
            "faithfulness": faith_avg,
        })
    return rows
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/eval/test_metrics_faithfulness.py -v
```

Expected: 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/spar/prompts/faithfulness_judge.txt src/spar/eval/metrics.py tests/eval/test_metrics_faithfulness.py
git commit -m "feat(eval): add compute_faithfulness LLM judge + compute_suite_metrics"
```

---

## Task 7: eval_suite.py

**Files:**
- Create: `src/spar/eval/eval_suite.py`
- Create: `tests/eval/test_eval_suite.py`

- [ ] **Step 1: Write failing test**

```python
# tests/eval/test_eval_suite.py
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from spar.eval.eval_suite import run_suite, print_comparison_table
from spar.pipeline.config import GraphConfig

GOLDSET = [
    {
        "query_id": "q001",
        "query": "What is maxHARQTx default?",
        "type": "parameter_lookup",
        "source_doc": "param_ref.md",
        "section": "4.1",
        "gold_answer": "The default is 5.",
    }
]

FAKE_STATE = {
    "query": "What is maxHARQTx default?",
    "raw_chunks": [
        {"text": "maxHARQTx default is 5.", "score": 0.9, "source_doc": "param_ref.md", "section_num": "4.1"}
    ],
    "reranked_chunks": [],
    "answer": "[stub]",
    "node_timings": {"route": 10.0, "rag_retrieve": 80.0, "generate": 5.0},
    "node_trace": ["route", "rag_retrieve", "generate"],
}


@pytest.mark.asyncio
async def test_run_suite_returns_one_result_per_config():
    configs = [GraphConfig(name="baseline"), GraphConfig(name="+reranker", use_reranker=True)]

    with patch("spar.eval.eval_suite.build_graph") as mock_build:
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=FAKE_STATE)
        mock_build.return_value = mock_graph

        results = await run_suite(
            configs=configs, goldset=GOLDSET,
            router=MagicMock(), reranker=MagicMock(),
            encoder=MagicMock(encode=MagicMock(return_value=np.zeros((1,3)))),
            milvus=MagicMock(), top_k=10,
        )

    assert len(results) == 2
    assert results[0]["config_name"] == "baseline"
    assert len(results[0]["per_query"]) == 1


@pytest.mark.asyncio
async def test_run_suite_per_query_has_recall_and_latency():
    configs = [GraphConfig(name="baseline")]

    with patch("spar.eval.eval_suite.build_graph") as mock_build:
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=FAKE_STATE)
        mock_build.return_value = mock_graph

        results = await run_suite(
            configs=configs, goldset=GOLDSET,
            router=MagicMock(), reranker=MagicMock(),
            encoder=MagicMock(encode=MagicMock(return_value=np.zeros((1,3)))),
            milvus=MagicMock(), top_k=10,
        )

    pq = results[0]["per_query"][0]
    assert "recall_at_5" in pq
    assert "recall_at_10" in pq
    assert "mrr" in pq
    assert "latency_ms" in pq
    assert pq["latency_ms"] == pytest.approx(95.0)  # 10+80+5
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/eval/test_eval_suite.py -v
```

Expected: FAIL — `spar.eval.eval_suite` not found.

- [ ] **Step 3: Create `src/spar/eval/eval_suite.py`**

```python
# src/spar/eval/eval_suite.py
"""
Multi-config performance suite — runs goldset through multiple GraphConfig variants
and prints a comparison table.

Usage:
    python -m spar.eval.eval_suite \
        --goldset data/goldsets/retrieval_goldset.jsonl \
        --configs baseline +reranker +qexpand full_retrieval \
        --top-k 10 \
        --output data/eval_results/suite_YYYYMMDD.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from spar.eval.metrics import (
    compute_faithfulness,
    compute_suite_metrics,
    recall_at_k,
    reciprocal_rank,
)
from spar.pipeline.config import GraphConfig, PRESET_CONFIGS
from spar.pipeline.graph import build_graph


def _load_goldset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def run_suite(
    configs: list[GraphConfig],
    goldset: list[dict],
    router,
    reranker,
    encoder,
    milvus,
    top_k: int = 10,
    llm_client=None,
    acronyms_path: Path | None = None,
) -> list[dict]:
    results = []
    for cfg in configs:
        graph = build_graph(
            router=router, reranker=reranker, encoder=encoder, milvus=milvus,
            config=cfg, acronyms_path=acronyms_path,
            llm=llm_client if cfg.use_real_generate else None,
        )
        per_query: list[dict[str, Any]] = []
        for gold in goldset:
            try:
                state = await graph.ainvoke({
                    "query": gold["query"],
                    "top_k": top_k,
                    "gold_chunks": [gold.get("section", "")],
                    "gold_answer": gold.get("gold_answer"),
                })
            except Exception as exc:
                per_query.append({"error": str(exc), "query_id": gold.get("query_id")})
                continue

            retrieved = state.get("reranked_chunks") or state.get("raw_chunks", [])
            pq: dict[str, Any] = {
                "query_id": gold.get("query_id"),
                "query": gold["query"],
                "recall_at_5": float(recall_at_k(retrieved, gold, 5)),
                "recall_at_10": float(recall_at_k(retrieved, gold, 10)),
                "mrr": reciprocal_rank(retrieved, gold),
                "latency_ms": sum((state.get("node_timings") or {}).values()),
                "faithfulness": None,
            }
            if (
                gold.get("gold_answer")
                and state.get("answer")
                and llm_client
                and cfg.use_real_generate
            ):
                pq["faithfulness"] = await compute_faithfulness(
                    answer=state["answer"],
                    context_chunks=retrieved,
                    gold_answer=gold["gold_answer"],
                    llm_client=llm_client,
                )
            per_query.append(pq)

        results.append({"config_name": cfg.name, "per_query": per_query})
    return results


def print_comparison_table(summary_rows: list[dict]) -> None:
    header = f"{'config':<18} | {'R@5':>6} | {'R@10':>6} | {'MRR':>6} | {'Faith':>6} | {'p50ms':>7}"
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}")
    print("-" * len(header))
    for row in summary_rows:
        faith = f"{row['faithfulness']:.3f}" if row["faithfulness"] is not None else "  -  "
        print(
            f"{row['config']:<18} | "
            f"{row['recall_at_5']:>6.3f} | "
            f"{row['recall_at_10']:>6.3f} | "
            f"{row['mrr']:>6.3f} | "
            f"{faith:>6} | "
            f"{row['p50_ms']:>7.0f}"
        )
    print(f"{sep}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goldset", required=True, type=Path)
    parser.add_argument("--configs", nargs="+", default=["baseline", "full_retrieval"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.goldset.exists():
        print(f"Error: goldset not found: {args.goldset}", file=sys.stderr)
        sys.exit(1)

    preset_map = {c.name: c for c in PRESET_CONFIGS}
    selected = []
    for name in args.configs:
        if name not in preset_map:
            print(f"Unknown config '{name}'. Available: {list(preset_map)}", file=sys.stderr)
            sys.exit(1)
        selected.append(preset_map[name])

    from spar.encoder.registry import get_encoder
    from spar.reranker.registry import get_reranker
    from spar.retrieval.milvus_client import SparMilvusClient
    from spar.router.hybrid_router import HybridRouter

    encoder = get_encoder()
    reranker = get_reranker()
    router = HybridRouter()
    client = SparMilvusClient()
    client.connect()

    goldset = _load_goldset(args.goldset)
    print(f"Goldset: {len(goldset)} queries | configs: {[c.name for c in selected]}")

    results = asyncio.run(
        run_suite(
            configs=selected, goldset=goldset, router=router, reranker=reranker,
            encoder=encoder, milvus=client, top_k=args.top_k,
        )
    )
    summary = compute_suite_metrics(results)
    print_comparison_table(summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": summary, "details": results}, indent=2))
        print(f"Saved: {args.output}")

    client.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/eval/test_eval_suite.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/spar/eval/eval_suite.py tests/eval/test_eval_suite.py
git commit -m "feat(eval): add eval_suite multi-config runner with comparison table"
```

---

## Task 8: Replace run_eval.py with graph-based invocation

**Files:**
- Modify: `src/spar/eval/run_eval.py`

- [ ] **Step 1: Write failing test** (append to `tests/eval/test_eval_suite.py`)

```python
from spar.eval.run_eval import _collect_results_via_graph


@pytest.mark.asyncio
async def test_collect_results_via_graph_returns_gold_and_retrieved():
    with patch("spar.eval.run_eval.build_graph") as mock_build:
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            **FAKE_STATE,
            "reranked_chunks": FAKE_STATE["raw_chunks"],
        })
        mock_build.return_value = mock_graph

        results = await _collect_results_via_graph(
            goldset=[GOLDSET[0]], doc_type="spec", top_k=10,
            router=MagicMock(), reranker=MagicMock(),
            encoder=MagicMock(encode=MagicMock(return_value=np.zeros((1,3)))),
            milvus=MagicMock(),
        )

    assert len(results) == 1
    assert "gold" in results[0]
    assert "retrieved" in results[0]
    assert results[0]["retrieved"] == FAKE_STATE["raw_chunks"]
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/eval/test_eval_suite.py::test_collect_results_via_graph_returns_gold_and_retrieved -v
```

Expected: FAIL — `_collect_results_via_graph` not defined.

- [ ] **Step 3: Rewrite `run_eval.py`**

```python
# src/spar/eval/run_eval.py
"""
Performance evaluation runner — Recall@5/10/50, MRR via graph.ainvoke()

Usage:
    python -m spar.eval.run_eval \
        --goldset data/goldsets/retrieval_goldset.jsonl \
        --doc-type spec \
        --top-k 50 \
        --output data/eval_results/phase1_eval.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from spar.eval.metrics import compute_metrics
from spar.pipeline.config import PRESET_CONFIGS
from spar.pipeline.graph import build_graph


def _load_goldset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def _collect_results_via_graph(
    goldset: list[dict],
    doc_type: str,
    top_k: int,
    router,
    reranker,
    encoder,
    milvus,
    acronyms_path: Path | None = None,
) -> list[dict]:
    cfg = next(c for c in PRESET_CONFIGS if c.name == "full_retrieval")
    graph = build_graph(
        router=router, reranker=reranker, encoder=encoder, milvus=milvus,
        config=cfg, acronyms_path=acronyms_path,
    )
    results = []
    for gold in goldset:
        try:
            state = await graph.ainvoke({"query": gold["query"], "top_k": top_k})
        except Exception as exc:
            print(f"[warn] {gold.get('query_id')} failed: {exc}", file=sys.stderr)
            continue
        retrieved = state.get("reranked_chunks") or state.get("raw_chunks", [])
        results.append({"gold": gold, "retrieved": retrieved})
    return results


def _print_summary(metrics: dict) -> None:
    print(f"\n{'='*50}")
    print(f"  n_queries : {metrics['n_queries']}")
    print(f"  MRR       : {metrics['mrr']:.4f}")
    for k in [5, 10, 50]:
        key = f"recall_at_{k}"
        if key in metrics:
            print(f"  Recall@{k:<3} : {metrics[key]:.4f}")
    print(f"{'='*50}")
    if metrics.get("by_type"):
        print("\n  By query type:")
        for qtype, m in metrics["by_type"].items():
            print(
                f"    {qtype:<15} n={m['n']:<4} "
                f"MRR={m['mrr']:.3f}  "
                f"R@5={m['recall_at_5']:.3f}  "
                f"R@10={m['recall_at_10']:.3f}"
            )


def _save_output(path: Path, metrics: dict, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metrics": metrics,
        "details": [
            {
                "query_id": r["gold"]["query_id"],
                "query": r["gold"]["query"],
                "type": r["gold"].get("type"),
                "expected_doc": r["gold"]["source_doc"],
                "expected_section": r["gold"]["section"],
                "retrieved_top3": r["retrieved"][:3],
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goldset", required=True, type=Path)
    parser.add_argument("--doc-type", default="spec")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.goldset.exists():
        print(f"Error: goldset not found: {args.goldset}", file=sys.stderr)
        sys.exit(1)

    from spar.encoder.registry import get_encoder
    from spar.reranker.registry import get_reranker
    from spar.retrieval.milvus_client import SparMilvusClient
    from spar.router.hybrid_router import HybridRouter

    encoder = get_encoder()
    reranker = get_reranker()
    router = HybridRouter()
    client = SparMilvusClient()
    client.connect()

    goldset = _load_goldset(args.goldset)
    print(f"Goldset: {len(goldset)} queries (doc_type={args.doc_type}, top_k={args.top_k})")

    results = asyncio.run(
        _collect_results_via_graph(
            goldset=goldset, doc_type=args.doc_type, top_k=args.top_k,
            router=router, reranker=reranker, encoder=encoder, milvus=client,
        )
    )
    metrics = compute_metrics(results)
    _print_summary(metrics)

    if args.output:
        _save_output(args.output, metrics, results)

    client.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/eval/ tests/pipeline/ -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/spar/eval/run_eval.py tests/eval/test_eval_suite.py
git commit -m "refactor(eval): replace direct Milvus calls with graph.ainvoke() in run_eval"
```

---

## Task 9: Full suite + docs update

- [ ] **Step 1: Run full test suite**

```bash
cd /home/han/.openclaw/workspace/remote_work/spar
pytest tests/ -v --tb=short
```

Fix any failures before continuing.

- [ ] **Step 2: Update `docs/prd.md`**

In Task 1.7.2 — check off:
- `[x]` eval_suite.py multi-config runner
- `[x]` graph-based eval (run_eval via `graph.ainvoke`)

In Task 5.1 — check off:
- `[x]` node_timings in SparState
- `[x]` real generate node wired to LLM
- `[x]` GraphConfig + PRESET_CONFIGS

- [ ] **Step 3: Update `AGENTS.md` directory map**

Under `eval/`, add:
```
├── eval_suite.py        # multi-config comparison runner
```

Under `pipeline/`, add:
```
├── config.py            # GraphConfig + PRESET_CONFIGS
```

- [ ] **Step 4: Final commit**

```bash
git add docs/prd.md AGENTS.md
git commit -m "docs: update prd + AGENTS for GraphConfig eval pipeline"
```
