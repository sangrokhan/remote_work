# LangGraph Config-Based Eval Pipeline — Design Spec

**Date:** 2026-05-01  
**Status:** Approved  
**Related PRD tasks:** Task 1.7 (골드셋 평가), Task 2.8 (Phase 2 평가), Task 5.1 (LangGraph 재구성)

---

## 1. Problem

`eval/run_eval.py` calls `SparMilvusClient.hybrid_search()` directly, bypassing the LangGraph pipeline. As a result:

- New features added to the graph (query expansion, reranker, context prep) are **not measured** by eval
- No ablation: cannot compare baseline vs +feature
- No E2E eval: generate node is a stub, faithfulness unmeasurable
- No per-node latency visibility

---

## 2. Goal

When a new feature is added as a graph node, running `eval_suite.py` automatically produces a comparison table showing retrieval and generation quality before and after the feature.

**Success criteria:**
- All eval queries flow through `graph.ainvoke()` — zero direct Milvus calls in eval
- `GraphConfig` controls which nodes are active at build time
- Comparison table: Recall@5/10, MRR, Faithfulness, p50 latency × config
- Adding a new feature = add a `GraphConfig` preset + implement the node

---

## 3. Architecture

```
GraphConfig
    │
    ▼
build_graph(config: GraphConfig) → CompiledGraph
    │
    ▼  (eval_suite.py)
for config in PRESET_CONFIGS:
    graph = build_graph(config)
    for gold_item in goldset:
        state = await graph.ainvoke({
            "query": ...,
            "gold_chunks": [...],
            "gold_answer": "...",
        })
        per_query = compute_metrics(state, gold_item)
    aggregate → config_result
    │
    ▼
comparison_table(configs × metrics) → stdout + JSON report
```

---

## 4. Components

### 4.1 `pipeline/config.py` (new)

```python
@dataclass
class GraphConfig:
    name: str
    use_query_expansion: bool = False
    use_prepare_context: bool = False
    use_reranker: bool = False
    use_real_generate: bool = False  # False = stub, True = real LLM
```

**Preset configs** (`PRESET_CONFIGS: list[GraphConfig]`):

| name | expansion | context | reranker | generate |
|---|---|---|---|---|
| `baseline` | ✗ | ✗ | ✗ | ✗ |
| `+reranker` | ✗ | ✗ | ✓ | ✗ |
| `+qexpand` | ✓ | ✗ | ✗ | ✗ |
| `+context` | ✗ | ✓ | ✗ | ✗ |
| `full_retrieval` | ✓ | ✓ | ✓ | ✗ |
| `e2e` | ✓ | ✓ | ✓ | ✓ |

### 4.2 `pipeline/graph.py` — updated

`build_graph(config: GraphConfig)` conditionally adds nodes:

- `use_query_expansion=False` → skip `preprocess` node (pass query through unchanged)
- `use_prepare_context=False` → skip `prepare_context` node
- `use_reranker=False` → edge `[retrieve] → generate` (skip rerank node)
- `use_real_generate=False` → `generate` node uses stub

Entry point and conditional routing edges remain unchanged.

### 4.3 `pipeline/state.py` — extended

New fields (all optional, non-breaking):

```python
# per-node execution time in ms
node_timings: dict[str, float]

# eval-only inputs (populated by eval_suite, ignored in production)
gold_chunks: list[str] | None    # expected section IDs
gold_answer: str | None          # reference answer text

# per-query eval output
eval_metrics: dict[str, Any]
```

### 4.4 `pipeline/nodes.py` — updated

- Each node wraps execution in a timing context: records `node_timings[node_name]` in state
- `generate` node: when `use_real_generate=True`, calls `spar.llm` client with retrieved context
- Timing wrapper pattern: `t0 = time.monotonic()` → execute → `timings[name] = (time.monotonic() - t0) * 1000`

### 4.5 `eval/eval_suite.py` (new)

```
Usage:
  python -m spar.eval.eval_suite \
    --goldset data/goldsets/retrieval_goldset.jsonl \
    --configs baseline +reranker +qexpand full_retrieval \
    --top-k 10 \
    --output data/eval_results/suite_YYYYMMDD.json
```

Responsibilities:
1. Load goldset (extended format — see 4.7)
2. For each config: build graph, run all queries, collect states
3. Aggregate per-query metrics → config-level summary
4. Print comparison table to stdout
5. Save full JSON report

### 4.6 `eval/run_eval.py` — replaced

Old: calls `SparMilvusClient.hybrid_search()` directly.  
New: thin wrapper around `eval_suite.py` with `--configs full_retrieval`.  
Backward-compatible CLI interface preserved.

### 4.7 Goldset format extension

Existing fields preserved. Add optional:

```jsonl
{
  "query_id": "q001",
  "query": "What is the default value of maxHARQTx?",
  "type": "parameter_lookup",
  "source_doc": "parameter_reference_lte.md",
  "section": "maxHARQTx",
  "gold_answer": "The default value of maxHARQTx is 5."
}
```

`gold_answer` is optional — if absent, faithfulness score is skipped for that item.

### 4.8 `eval/metrics.py` — extended

New functions:

```python
def compute_faithfulness(answer: str, context_chunks: list[str], gold_answer: str, llm_client) -> float:
    """LLM judge: is the answer grounded in the provided context? Returns 0.0–1.0."""

def compute_suite_metrics(results: list[dict]) -> dict:
    """Aggregate Recall@K, MRR, faithfulness, latency from per-query results."""
```

Faithfulness judge prompt (in `spar/prompts/faithfulness_judge.txt`):
- Input: answer, retrieved context, gold answer
- Output: score 0–1 + one-line justification
- Uses existing `spar.llm` client — no new dependency

---

## 5. Data Flow (per query)

```
gold_item
    │ query, gold_chunks, gold_answer
    ▼
preprocess (if enabled) ──► expanded_query
    │                        node_timings["preprocess"]
    ▼
prepare_context (if enabled) ──► history_context
    │
    ▼
route ──► route_result
    │
    ▼
[rag_retrieve | structured_retrieve | multi_hop_retrieve]
    │ raw_chunks, node_timings["*_retrieve"]
    ▼
rerank (if enabled) ──► reranked_chunks
    │                    node_timings["rerank"]
    ▼
generate ──► answer
    │         node_timings["generate"]
    ▼
eval_suite.collect():
    recall = hits(reranked_chunks, gold_chunks) / len(gold_chunks)
    mrr    = 1 / rank_of_first_hit(reranked_chunks, gold_chunks)
    faith  = compute_faithfulness(answer, chunks, gold_answer)  # if use_real_generate
    lat    = sum(node_timings.values())
```

---

## 6. Error Handling

- Node failure → `state["error"]` set, `node_trace` records failure point
- Eval harness: per-query errors logged, do not abort suite run
- Missing `gold_answer` → faithfulness = `None` (excluded from aggregate)

---

## 7. What Is NOT in Scope

- RAGAS integration (Phase 4)
- Streaming eval
- A/B statistical significance testing
- Eval CI gate (INF-4 — separate task)

---

## 8. Files Created / Modified

| Path | Action |
|---|---|
| `src/spar/pipeline/config.py` | Create |
| `src/spar/pipeline/graph.py` | Modify — accept `GraphConfig` |
| `src/spar/pipeline/state.py` | Modify — add timing + eval fields |
| `src/spar/pipeline/nodes.py` | Modify — timing wrapper, real generate |
| `src/spar/eval/eval_suite.py` | Create |
| `src/spar/eval/run_eval.py` | Modify — delegate to eval_suite |
| `src/spar/eval/metrics.py` | Modify — add faithfulness, suite aggregation |
| `src/spar/prompts/faithfulness_judge.txt` | Create |
