# Design: LLM-based Query Rewriting & Complexity Classification

**Date:** 2026-05-01  
**PRD:** Task 2.5 (멀티턴 질의 rewrite, 지시어 명시화)  
**Status:** Approved

---

## Overview

Add a LangGraph node that rewrites ambiguous queries into self-contained form using a small LLM, and classifies them as `simple` or `complex`. The node is optional: removing it from the graph bypasses rewriting and passes the original query directly to retrieval.

---

## Architecture

### Output Schema

```python
@dataclass
class QueryRewriteResult:
    original: str
    rewritten: str
    complexity: Literal["simple", "complex"]
    rationale: str
```

### LLM Call

- Single call via `get_client(LLMRole.ROUTER)`
- Prompt instructs LLM to return JSON with `rewritten`, `complexity`, `rationale`
- JSON parsed from response text (no structured output API — model-agnostic)
- Fallback on parse failure: `rewritten = original`, `complexity = "simple"`, `rationale = "parse_error"`

### Prompt

File: `src/spar/prompts/query_rewrite_system.txt`

Content covers:
1. Resolve co-references ("그 파라미터" → actual parameter name from history)
2. Make query self-contained from history context
3. Classify: `simple` (single-hop, direct lookup) vs `complex` (multi-hop, comparative, requires decomposition)
4. Return JSON only: `{"rewritten": "...", "complexity": "simple|complex", "rationale": "..."}`

---

## Pipeline Integration

### New node: `rewrite_query`

Inserted between `preprocess` and `prepare_context`:

```
preprocess → rewrite_query → prepare_context → route → ...
```

If `rewrite_query` node is removed from graph:

```
preprocess → prepare_context → route → ...
```

No other nodes need modification for bypass to work — `prepare_context` already reads `query` from state; `rewritten_query` is optional (`total=False`).

### State changes (`state.py`)

```python
rewritten_query: str          # LLM-rewritten query; falls back to query if absent
query_complexity: Literal["simple", "complex"]  # classification result
```

Downstream nodes that currently read `state["query"]` for retrieval will prefer `state.get("rewritten_query", state["query"])`.

### Complexity-based branching

`_route_selector` extended: if `query_complexity == "complex"` is set, it can influence routing to `multi_hop_retrieve` or `decompose` (coordinated with Task 2.6 HyDE/multi-query). For now, complexity is logged to `node_trace` and available for downstream use; hard routing branch added when Task 2.6 lands.

---

## Files Changed

| File | Change |
|------|--------|
| `src/spar/retrieval/query_rewriter.py` | Add `QueryRewriteResult` dataclass + `rewrite_query()` function |
| `src/spar/prompts/query_rewrite_system.txt` | New system prompt for rewrite + classify |
| `src/spar/pipeline/state.py` | Add `rewritten_query`, `query_complexity` fields |
| `src/spar/pipeline/nodes.py` | Add `rewrite_query` node method |
| `src/spar/pipeline/graph.py` | Insert `rewrite_query` node between `preprocess` and `prepare_context` |
| `tests/unit/retrieval/test_query_rewriter.py` | Unit tests for `rewrite_query()` with mock LLM |

---

## Error Handling

| Failure | Behavior |
|---------|----------|
| LLM call exception | Log warning, fallback to original query, complexity = "simple" |
| JSON parse error | Same fallback |
| Missing history | Rewrite still runs; just fewer co-refs to resolve |

---

## Testing

- Unit: mock LLM response → assert `QueryRewriteResult` fields
- Unit: malformed JSON response → assert fallback behavior
- Unit: `build_graph()` with node vs without → assert edge structure differs
