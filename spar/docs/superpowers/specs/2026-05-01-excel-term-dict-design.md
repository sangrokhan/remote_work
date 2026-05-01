# Excel Term Dictionary — Chunk Pre-filter Design

**Date:** 2026-05-01  
**Status:** Draft

---

## Problem

No mechanism exists to leverage structured Excel data (parameter names, alarm IDs, MO names) during chunk retrieval. Queries referencing specific domain terms rely solely on BM25/dense hybrid search with no guarantee that term-relevant chunks surface in results.

## Goal

1. Ingest Excel columns → domain term entries in `dictionary/acronyms.json`
2. Tag Milvus chunks with matched terms at ingest time (`keywords` field)
3. At query time, detect terms in query → narrow Milvus search via `array_contains` expr filter

---

## Non-Goals

- Type-based term separation (param vs alarm vs MO) — single flat dictionary
- Runtime / session-scoped Excel upload — offline ingest only
- Fuzzy matching at query time (future work, Phase 2)

---

## Architecture

### Data Flow

```
[Offline — Ingest]

data/*.xlsx
  └─ excel_loader.py
       user specifies columns (config)
       → extracts cell values, normalizes
       → merges into dictionary/acronyms.json
            new entries: {"NRCellDU": {"type": "keyword"}, ...}
            existing acronym entries: {"AMF": {"full": "...", "type": "acronym"}}

  ingest pipeline:
  chunker.dispatch(text, source_doc, doc_type)
    → term_tagger(chunk, keywords_set)
         exact word-boundary match on chunk["text"]
         → chunk["keywords"] = [matched terms, ...] (max 50)
    → Milvus upsert

[Online — Query]

query → preprocess node
  → expand_query(query, acronyms, reverse_index)   [existing]
  → extract_terms(query, keywords_set)              [new]
       → SparState["matched_terms"] = ["NRCellDU"]

  → route node → build_expr(route_result, matched_terms)
       existing expr + " && array_contains(keywords, 'NRCellDU')"

  → hybrid_search(expr=combined_expr)
```

---

## Components

### 1. `src/spar/ingest/excel_loader.py` (new)

```python
def load_excel_terms(path: str | Path, columns: list[str]) -> dict[str, dict]:
    """Read Excel, extract specified column values → {term: {"type": "keyword"}}."""

def merge_into_acronyms(terms: dict, acronyms_path: str | Path) -> None:
    """Merge keyword entries into acronyms.json. Never overwrite type==acronym entries."""
```

- Library: `openpyxl` (already available or add to deps)
- Normalization: strip whitespace; preserve original casing (term matching is case-insensitive at match time)
- Column config: passed as argument (CLI or config file)

### 2. `dictionary/acronyms.json` schema extension

Existing entries unchanged:
```json
{"AMF": {"full": "Access and Mobility Management Function", "type": "acronym"}}
```

New keyword entries added by excel_loader:
```json
{"NRCellDU": {"type": "keyword"},
 "maxRetransmissions": {"type": "keyword"},
 "ALM-001": {"type": "keyword"}}
```

`type` field is the discriminator. Entries without `type` default to `"acronym"` for backwards compatibility.

### 3. `src/spar/preprocessing/abbrev_mapper.py` extensions

```python
def load_keywords(acronyms: dict) -> set[str]:
    """Return set of terms where type == 'keyword'."""

def extract_terms(query: str, keywords: set[str]) -> list[str]:
    """Case-insensitive word-boundary match of keywords against query tokens.
    Returns matched terms in original casing from keywords set."""
```

`expand_query()` unchanged — continues to use only `type == "acronym"` entries.

### 4. `src/spar/ingest/term_tagger.py` (new)

```python
def tag_chunk(chunk: Chunk, keywords: set[str]) -> Chunk:
    """Scan chunk['text'] for keyword matches → populate chunk['keywords'].
    Word-boundary exact match, case-insensitive. Max 50 terms (Milvus schema limit)."""
```

Called after `chunker.dispatch()` in the ingest pipeline.

### 5. `src/spar/pipeline/nodes.py` — `preprocess` node extension

```python
async def preprocess(self, state: SparState) -> SparState:
    query = state["query"]
    expanded = expand_query(query, self._acronyms, self._reverse_index)  # existing
    matched = extract_terms(query, self._keywords)                        # new
    return {**state, "expanded_query": expanded, "matched_terms": matched, ...}
```

`SparState` gains optional field `matched_terms: list[str]`.

`SparNodes.__init__` loads keywords via `load_keywords(self._acronyms)`.

### 6. `src/spar/retrieval/routing.py` — `build_expr` extension

```python
def build_expr(route_result: RouteResult, matched_terms: list[str] | None = None) -> str | None:
    """Existing route-based expr + array_contains filter for each matched term."""
```

When `matched_terms` is non-empty:
```python
term_clauses = [f'array_contains(keywords, "{t}")' for t in matched_terms]
term_expr = " || ".join(term_clauses)  # OR: chunk must match ANY term
# Combined: (existing_expr) && (term_expr)
```

OR semantics (any matched term) avoids over-filtering when query contains multiple terms.

---

## Milvus Schema

`keywords` field exists but `max_length=32` is insufficient — parameter names can exceed 32 chars (e.g., `maxRetransmissionsReestablishment` = 34 chars).

**Required schema change:** increase `max_length` to 128.

```python
FieldSchema(name="keywords", dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR, max_capacity=50, max_length=128)
```

This requires re-creating the Milvus collection (schema is immutable after creation). Re-ingest required after schema change.

---

## State Changes

`SparState` (`pipeline/state.py`):
```python
matched_terms: list[str]  # new optional field, default []
```

`build_expr()` call sites in `nodes.py` updated to pass `matched_terms`.

---

## Ingest CLI / Script

New script or extension to existing ingest script:

```bash
python scripts/ingest_excel.py \
  --file data/parameters.xlsx \
  --columns "Parameter Name" "IE Name"
```

Runs `load_excel_terms()` → `merge_into_acronyms()` → prints summary of added entries.

---

## Fallback Behavior

- No Excel provided → `acronyms.json` has no keyword entries → `extract_terms()` returns `[]` → `build_expr()` unchanged → existing behavior preserved
- Query contains no matching terms → `matched_terms = []` → no expr added → full search
- All matched chunks filtered out by Milvus → Milvus returns empty → handled by existing fallback (no change needed)

---

## Future Work (Phase 2)

- Fuzzy / BM25-based term detection at query time (currently exact match only)
- Type-based filtering (param vs alarm vs MO) if query intent classification is added
- Auto-extraction of Excel columns via LLM header detection

---

## Files Created / Modified

| File | Change |
|------|--------|
| `src/spar/ingest/excel_loader.py` | New |
| `src/spar/ingest/term_tagger.py` | New |
| `src/spar/preprocessing/abbrev_mapper.py` | Add `load_keywords`, `extract_terms` |
| `src/spar/pipeline/nodes.py` | Extend `preprocess`, load keywords |
| `src/spar/pipeline/state.py` | Add `matched_terms` field |
| `src/spar/retrieval/routing.py` | Extend `build_expr` signature |
| `scripts/ingest_excel.py` | New (CLI entry point) |
| `dictionary/acronyms.json` | Schema extended (type field), new keyword entries at runtime |
