# Excel Term Dictionary + Chunk Pre-filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load domain terms (parameter names, alarm IDs, MO names) from Excel columns into `dictionary/acronyms.json`, tag Milvus chunks with matched terms at ingest time, and narrow hybrid search scope at query time via Milvus `array_contains` expr filter.

**Architecture:** Excel files in `data/` are parsed offline; extracted terms are stored in a new `"keywords"` top-level section of `dictionary/acronyms.json`. During ingest, each chunk's `keywords` ARRAY field is populated by exact word-boundary matching against this set. At query time, the `preprocess` pipeline node extracts matched terms from the query and passes them to `build_expr()`, which appends `array_contains` clauses to the Milvus filter expression.

**Tech Stack:** Python 3.12, openpyxl, Milvus (pymilvus), LangGraph pipeline (SparState/SparNodes), pytest

---

## Schema Note

`acronyms.json` uses a nested structure: `{"global": {...}, "conflicts": {...}}`. Excel keywords go into a new top-level key `"keywords"`:

```json
{
  "global": {"HO": {"expansion": "Handover", "variants": []}},
  "conflicts": {"CA": {"candidates": [...], "variants": []}},
  "keywords": {"NRCellDU": {}, "maxRetransmissions": {}, "ALM-001": {}}
}
```

`load_keywords(acronyms)` reads `acronyms.get("keywords", {}).keys()`. Existing `global`/`conflicts` sections untouched.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/spar/retrieval/milvus_client.py` | Modify | `keywords` field `max_length` 32→128 |
| `src/spar/preprocessing/abbrev_mapper.py` | Modify | Add `load_keywords`, `extract_terms` |
| `src/spar/ingest/excel_loader.py` | Create | Parse Excel, merge terms into acronyms.json |
| `src/spar/ingest/term_tagger.py` | Create | Tag chunk `keywords` field at ingest time |
| `src/spar/retrieval/routing.py` | Modify | `build_expr` accepts optional `matched_terms` |
| `src/spar/pipeline/state.py` | Modify | Add `matched_terms: list[str]` to `SparState` |
| `src/spar/pipeline/nodes.py` | Modify | `preprocess` calls `extract_terms`; load `_keywords` |
| `scripts/ingest_excel.py` | Create | CLI entry point for Excel → acronyms.json |
| `tests/retrieval/test_milvus_client.py` | Modify | Assert `max_length=128` |
| `tests/preprocessing/test_abbrev_mapper.py` | Modify | Tests for `load_keywords`, `extract_terms` |
| `tests/ingest/test_excel_loader.py` | Create | Tests for `load_excel_terms`, `merge_into_acronyms` |
| `tests/ingest/test_term_tagger.py` | Create | Tests for `tag_chunk` |
| `tests/retrieval/test_routing.py` | Modify | Tests for `build_expr` with `matched_terms` |
| `tests/pipeline/test_nodes.py` | Modify | Tests for updated `preprocess` |
| `pyproject.toml` | Modify | Add `openpyxl` dependency |

---

## Task 1: Add openpyxl dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add openpyxl to dependencies**

Open `pyproject.toml`. In the `[project] dependencies` list, add:
```toml
"openpyxl>=3.1",
```

- [ ] **Step 2: Install**

```bash
pip install openpyxl
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add openpyxl dependency for Excel ingestion"
```

---

## Task 2: Fix Milvus keywords field max_length (32 → 128)

**Files:**
- Modify: `src/spar/retrieval/milvus_client.py:101-106`
- Test: `tests/retrieval/test_milvus_client.py`

- [ ] **Step 1: Write failing test**

In `tests/retrieval/test_milvus_client.py`, add:

```python
from spar.retrieval.milvus_client import _build_schema

def test_keywords_field_max_length():
    schema = _build_schema()
    kw_field = next(f for f in schema.fields if f.name == "keywords")
    assert kw_field.max_length == 128
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/retrieval/test_milvus_client.py::test_keywords_field_max_length -v
```

Expected: FAIL — `assert 32 == 128`

- [ ] **Step 3: Fix schema**

In `src/spar/retrieval/milvus_client.py`, find the `keywords` field definition and change `max_length=32` to `max_length=128`:

```python
FieldSchema(
    name="keywords",
    dtype=DataType.ARRAY,
    element_type=DataType.VARCHAR,
    max_capacity=50,
    max_length=128,
),
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/retrieval/test_milvus_client.py::test_keywords_field_max_length -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/spar/retrieval/milvus_client.py tests/retrieval/test_milvus_client.py
git commit -m "fix(schema): increase keywords field max_length from 32 to 128"
```

---

## Task 3: abbrev_mapper — add load_keywords and extract_terms

**Files:**
- Modify: `src/spar/preprocessing/abbrev_mapper.py`
- Test: `tests/preprocessing/test_abbrev_mapper.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/preprocessing/test_abbrev_mapper.py`:

```python
from spar.preprocessing.abbrev_mapper import load_keywords, extract_terms

ACRONYMS_WITH_KEYWORDS = {
    "global": {"HO": {"expansion": "Handover", "variants": []}},
    "conflicts": {},
    "keywords": {"NRCellDU": {}, "maxRetransmissions": {}, "ALM-001": {}},
}


class TestLoadKeywords:
    def test_returns_keyword_set(self):
        result = load_keywords(ACRONYMS_WITH_KEYWORDS)
        assert result == {"NRCellDU", "maxRetransmissions", "ALM-001"}

    def test_empty_when_no_keywords_section(self):
        result = load_keywords({"global": {}, "conflicts": {}})
        assert result == set()


class TestExtractTerms:
    def test_exact_match(self):
        kws = {"NRCellDU", "maxRetransmissions"}
        result = extract_terms("What is NRCellDU configuration?", kws)
        assert "NRCellDU" in result
        assert "maxRetransmissions" not in result

    def test_case_insensitive(self):
        kws = {"NRCellDU"}
        result = extract_terms("configure nrcelldu now", kws)
        assert "NRCellDU" in result

    def test_word_boundary(self):
        # "NRCellDUx" should NOT match "NRCellDU"
        kws = {"NRCellDU"}
        result = extract_terms("NRCellDUx parameter", kws)
        assert result == []

    def test_multiple_matches(self):
        kws = {"NRCellDU", "maxRetransmissions", "ALM-001"}
        result = extract_terms("NRCellDU maxRetransmissions settings", kws)
        assert set(result) == {"NRCellDU", "maxRetransmissions"}

    def test_empty_query(self):
        assert extract_terms("", {"NRCellDU"}) == []

    def test_empty_keywords(self):
        assert extract_terms("NRCellDU config", set()) == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/preprocessing/test_abbrev_mapper.py::TestLoadKeywords tests/preprocessing/test_abbrev_mapper.py::TestExtractTerms -v
```

Expected: FAIL — `ImportError: cannot import name 'load_keywords'`

- [ ] **Step 3: Implement load_keywords and extract_terms**

Add to `src/spar/preprocessing/abbrev_mapper.py` (top — ensure `import re` already present):

```python
def load_keywords(acronyms: dict) -> set[str]:
    return set(acronyms.get("keywords", {}).keys())


def extract_terms(query: str, keywords: set[str]) -> list[str]:
    if not query or not keywords:
        return []
    matched: list[str] = []
    for kw in keywords:
        pattern = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
        if pattern.search(query):
            matched.append(kw)
    return matched
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/preprocessing/test_abbrev_mapper.py::TestLoadKeywords tests/preprocessing/test_abbrev_mapper.py::TestExtractTerms -v
```

Expected: all PASS

- [ ] **Step 5: Run full abbrev_mapper test suite to check no regression**

```bash
pytest tests/preprocessing/test_abbrev_mapper.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/spar/preprocessing/abbrev_mapper.py tests/preprocessing/test_abbrev_mapper.py
git commit -m "feat(preprocessing): add load_keywords and extract_terms to abbrev_mapper"
```

---

## Task 4: excel_loader — parse Excel + merge into acronyms.json

**Files:**
- Create: `src/spar/ingest/excel_loader.py`
- Create: `tests/ingest/test_excel_loader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/ingest/test_excel_loader.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import openpyxl
import pytest

from spar.ingest.excel_loader import load_excel_terms, merge_into_acronyms


def _make_excel(tmp_path: Path, rows: list[dict]) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    if rows:
        ws.append(list(rows[0].keys()))
        for row in rows:
            ws.append(list(row.values()))
    path = tmp_path / "test.xlsx"
    wb.save(path)
    return path


class TestLoadExcelTerms:
    def test_extracts_single_column(self, tmp_path):
        path = _make_excel(tmp_path, [
            {"Parameter Name": "maxRetransmissions", "Description": "Max retrans"},
            {"Parameter Name": "t301", "Description": "Timer T301"},
        ])
        result = load_excel_terms(path, columns=["Parameter Name"])
        assert "maxRetransmissions" in result
        assert "t301" in result
        assert result["maxRetransmissions"] == {"type": "keyword"}

    def test_extracts_multiple_columns(self, tmp_path):
        path = _make_excel(tmp_path, [
            {"Param": "maxRetrans", "Alarm": "ALM-001"},
        ])
        result = load_excel_terms(path, columns=["Param", "Alarm"])
        assert "maxRetrans" in result
        assert "ALM-001" in result

    def test_skips_empty_cells(self, tmp_path):
        path = _make_excel(tmp_path, [
            {"Parameter Name": ""},
            {"Parameter Name": None},
            {"Parameter Name": "validParam"},
        ])
        result = load_excel_terms(path, columns=["Parameter Name"])
        assert list(result.keys()) == ["validParam"]

    def test_ignores_unknown_column(self, tmp_path):
        path = _make_excel(tmp_path, [{"Param": "p1"}])
        result = load_excel_terms(path, columns=["NonExistent"])
        assert result == {}

    def test_strips_whitespace(self, tmp_path):
        path = _make_excel(tmp_path, [{"Param": "  NRCellDU  "}])
        result = load_excel_terms(path, columns=["Param"])
        assert "NRCellDU" in result


class TestMergeIntoAcronyms:
    def test_adds_keywords_section(self, tmp_path):
        acronyms_path = tmp_path / "acronyms.json"
        acronyms_path.write_text(json.dumps({"global": {}, "conflicts": {}}))
        merge_into_acronyms({"NRCellDU": {"type": "keyword"}}, acronyms_path)
        data = json.loads(acronyms_path.read_text())
        assert data["keywords"]["NRCellDU"] == {"type": "keyword"}

    def test_does_not_overwrite_global_entries(self, tmp_path):
        acronyms_path = tmp_path / "acronyms.json"
        existing = {"global": {"HO": {"expansion": "Handover"}}, "conflicts": {}}
        acronyms_path.write_text(json.dumps(existing))
        merge_into_acronyms({"HO": {"type": "keyword"}}, acronyms_path)
        data = json.loads(acronyms_path.read_text())
        # "HO" in global must remain untouched; may also appear in keywords
        assert data["global"]["HO"]["expansion"] == "Handover"

    def test_creates_file_if_not_exists(self, tmp_path):
        acronyms_path = tmp_path / "new_acronyms.json"
        merge_into_acronyms({"p1": {"type": "keyword"}}, acronyms_path)
        data = json.loads(acronyms_path.read_text())
        assert "p1" in data["keywords"]

    def test_merges_without_removing_existing_keywords(self, tmp_path):
        acronyms_path = tmp_path / "acronyms.json"
        acronyms_path.write_text(json.dumps({"global": {}, "conflicts": {}, "keywords": {"existing": {}}}))
        merge_into_acronyms({"new": {"type": "keyword"}}, acronyms_path)
        data = json.loads(acronyms_path.read_text())
        assert "existing" in data["keywords"]
        assert "new" in data["keywords"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/ingest/test_excel_loader.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'spar.ingest.excel_loader'`

- [ ] **Step 3: Implement excel_loader.py**

Create `src/spar/ingest/excel_loader.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import openpyxl


def load_excel_terms(path: str | Path, columns: list[str]) -> dict[str, dict]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active

    header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
    headers = [str(h).strip() if h is not None else "" for h in header_row]
    col_indices = [headers.index(c) for c in columns if c in headers]

    terms: dict[str, dict] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        for idx in col_indices:
            if idx < len(row) and row[idx] is not None:
                term = str(row[idx]).strip()
                if term:
                    terms[term] = {"type": "keyword"}

    wb.close()
    return terms


def merge_into_acronyms(terms: dict[str, dict], acronyms_path: str | Path) -> None:
    path = Path(acronyms_path)
    if path.exists():
        data: dict = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {"global": {}, "conflicts": {}, "keywords": {}}

    if "keywords" not in data:
        data["keywords"] = {}

    for term, entry in terms.items():
        data["keywords"][term] = entry

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/ingest/test_excel_loader.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/spar/ingest/excel_loader.py tests/ingest/test_excel_loader.py
git commit -m "feat(ingest): add excel_loader for Excel term extraction"
```

---

## Task 5: term_tagger — tag chunk keywords field

**Files:**
- Create: `src/spar/ingest/term_tagger.py`
- Create: `tests/ingest/test_term_tagger.py`

- [ ] **Step 1: Write failing tests**

Create `tests/ingest/test_term_tagger.py`:

```python
from __future__ import annotations

from spar.ingest.term_tagger import tag_chunk


def _chunk(text: str) -> dict:
    return {
        "chunk_id": "abc123",
        "doc_type": "spec",
        "source_doc": "doc.md",
        "text": text,
        "keywords": [],
    }


class TestTagChunk:
    def test_matches_present_keyword(self):
        chunk = _chunk("The NRCellDU object configures the cell.")
        result = tag_chunk(chunk, {"NRCellDU", "maxRetransmissions"})
        assert "NRCellDU" in result["keywords"]

    def test_no_match_returns_empty(self):
        chunk = _chunk("This is an unrelated paragraph.")
        result = tag_chunk(chunk, {"NRCellDU"})
        assert result["keywords"] == []

    def test_case_insensitive(self):
        chunk = _chunk("configure nrcelldu parameters")
        result = tag_chunk(chunk, {"NRCellDU"})
        assert "NRCellDU" in result["keywords"]

    def test_word_boundary_prevents_partial_match(self):
        chunk = _chunk("NRCellDUExtra is not NRCellDU")
        result = tag_chunk(chunk, {"NRCellDU"})
        assert "NRCellDU" in result["keywords"]
        assert len(result["keywords"]) == 1  # only one match

    def test_caps_at_50(self):
        keywords = {f"term{i}" for i in range(60)}
        text = " ".join(keywords)
        chunk = _chunk(text)
        result = tag_chunk(chunk, keywords)
        assert len(result["keywords"]) <= 50

    def test_returns_same_chunk_reference(self):
        chunk = _chunk("NRCellDU")
        result = tag_chunk(chunk, {"NRCellDU"})
        assert result is chunk
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/ingest/test_term_tagger.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'spar.ingest.term_tagger'`

- [ ] **Step 3: Implement term_tagger.py**

Create `src/spar/ingest/term_tagger.py`:

```python
from __future__ import annotations

import re
from typing import Any

Chunk = dict[str, Any]


def tag_chunk(chunk: Chunk, keywords: set[str]) -> Chunk:
    text = chunk.get("text", "")
    matched: list[str] = []
    for kw in keywords:
        if len(matched) >= 50:
            break
        pattern = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
        if pattern.search(text):
            matched.append(kw)
    chunk["keywords"] = matched
    return chunk
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/ingest/test_term_tagger.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/spar/ingest/term_tagger.py tests/ingest/test_term_tagger.py
git commit -m "feat(ingest): add term_tagger for chunk keywords population"
```

---

## Task 6: routing.py — extend build_expr with matched_terms

**Files:**
- Modify: `src/spar/retrieval/routing.py`
- Test: `tests/retrieval/test_routing.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/retrieval/test_routing.py`:

```python
def test_build_expr_with_matched_terms():
    result = _result(Route.DEFAULT_RAG)
    expr = build_expr(result, matched_terms=["NRCellDU"])
    assert expr is not None
    assert 'array_contains(keywords, "NRCellDU")' in expr


def test_build_expr_multiple_terms():
    result = _result(Route.DEFAULT_RAG)
    expr = build_expr(result, matched_terms=["NRCellDU", "maxRetransmissions"])
    assert 'array_contains(keywords, "NRCellDU")' in expr
    assert 'array_contains(keywords, "maxRetransmissions")' in expr


def test_build_expr_terms_combined_with_product_filter():
    result = _result(Route.DEFAULT_RAG, product="samsung_ran")
    expr = build_expr(result, matched_terms=["NRCellDU"])
    assert "product" in expr
    assert 'array_contains(keywords, "NRCellDU")' in expr
    assert "&&" in expr


def test_build_expr_empty_terms_unchanged():
    result = _result(Route.DEFAULT_RAG)
    expr_no_terms = build_expr(result)
    expr_empty_terms = build_expr(result, matched_terms=[])
    assert expr_no_terms == expr_empty_terms


def test_build_expr_none_terms_unchanged():
    result = _result(Route.DEFAULT_RAG)
    expr_none = build_expr(result, matched_terms=None)
    expr_base = build_expr(result)
    assert expr_none == expr_base
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/retrieval/test_routing.py::test_build_expr_with_matched_terms tests/retrieval/test_routing.py::test_build_expr_multiple_terms -v
```

Expected: FAIL — `TypeError: build_expr() got an unexpected keyword argument 'matched_terms'`

- [ ] **Step 3: Extend build_expr**

In `src/spar/retrieval/routing.py`, replace the existing `build_expr` function:

```python
def build_expr(
    result: RouteResult,
    matched_terms: list[str] | None = None,
) -> str | None:
    clauses: list[str] = []

    if result.product and result.product != "both":
        clauses.append(f'product == "{result.product}"')
    if result.release:
        clauses.append(f'release == "{result.release}"')

    if matched_terms:
        term_clauses = [f'array_contains(keywords, "{t}")' for t in matched_terms]
        clauses.append("(" + " || ".join(term_clauses) + ")")

    return " && ".join(clauses) if clauses else None
```

- [ ] **Step 4: Run all routing tests**

```bash
pytest tests/retrieval/test_routing.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/spar/retrieval/routing.py tests/retrieval/test_routing.py
git commit -m "feat(routing): extend build_expr with matched_terms array_contains filter"
```

---

## Task 7: state.py — add matched_terms field

**Files:**
- Modify: `src/spar/pipeline/state.py`

- [ ] **Step 1: Add field**

In `src/spar/pipeline/state.py`, add `matched_terms` to `SparState`:

```python
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
    matched_terms: list[str]          # new — terms matched from keywords dict

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

- [ ] **Step 2: Verify existing pipeline tests still pass**

```bash
pytest tests/pipeline/ -v
```

Expected: all PASS (TypedDict with `total=False` — no runtime change)

- [ ] **Step 3: Commit**

```bash
git add src/spar/pipeline/state.py
git commit -m "feat(state): add matched_terms field to SparState"
```

---

## Task 8: nodes.py — wire keywords into preprocess and retrieve

**Files:**
- Modify: `src/spar/pipeline/nodes.py`
- Test: `tests/pipeline/test_nodes.py` or `tests/pipeline/test_nodes_retrieval.py`

- [ ] **Step 1: Write failing tests**

In `tests/pipeline/test_nodes.py` (or `test_nodes_retrieval.py`), add:

```python
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from spar.pipeline.nodes import SparNodes
from spar.pipeline.state import SparState


def _make_nodes(keywords: set[str] | None = None) -> SparNodes:
    nodes = SparNodes.__new__(SparNodes)
    nodes._acronyms = {"global": {}, "conflicts": {}, "keywords": {k: {} for k in (keywords or set())}}
    nodes._reverse_index = {}
    nodes._keywords = keywords or set()
    nodes.router = MagicMock()
    nodes.reranker = MagicMock()
    nodes.encoder = MagicMock()
    nodes.milvus = MagicMock()
    nodes._decomposer = MagicMock()
    return nodes


@pytest.mark.asyncio
async def test_preprocess_populates_matched_terms():
    nodes = _make_nodes(keywords={"NRCellDU", "maxRetransmissions"})
    state: SparState = {"query": "What is NRCellDU config?"}
    result = await nodes.preprocess(state)
    assert "NRCellDU" in result["matched_terms"]
    assert "maxRetransmissions" not in result["matched_terms"]


@pytest.mark.asyncio
async def test_preprocess_empty_matched_terms_when_no_match():
    nodes = _make_nodes(keywords={"ALM-001"})
    state: SparState = {"query": "general question about 5G"}
    result = await nodes.preprocess(state)
    assert result["matched_terms"] == []


@pytest.mark.asyncio
async def test_preprocess_empty_matched_terms_when_no_keywords():
    nodes = _make_nodes(keywords=set())
    state: SparState = {"query": "NRCellDU question"}
    result = await nodes.preprocess(state)
    assert result["matched_terms"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/pipeline/test_nodes.py -k "matched_terms" -v
```

Expected: FAIL — `AttributeError: 'SparNodes' object has no attribute '_keywords'`

- [ ] **Step 3: Update imports in nodes.py**

At the top of `src/spar/pipeline/nodes.py`, extend the abbrev_mapper import:

```python
from spar.preprocessing.abbrev_mapper import (
    build_reverse_index,
    expand_query,
    extract_terms,
    load_acronyms,
    load_keywords,
)
```

- [ ] **Step 4: Add _keywords to SparNodes dataclass**

In `src/spar/pipeline/nodes.py`, find the `@dataclass` class definition and add:

```python
@dataclass
class SparNodes:
    router: HybridRouter
    reranker: CrossEncoderClient
    encoder: EncoderClient
    milvus: SparMilvusClient
    _acronyms: dict
    _reverse_index: dict[str, str]
    _keywords: set[str]          # new
    _decomposer: QueryDecomposer = field(default_factory=QueryDecomposer)
```

(Add `from dataclasses import dataclass, field` if `field` not already imported.)

- [ ] **Step 5: Update create() classmethod to load keywords**

In `src/spar/pipeline/nodes.py`, inside `create()`, after the lines that load `acronyms` and `reverse_index`:

```python
acronyms = load_acronyms(path)
reverse_index = build_reverse_index(acronyms)
keywords = load_keywords(acronyms)           # new
```

And pass to the constructor:

```python
return cls(
    router=router,
    reranker=reranker,
    encoder=encoder,
    milvus=milvus,
    _acronyms=acronyms,
    _reverse_index=reverse_index,
    _keywords=keywords,                      # new
)
```

- [ ] **Step 6: Update preprocess node**

Replace the existing `preprocess` method:

```python
async def preprocess(self, state: SparState) -> SparState:
    query = state["query"]
    expanded = expand_query(query, self._acronyms, self._reverse_index)
    matched = extract_terms(query, self._keywords)
    return {
        **state,
        "expanded_query": expanded,
        "matched_terms": matched,
        "node_trace": _append_trace(state, "preprocess"),
    }
```

- [ ] **Step 7: Pass matched_terms to build_expr call sites**

There are multiple call sites in `nodes.py` that call `build_expr(route_result)`. Update each one:

```python
# in rag_retrieve:
matched_terms = state.get("matched_terms", [])
expr = build_expr(route_result, matched_terms=matched_terms)

# in decomposed_retrieve (inside _retrieve_one closure):
matched_terms = state.get("matched_terms", [])
expr = build_expr(route_result, matched_terms=matched_terms)

# in structured_retrieve and multi_hop_retrieve — these delegate to rag_retrieve,
# so no additional change needed there.
```

- [ ] **Step 8: Run tests**

```bash
pytest tests/pipeline/test_nodes.py -v
```

Expected: all PASS

- [ ] **Step 9: Run full test suite**

```bash
pytest -v
```

Expected: all PASS. Check for no regressions in `test_nodes_retrieval.py`.

- [ ] **Step 10: Commit**

```bash
git add src/spar/pipeline/nodes.py
git commit -m "feat(nodes): wire keyword extraction into preprocess node and build_expr"
```

---

## Task 9: scripts/ingest_excel.py — CLI entry point

**Files:**
- Create: `scripts/ingest_excel.py`

- [ ] **Step 1: Implement CLI script**

Create `scripts/ingest_excel.py`:

```python
#!/usr/bin/env python3
"""CLI: load Excel term columns into dictionary/acronyms.json."""
from __future__ import annotations

import argparse
from pathlib import Path

from spar.ingest.excel_loader import load_excel_terms, merge_into_acronyms

_DEFAULT_ACRONYMS = Path(__file__).parent.parent / "dictionary" / "acronyms.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Excel column terms into acronyms.json keyword section.")
    parser.add_argument("--file", required=True, help="Path to Excel file (.xlsx)")
    parser.add_argument("--columns", nargs="+", required=True, help="Column names to extract as terms")
    parser.add_argument(
        "--acronyms",
        default=str(_DEFAULT_ACRONYMS),
        help=f"Path to acronyms.json (default: {_DEFAULT_ACRONYMS})",
    )
    args = parser.parse_args()

    terms = load_excel_terms(args.file, args.columns)
    print(f"Extracted {len(terms)} terms from {args.file}")

    merge_into_acronyms(terms, args.acronyms)
    print(f"Merged into {args.acronyms}")

    for term in sorted(terms)[:20]:
        print(f"  + {term}")
    if len(terms) > 20:
        print(f"  ... and {len(terms) - 20} more")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test manually**

```bash
# Create a small test Excel first:
python - <<'EOF'
import openpyxl
wb = openpyxl.Workbook()
ws = wb.active
ws.append(["Parameter Name", "Description"])
ws.append(["maxRetransmissions", "Max retrans count"])
ws.append(["NRCellDU", "NR cell DU MO"])
wb.save("/tmp/test_params.xlsx")
EOF

python scripts/ingest_excel.py \
  --file /tmp/test_params.xlsx \
  --columns "Parameter Name"
```

Expected output:
```
Extracted 2 terms from /tmp/test_params.xlsx
Merged into .../dictionary/acronyms.json
  + NRCellDU
  + maxRetransmissions
```

- [ ] **Step 3: Verify acronyms.json updated**

```bash
python -c "
import json
data = json.load(open('dictionary/acronyms.json'))
print('keywords section:', list(data.get('keywords', {}).keys())[:5])
"
```

Expected: shows extracted term names.

- [ ] **Step 4: Commit**

```bash
git add scripts/ingest_excel.py
git commit -m "feat(scripts): add ingest_excel CLI for Excel → acronyms.json term loading"
```

---

## Task 10: Integration — verify ingest pipeline uses term_tagger

The ingest pipeline (wherever `chunker.dispatch()` is called and chunks are sent to Milvus) must call `tag_chunk` after dispatch. Locate the ingest script that builds and upserts chunks.

- [ ] **Step 1: Find the ingest script**

```bash
find scripts/ -name "*.py" | xargs grep -l "dispatch\|upsert\|insert" 2>/dev/null
grep -r "dispatch\|chunker" scripts/ src/ --include="*.py" -l | grep -v __pycache__
```

- [ ] **Step 2: Add term_tagger call to ingest pipeline**

In the script/file that calls `chunker.dispatch()`, add:

```python
from spar.ingest.term_tagger import tag_chunk
from spar.preprocessing.abbrev_mapper import load_acronyms, load_keywords
from pathlib import Path

_ACRONYMS_PATH = Path("dictionary/acronyms.json")
acronyms = load_acronyms(_ACRONYMS_PATH)
keywords = load_keywords(acronyms)

# After chunker.dispatch():
chunks = chunker.dispatch(text, source_doc, doc_type=doc_type)
chunks = [tag_chunk(chunk, keywords) for chunk in chunks]
# then upsert chunks to Milvus
```

- [ ] **Step 3: Commit**

```bash
git add <modified ingest file>
git commit -m "feat(ingest): apply term_tagger to chunks before Milvus upsert"
```

---

## Self-Review Checklist

- [x] Spec goal 1 (Excel → acronyms.json keywords section): Task 4
- [x] Spec goal 2 (Tag Milvus chunks at ingest): Task 5 + Task 10
- [x] Spec goal 3 (Query-time term detection → expr filter): Task 3 + Task 6 + Task 8
- [x] Milvus schema max_length fix: Task 2
- [x] No regression when no Excel provided: `extract_terms` returns `[]`, `build_expr` unchanged
- [x] openpyxl dep added: Task 1
- [x] CLI entry point: Task 9
- [x] All tasks have complete code, no TBD
