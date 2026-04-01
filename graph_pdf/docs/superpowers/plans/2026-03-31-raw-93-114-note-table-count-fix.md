# Raw-93-114 Note/Table Count Fix Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent note regions inside `raw-93-114` from being extracted as table regions so the golden table count remains 21.

**Architecture:** Keep note extraction and table extraction separate at the geometry stage. Pass detected note bounding boxes into table-region discovery so note-owned edges are excluded before table crops are built, then keep the existing bbox-level conflict filter as a secondary safety check.

**Tech Stack:** Python, pdfplumber, unittest

---

### Task 1: Reproduce the note/table overlap failure in a focused unit test

**Files:**
- Modify: `tests/test_tables.py`
- Reference: `extractor/tables.py`

- [ ] **Step 1: Write the failing test**

Add a unit test that builds a synthetic page where:
- a large black table region spans the page
- a smaller note-like region sits inside it
- passing the note bbox into table extraction should suppress the note-only region or split it out of the table region result

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `python3 -m unittest -v tests.test_tables.TableModuleTests.test_extract_tables_excludes_note_owned_regions`

Expected: FAIL because `_extract_tables(...)` does not yet accept or honor excluded note regions.

### Task 2: Exclude note-owned geometry during table region discovery

**Files:**
- Modify: `extractor/tables.py`
- Reference: `extractor/pipeline.py`

- [ ] **Step 1: Add the minimal API surface**

Extend `_table_regions(...)` and `_extract_tables(...)` to accept `excluded_bboxes`.

- [ ] **Step 2: Implement the minimal geometry filter**

Filter out horizontal and vertical edges that significantly overlap any excluded bbox before connected-component table grouping starts.

- [ ] **Step 3: Keep behavior narrow**

Do not change table normalization or note extraction logic unless the new test proves the geometry filter is insufficient.

- [ ] **Step 4: Run the targeted test**

Run: `python3 -m unittest -v tests.test_tables.TableModuleTests.test_extract_tables_excludes_note_owned_regions`

Expected: PASS

### Task 3: Wire pipeline note detection into table extraction

**Files:**
- Modify: `extractor/pipeline.py`
- Reference: `extractor/notes.py`

- [ ] **Step 1: Pass detected note bboxes into `_extract_tables(...)`**

Use the already collected `note_bboxes` from pipeline note detection as `excluded_bboxes` for table extraction.

- [ ] **Step 2: Preserve the existing bbox conflict filter**

Keep `_table_bbox_conflicts_with_note_region(...)` in place as a secondary guard for note-only tables that still survive extraction.

- [ ] **Step 3: Run raw-93-114 focused verification**

Run a local extraction for `samples/raw-93-114.dump` and verify generated `table_count` becomes 21.

### Task 4: Verify regression coverage and summarize residual diffs

**Files:**
- Verify: `tests/test_samples.py`
- Verify: `samples/gold/raw-93-114/md/FGR-BC0401_tables.md`

- [ ] **Step 1: Run focused regression checks**

Run:
- `python3 -m unittest -v tests.test_tables.TableModuleTests.test_extract_tables_excludes_note_owned_regions`
- `python3 -m unittest -v tests.test_samples.SampleRawDumpTests.test_gold_summaries_match_table_markdown_counts`

- [ ] **Step 2: Run the sample snapshot test if practical**

Run: `python3 -m unittest -v tests.test_samples.SampleRawDumpTests.test_sample_raw_dumps_parse_via_from_raw`

Expected: table-count mismatch for `raw-93-114` should be resolved; any remaining failures should be summarized separately as pre-existing content diffs.
