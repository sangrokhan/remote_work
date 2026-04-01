# Preserve Pre-Table Body Text Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve section text that appears immediately before tables so sample body markdown no longer drops or truncates `Parameter Descriptions of ...`, section labels, or standalone headings.

**Architecture:** Split the fix into two independent mechanisms. First, tighten ownership rules for table-adjacent body lines so only true orphan table headers are excluded from body extraction. Second, prevent body block flattening from merging standalone section labels and heading-like lines into the following sentence when they should remain separate lines.

**Tech Stack:** Python, `unittest`, `pdfplumber`, existing extractor heuristics in `extractor/pipeline.py`, `extractor/tables.py`, and `extractor/text.py`

---

## File Map

- Modify: `extractor/tables.py`
  Responsibility: refine orphan table header detection and table-owned body line selection.
- Modify: `extractor/pipeline.py`
  Responsibility: limit body-line exclusion and post-filtering to true table-owned lines only.
- Modify: `extractor/text.py`
  Responsibility: keep heading-like or label-like lines from being flattened into adjacent paragraph text.
- Modify: `tests/test_pipeline.py`
  Responsibility: add focused regression tests for pre-table text preservation and heading/paragraph separation.
- Modify: `tests/test_samples.py`
  Responsibility: keep sample-level verification aligned with `samples/gold`.

## Chunk 1: Reproduce and Isolate the Two Failure Modes

### Task 1: Add a focused regression for lost pre-table descriptor lines

**Files:**
- Modify: `tests/test_pipeline.py`
- Read: `extractor/pipeline.py`
- Read: `extractor/tables.py`

- [ ] **Step 1: Write the failing test**

Create a unit test that simulates:
- body text line: `Parameter Descriptions of dscp-bypass-function`
- table immediately below it
- no other confounding content

Assert:
- the descriptor line remains in body markdown
- the table still renders in `table_markdown`
- the descriptor line is not duplicated into the table block

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest -v tests.test_pipeline.PipelineExtractionTests.<new_test_name>`
Expected: FAIL because the descriptor line is either excluded from body extraction or reduced to a suffix like `entries`.

- [ ] **Step 3: Add a second focused regression for heading/paragraph flattening**

Create a unit test that simulates:
- standalone line: `Configuration Parameters`
- following sentence: `To configure the feature settings, run the associated commands and set the key parameters.`

Assert:
- output keeps them on separate lines in body markdown
- no flattening into `Configuration Parameters To configure ...`

- [ ] **Step 4: Run the second test to verify it fails**

Run: `python3 -m unittest -v tests.test_pipeline.PipelineExtractionTests.<new_heading_merge_test_name>`
Expected: FAIL because `_join_non_heading_block_lines()` currently flattens both lines into one paragraph block.

- [ ] **Step 5: Commit the failing tests**

```bash
git add tests/test_pipeline.py
git commit -m "test: cover pre-table body text preservation"
```

## Chunk 2: Fix Over-Aggressive Table-Owned Line Detection

### Task 2: Narrow orphan table header detection in `extractor/tables.py`

**Files:**
- Modify: `extractor/tables.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Review current ownership heuristic**

Inspect:
- `_looks_like_orphan_table_header_line()`
- `_table_owned_body_lines()`

Current risk points:
- substring term matching treats `Descriptions` as `description`
- paths like `.../function/...` introduce extra header terms
- vertical proximity alone is enough to mark a line as table-owned

- [ ] **Step 2: Implement the minimal heuristic change**

Expected direction:
- require stronger evidence than “contains two header terms”
- reject lines with narrative phrases such as `Parameter Descriptions of`
- prefer exact header-shape signals such as:
  - short noun-only labels
  - no prepositions like `of`
  - high similarity to actual table header rows

Implementation should stay local to `extractor/tables.py`.

- [ ] **Step 3: Run focused tests**

Run:
- `python3 -m unittest -v tests.test_pipeline.PipelineExtractionTests.<new_test_name>`
- `python3 -m unittest -v tests.test_pipeline.PipelineExtractionTests.test_pipeline_excludes_table_owned_orphan_header_lines_from_body_text`

Expected:
- new descriptor preservation test passes
- existing orphan-header ownership test still passes

- [ ] **Step 4: Commit**

```bash
git add extractor/tables.py tests/test_pipeline.py
git commit -m "fix: narrow table-owned orphan header detection"
```

## Chunk 3: Fix Heading/Label Flattening in Body Text

### Task 3: Preserve standalone labels before explanatory paragraphs

**Files:**
- Modify: `extractor/text.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Review block merge and join flow**

Inspect:
- `_should_merge_paragraph_lines()`
- `_build_body_blocks()`
- `_join_non_heading_block_lines()`

Observed failure:
- lines like `Configuration Parameters`, `Feature Scope`, `Preconditions`, `Activation Procedure`, `Non Stand Alone`, `Stand Alone` are treated as ordinary paragraph lines and flattened with the next sentence.

- [ ] **Step 2: Implement the minimal separation rule**

Expected direction:
- detect short label-like standalone lines
- keep them as separate output lines even when font/spacing makes them look paragraph-adjacent

Possible implementation approaches:
- introduce `_looks_like_standalone_label_line(text)`
- prevent paragraph merge when previous line is a label-like line
- or prevent block join from flattening a leading label line with following narrative text

Prefer the smallest change that preserves current body formatting elsewhere.

- [ ] **Step 3: Run focused tests**

Run:
- `python3 -m unittest -v tests.test_pipeline.PipelineExtractionTests.<new_heading_merge_test_name>`
- any existing `tests/test_pipeline.py` body-text formatting tests affected by this change

Expected: label line remains separate from following narrative sentence.

- [ ] **Step 4: Commit**

```bash
git add extractor/text.py tests/test_pipeline.py
git commit -m "fix: preserve standalone labels before tables"
```

## Chunk 4: Validate on Real Samples

### Task 4: Re-run sample-level verification

**Files:**
- Modify if needed: `samples/gold/raw-32-37/md/FGR-BC0008.md`
- Modify if needed: `samples/gold/raw-38-50/md/FGR-BC0201.md`
- Modify if needed: `samples/gold/raw-93-114/md/FGR-BC0401.md`
- Test: `tests/test_samples.py`

- [ ] **Step 1: Run the failing sample snapshot test**

Run:
`python3 -m unittest -v tests.test_samples.SampleRawDumpTests.test_sample_raw_dumps_parse_via_from_raw`

Expected:
- `raw-32-37`, `raw-38-50`, `raw-93-114` body markdown diffs shrink or disappear.

- [ ] **Step 2: Inspect remaining diffs**

If failures remain, classify them:
- still-lost pre-table text
- heading/paragraph flattening
- unrelated body text formatting

Do not broaden the fix until the remaining diff type is identified.

- [ ] **Step 3: Update golden only if the extractor output is demonstrably correct**

If sample failures are due to stale fixture expectations rather than extraction bugs:
- update only the affected `samples/gold/.../*.md` files
- rerun the same test immediately after the fixture change

- [ ] **Step 4: Run supporting verification**

Run:
- `python3 -m unittest -v tests.test_samples.SampleRawDumpTests.test_gold_summaries_match_table_markdown_counts`
- `python3 -m unittest -v tests.test_samples.SampleRawDumpTests.test_raw_93_114_table_count_matches_golden_table_file`

- [ ] **Step 5: Final commit**

```bash
git add extractor/tables.py extractor/text.py extractor/pipeline.py tests/test_pipeline.py tests/test_samples.py samples/gold
git commit -m "fix: preserve pre-table body text in sample markdown"
```

## Root Cause Summary

- `extractor/tables.py::_looks_like_orphan_table_header_line()` is broad enough to classify `Parameter Descriptions of ...` lines as table-owned because:
  - it uses substring term checks
  - `description` matches inside `descriptions`
  - path strings contribute words like `function`
- `extractor/pipeline.py` then removes those lines twice:
  - geometrically through `page_excluded_bboxes`
  - textually through `table_owned_texts` filtering
- `extractor/text.py::_join_non_heading_block_lines()` intentionally flattens paragraph blocks, which merges label-like standalone lines with the following sentence when they are not classified as headings.

## Success Criteria

- `Parameter Descriptions of ...` lines remain in body markdown immediately before their referenced table.
- standalone labels such as `Configuration Parameters`, `Feature Scope`, `Preconditions`, `Activation Procedure`, `Non Stand Alone`, and `Stand Alone` remain on their own lines.
- existing orphan-header exclusion still removes true stray table headers.
- `python3 -m unittest -v tests.test_samples.SampleRawDumpTests.test_sample_raw_dumps_parse_via_from_raw` passes, or any remaining failures are clearly outside this scope.
