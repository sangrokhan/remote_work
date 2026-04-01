# Thin Black Rect Table Geometry Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild table region and row/column geometry from visible thin black separator rects using a fixed thickness threshold of `0.5`, eliminating PRD-unbacked edge heuristics.

**Architecture:** Stop treating pdfplumber `rect_edge` output as if it were a visible border. Instead, inspect original `page.rects` and use only black `fill` rects whose thin axis is `<= 0.5` as separators. Build table regions and row/column boundaries directly from those separator rects, then keep the extracted grid structure without post-hoc structural column collapse.

**Tech Stack:** Python, pdfplumber, unittest

---

## File Map

- Modify: `extractor/tables.py`
  - Replace edge-driven table-region admission with thin-black-rect-driven region discovery.
  - Add separator-rect helpers for horizontal and vertical geometry extraction.
  - Remove remaining x-position merge heuristics from column boundary reconstruction.
  - Stop post-hoc structural column collapse on black-line-grid output.
- Modify: `tests/test_tables.py`
  - Add focused failing tests for separator rect classification.
  - Add failing tests for header-only table chunk geometry using thin black rects.
  - Add failing tests that prove page header/footer bars do not become table separators when threshold is `0.5`.
- Modify: `tests/test_samples.py`
  - Keep the existing `raw-93-114` table 19 regression.
  - Add or tighten sample assertions only after geometry path is stable.

## Chunk 1: Separator Rect Classification

### Task 1: Add failing tests for `0.5` threshold behavior

**Files:**
- Modify: `tests/test_tables.py`

- [ ] **Step 1: Write the failing tests**

Add tests that assert:
- a black filled rect with `width=0.48`, `height=16.08` is a vertical separator
- a black filled rect with `height=0.48`, `width=141.50` is a horizontal separator
- a black filled rect with `height=0.96`, `width=454.27` is **not** a separator
- a blue fill rect with the same dimensions is **not** a separator

- [ ] **Step 2: Run the focused tests to confirm failure**

Run: `python3 -m unittest -v tests.test_tables.TableModuleTests`

Expected: FAIL on new separator-rect tests

- [ ] **Step 3: Implement minimal separator helpers**

In `extractor/tables.py`:
- add `_is_black_fill_rect(...)`
- add `_is_vertical_separator_rect(...)`
- add `_is_horizontal_separator_rect(...)`

Rules:
- black comes from `non_stroking_color`
- `fill` must be true
- one axis must be `<= 0.5`
- do **not** add a second-axis minimum rule here

- [ ] **Step 4: Re-run focused tests**

Run: `python3 -m unittest -v tests.test_tables.TableModuleTests`

Expected: new separator tests pass

## Chunk 2: Table Region Discovery From Separator Rects

### Task 2: Replace edge-component region detection

**Files:**
- Modify: `extractor/tables.py`
- Test: `tests/test_tables.py`

- [ ] **Step 1: Write failing tests for table-region discovery**

Add tests that build a synthetic page with:
- two thin black horizontal fill rects
- one or more thin black vertical fill rects between them

Assert:
- `_table_regions(...)` returns one region
- a header/footer full-width `0.96` black bar is ignored
- a blue header background rect does not create a region by itself

- [ ] **Step 2: Run the focused region tests**

Run: `python3 -m unittest -v tests.test_tables.TableModuleTests.test_table_regions_accept_two_horizontal_lines_with_vertical_connections`

Expected: fail until `_table_regions(...)` is rewritten

- [ ] **Step 3: Rewrite `_table_regions(...)`**

Implementation rules:
- derive horizontal candidates from thin black horizontal fill rects
- derive vertical candidates from thin black vertical fill rects
- body bounds and `excluded_bboxes` filtering still apply
- connect horizontal separators into one table region only when a vertical separator intersects them
- compute region `x0/x1` from actual separator rect extents, not `rect_edge`

- [ ] **Step 4: Re-run focused table-region tests**

Run: `python3 -m unittest -v tests.test_tables.TableModuleTests`

Expected: synthetic region tests pass

## Chunk 3: Row/Column Boundary Reconstruction

### Task 3: Rebuild row/column bands from separator rects directly

**Files:**
- Modify: `extractor/tables.py`
- Test: `tests/test_tables.py`

- [ ] **Step 1: Write failing tests for header-only chunk geometry**

Create a synthetic crop representing:
- 2 horizontal thin black rects
- 2 vertical thin black rects
- 3 header labels

Assert:
- row count is `1`
- column count is `3`
- no spacer columns are produced

- [ ] **Step 2: Run the focused geometry test**

Run: `python3 -m unittest -v tests.test_tables.TableModuleTests.test_build_column_bands_collapses_nearby_visible_vertical_edges_into_single_boundary`

Expected: fail until column reconstruction stops using `rect_edge` x heuristics

- [ ] **Step 3: Replace `_build_row_bands(...)` and `_build_column_bands(...)` inputs**

Implementation rules:
- use separator rects directly, not `page.horizontal_edges` / `page.vertical_edges`
- row boundaries come from thin black horizontal rect positions
- column boundaries come from thin black vertical rect positions
- remove `_merge_numeric_positions(..., tolerance=...)` dependence for this path
- if duplicate positions remain, dedupe only exact or near-exact floating noise from the same rect source, not semantic grouping

- [ ] **Step 4: Re-run focused geometry tests**

Run: `python3 -m unittest -v tests.test_tables.TableModuleTests`

Expected: header-only chunk reconstructs as `3` columns, not `5` or `9`

## Chunk 4: Remove Post-hoc Structural Rewrites

### Task 4: Keep black-geometry output intact

**Files:**
- Modify: `extractor/tables.py`
- Test: `tests/test_samples.py`

- [ ] **Step 1: Add failing regression for `raw-93-114` table 19**

Use or keep:
- `tests.test_samples.SampleRawDumpTests.test_raw_93_114_table_19_matches_golden_when_header_starts_on_previous_page`

- [ ] **Step 2: Remove structural post-processing from output path**

In `extractor/tables.py`:
- stop calling `_collapse_structural_triplet_columns(...)` in `_append_output_table(...)`
- if another path still depends on it, gate it off for black-geometry tables rather than keeping it globally active

- [ ] **Step 3: Re-run the targeted sample test**

Run: `python3 -m unittest -v tests.test_samples.SampleRawDumpTests.test_raw_93_114_table_19_matches_golden_when_header_starts_on_previous_page`

Expected: table 19 exact match passes

## Chunk 5: Sample Verification

### Task 5: Verify sample table output end to end

**Files:**
- Modify: `tests/test_samples.py` only if assertions need cleanup after geometry rewrite

- [ ] **Step 1: Run the table-focused sample checks**

Run:
- `python3 -m unittest -v tests.test_samples.SampleRawDumpTests.test_raw_93_114_table_count_matches_golden_table_file`
- `python3 -m unittest -v tests.test_samples.SampleRawDumpTests.test_raw_93_114_table_19_matches_golden_when_header_starts_on_previous_page`

- [ ] **Step 2: Re-run full sample snapshot test**

Run: `python3 -m unittest -v tests.test_samples.SampleRawDumpTests.test_sample_raw_dumps_parse_via_from_raw`

Expected:
- table markdown should not regress on `raw-32-37` or `raw-38-50`
- if failures remain, they should be body-markdown-only and treated separately

- [ ] **Step 3: Run relevant table unit tests**

Run: `python3 -m unittest -v tests.test_tables`

- [ ] **Step 4: Commit**

```bash
git add extractor/tables.py tests/test_tables.py tests/test_samples.py docs/superpowers/plans/2026-04-01-thin-black-rect-table-geometry.md
git commit -m "fix: derive table geometry from thin black separator rects"
```

