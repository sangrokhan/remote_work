# Open-Sided Table Extractor Replacement PRD

> **제약:** 본 PRD 범위에서 휴리스틱한 접근은 사용할 수 없으며, 재현 가능한 규칙 기반/기하 기반 로직만 적용한다.

## Purpose

Replace `pdfplumber.extract_tables()` with an in-house extractor that we can control directly.

The replacement must support document tables whose left and right outer borders are open.

The new extractor should decide row and cell structure from explicit geometry that we own, rather than attempting to repair third-party table output after row splitting has already happened.

## Problem Statement

The current parser detects candidate table regions first, but delegates row and cell construction to `extract_tables()`.

That makes the most important structural decision outside our control.

When a logical cell is split into multiple extracted rows, later repair logic can only guess whether the split was caused by wrapping, narrow columns, or broken structural detection.

The replacement should move row and column formation into a deterministic extractor built on top of explicit black line geometry.

## Scope Summary

This PRD defines a three-stage replacement plan.

Stage 1 is limited to tables whose structure is visible from explicit black line segments.

Tables that do not expose enough black-line structure are outside the Stage 1 extraction contract.

Those failures must remain visible in debug output.

They must be excluded from Stage-1-based extractor outputs and comparison artifacts.

They do not affect production markdown table output until the Stage 3 cutover replaces the old path.

## Goals

- Remove all production dependence on `extract_tables()`.
- Build a page-local extractor from explicit black line geometry and line payload text assignment.
- Support left-open and right-open tables as normal inputs.
- Preserve deterministic intermediate geometry for debugging and downstream continuation logic.
- Keep merged-cell interpretation and cross-page continuation as later stages rather than forcing them into the first extractor milestone.

## Non-Goals

- Supporting every arbitrary table-like shape in Stage 1.
- Guessing missing columns from text alignment in Stage 1.
- Recovering tables that do not expose enough black horizontal structure in Stage 1.
- Solving merged-cell interpretation in Stage 1.
- Solving cross-page continuation in Stage 1.

## Stage 1: Black-Line Grid Reconstruction

### Objective

Build a page-local table extractor that reconstructs rows and columns only from explicit black horizontal and vertical line segments.

### Input Contract

- Reuse the current table-region selection result as the input `table bbox`.
- Use only explicit black line segments as internal table-structure signals.
- Ignore gray, blue, or otherwise non-black lines for Stage 1 grid reconstruction.
- Use `line payload` units for text assignment.
- Do not use text alignment to infer missing columns in Stage 1.

### Structural Rules

- Row separators are built from black horizontal line `y` positions.
- Column separators are built from black vertical line `x` positions.
- Left and right outer table bounds come from the selected `table bbox`.
- The selected `table bbox` left and right edges are inherited outer bounds, not inferred structural signals.
- Column count is not fixed and must follow the detected vertical separators.
- If no internal vertical separators exist, the table is treated as a row-only table.
- If black horizontal separators are insufficient to form stable row bands, extraction fails for that region.

### Output Contract

Stage 1 must emit:

- `bbox`
- `row_line_positions`
- `column_line_positions`
- `row_bands`
- `column_bands`
- `cell_boxes`
- `rows`
- debug payload for intermediate geometry and text assignment

For row-only tables:

- `column_line_positions` must be empty
- `column_bands` must contain exactly one full-width band derived from `table bbox`
- `cell_boxes` must contain one full-width cell per row band
- `rows` must contain one cell per row

### Failure Handling

If Stage 1 cannot form a valid grid from black-line geometry:

- keep the region and failure reason in debug output
- exclude the region from Stage-1-based extractor outputs and comparison artifacts

Stage 1 does not fall back to alternative row reconstruction heuristics.

### Acceptance Criteria

- A new page-local extractor path exists and can run without calling `extract_tables()`.
- Clear document tables with visible black-line structure are reconstructed from explicit row and column positions.
- Open-sided tables use `table bbox` left and right edges as outer bounds without requiring left or right vertical borders.
- Tables with no internal vertical separators are handled as row-only tables.
- Regions without enough black horizontal structure fail deterministically and surface only in debug output.

## Stage 2: Merge Interpretation and Structural Recovery

### Objective

Interpret missing boundaries and merged-cell-like structures on top of Stage 1 output.

### Scope

- colspan-like cases caused by missing vertical separators within an already-formed Stage 1 grid
- rowspan-like cases caused by partial missing horizontal separators after Stage 1 has already formed stable row bands
- structural normalization for layouts currently repaired by row-fragment heuristics
- recovery logic built on Stage 1 geometry rather than on already-damaged row output

Stage 2 does not recover regions that Stage 1 rejected for lacking enough horizontal structure to form row bands.

### Acceptance Criteria

- Merge interpretation consumes Stage 1 bands and cell geometry rather than re-deriving structure from text only.
- Targeted merge-recovery behaviors are covered by focused unit tests and agreed fixture diffs.
- The system can distinguish between same-cell wrapping, true new rows, and missing-boundary merge cases.

## Stage 3: Cross-Page Continuation and Pipeline Replacement

### Objective

Reconnect the new extractor to document-level continuation logic and fully remove production dependence on `extract_tables()`.

### Scope

- repeated-header handling
- continuation candidate selection
- pending-table merge across pages
- final markdown output path cutover
- removal of all remaining production `extract_tables()` usage

### Acceptance Criteria

- Cross-page continuation works from the new extractor output.
- Repeated-header stripping is verified by focused tests and agreed fixture outputs.
- Final markdown table output is produced only from the new extractor path.
- No production `extract_tables()` usage remains.

## Testing Strategy

### Stage 1 Unit Tests

- black-line filtering
- row-line normalization
- column-line normalization
- open-sided table band construction
- row-only table handling
- line-payload-to-cell assignment
- extraction failure signaling

### Stage 2 Unit Tests

- merged-cell interpretation

### Stage 3 Unit Tests

- cross-page continuation compatibility
- repeated-header stripping on the new extractor output

### Fixture Tests

Use the existing raw dumps as required regression fixtures:

- `samples/raw-32-37.dump`
- `samples/raw-38-50.dump`
- `samples/raw-93-114.dump`

### Debug Validation

For every extracted or failed region, debug output should explain:

- which black lines were accepted
- which row and column positions were formed
- how row bands and column bands were created
- how each line payload was assigned
- why extraction failed when it did

## Implementation Planning Notes

The implementation plan that follows this PRD should keep Stage 1, Stage 2, and Stage 3 separate.

Stage 1 should prioritize deterministic geometry reconstruction and failure visibility.

Stage 2 should add explicit missing-boundary interpretation without broadening Stage 1 signals.

Stage 3 should integrate continuation logic only after Stage 1 and Stage 2 outputs are stable on the required sample fixtures.

Production markdown table output remains on the old path until Stage 3 performs the cutover.
