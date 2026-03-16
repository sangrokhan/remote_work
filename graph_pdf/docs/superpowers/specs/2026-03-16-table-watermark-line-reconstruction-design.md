# Table Watermark Line Reconstruction Design

**Context**

`extractor.py` currently renders tables as markdown tables and preserves every cell newline as `<br>`. In practice this causes two failures:

1. Watermark text fragments can bleed into table cells during `pdfplumber` table extraction.
2. Wrapped text inside a cell is indistinguishable from intentional multi-line content.

**Goal**

Extract only real table content after watermark cleanup, then render table output in a row-structured text format where:

- cell content wrapped by layout is collapsed into one line
- intentional multi-line cell content remains multiple lines within the same row block
- row boundaries remain explicit

**Approach**

Add a table-cell normalization stage between `extract_tables()` and output rendering.

- Remove watermark artifacts from cell text using the same known watermark vocabulary plus fragment heuristics observed in current output (`FID`, `N`, `O`, `C` bleed cases).
- Reconstruct cell line groups from raw cell text:
  - keep bullet-like lines as separate logical lines
  - keep clearly separated lines as separate logical lines
  - collapse non-bullet wrapped lines into a single logical line
- Replace markdown-table rendering with a structured row-block renderer that prints each row with header-labeled fields.

**Output Shape**

Example:

```text
### Page 1 table 1
- Row 1
  Item: Laptop
  Qty: 12
  Price: $120
  Notes:
  - line 1
```

For rows with only one logical line per cell, values stay on one line. For cells with multiple logical lines, the first line appears inline when possible and subsequent logical lines are emitted below that field.

**Verification**

- add failing tests for watermark fragment cleanup in table cells
- add failing tests for wrapped cell text collapsing
- keep existing end-to-end verification for body text and image output
