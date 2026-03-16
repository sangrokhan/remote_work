# Table Watermark Line Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove watermark bleed from extracted table cells and render table rows in a structured text format that distinguishes wrapped text from intentional cell line breaks.

**Architecture:** Keep `pdfplumber` extraction as-is, then normalize each extracted cell before output rendering. Replace markdown-table output with row-structured text blocks so cell-internal logical lines can be represented without markdown table constraints.

**Tech Stack:** Python 3, `pdfplumber`, stdlib `unittest`

---

## Chunk 1: Tests

### Task 1: Add failing unit coverage for cell normalization and table rendering

**Files:**
- Create: `tests/test_extractor.py`
- Modify: `verify.py`

- [ ] **Step 1: Write the failing test**

Write tests covering:

- watermark fragments removed from cells
- wrapped non-bullet lines collapse to one line
- bullet lines remain separate logical lines
- rendered table output no longer contains markdown table syntax

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_extractor -v`
Expected: FAIL because the current implementation still preserves `<br>` markdown cells and has no structured renderer helpers.

- [ ] **Step 3: Update end-to-end verification expectations**

Adjust `verify.py` so it parses the new row-block table output and checks that known table rows still exist without watermark bleed.

- [ ] **Step 4: Run targeted tests again**

Run: `python3 -m unittest tests.test_extractor -v`
Expected: still FAIL until implementation lands.

## Chunk 2: Implementation

### Task 2: Normalize extracted table cells

**Files:**
- Modify: `extractor.py`
- Test: `tests/test_extractor.py`

- [ ] **Step 1: Implement cell cleanup helpers**

Add helpers that:

- remove watermark tokens and fragment artifacts
- split cell text into candidate lines
- merge wrapped prose lines while keeping bullets / explicit list lines separate

- [ ] **Step 2: Run unit tests**

Run: `python3 -m unittest tests.test_extractor -v`
Expected: failing rendering-related assertions may remain, normalization tests should improve.

### Task 3: Replace markdown table rendering with row-block output

**Files:**
- Modify: `extractor.py`
- Modify: `verify.py`
- Test: `tests/test_extractor.py`

- [ ] **Step 1: Implement structured renderer**

Render each table as:

- table heading
- repeated `- Row N` blocks
- header-labeled fields

- [ ] **Step 2: Run unit tests**

Run: `python3 -m unittest tests.test_extractor -v`
Expected: PASS

- [ ] **Step 3: Run end-to-end verification**

Run: `python3 verify.py`
Expected: PASS

### Task 4: Final verification

**Files:**
- Modify: `README.md` if output contract examples need refresh

- [ ] **Step 1: Run full verification**

Run: `python3 -m unittest tests.test_extractor -v`
Expected: PASS

Run: `python3 verify.py`
Expected: PASS

- [ ] **Step 2: Commit**

```bash
git add extractor.py verify.py tests/test_extractor.py docs/superpowers/specs/2026-03-16-table-watermark-line-reconstruction-design.md docs/superpowers/plans/2026-03-16-table-watermark-line-reconstruction.md
git commit -m "fix: clean table watermark bleed and restructure multiline cell output"
```
