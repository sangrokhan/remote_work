# Typography-Aware Body Label Separation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent standalone body labels such as `Configuration Parameters`, `Feature Scope`, `Non Stand Alone`, and `Activation Procedure` from being merged into the following narrative sentence by using typography-aware detection.

**Architecture:** Keep the fix narrow and data-driven. First, codify the failing label-line cases with regression tests. Then add a lightweight label classifier in `extractor/text.py` that uses observed typography signals, especially font family and color, to mark these lines as standalone body labels instead of ordinary paragraph lines. Finally, verify on the three failing sample dumps and ensure the rule does not over-promote normal body text or repeated heading lines.

**Tech Stack:** Python, `unittest`, `pdfplumber`, typography payloads from `extractor/text.py`

---

## Investigation Findings

### What is happening now

- `_line_kind()` in `extractor/text.py` only distinguishes `heading` vs `paragraph`.
- If `heading_levels` does not map the line font size, the line becomes `paragraph`.
- `_should_merge_paragraph_lines()` then merges adjacent paragraph lines by gap only.
- `_join_non_heading_block_lines()` flattens the whole paragraph block into a single line.

### Why the problematic lines get merged

The failing label lines are **not larger than body text**:
- label line size: `11.04`
- following body line size: `11.04`

So **font size alone is not enough**.

### Strong signals found in the raw payloads

For merged label lines in `raw-32-37`, `raw-38-50`, and `raw-93-114`:

- standalone label line:
  - `fontname`: `ABCDEE+Franklin Gothic Medium`
  - `color`: `(0.047, 0.302, 0.635)` blue
  - `size`: `11.04`
  - `x0`: `157.1`
  - short text, usually 1 to 3 words, sometimes 7 words
- following narrative line:
  - `fontname`: `Times New Roman`
  - `color`: `(0.0,)` black
  - `size`: `11.04`
  - same `x0`: `157.1`

Examples confirmed from raw dumps:
- `Configuration Parameters`
- `Activation/Deactivation Parameters`
- `Feature Scope`
- `Non Stand Alone`
- `Stand Alone`
- `Preconditions`
- `Activation Procedure`
- `Deactivation Procedure`
- `RLC parameter adaptation for VoNR PUSCH repetition`

### Additional typography pattern

- major section headings such as `FEATURE DESCRIPTION`, `SYSTEM OPERATION`, `REFERENCE` also use blue `Franklin Gothic Medium`, but at `15.96`.
- this suggests:
  - `15.96` blue Franklin lines are already section headings
  - `11.04` blue Franklin lines behave like **standalone body labels / sublabels**

## File Map

- Modify: `extractor/text.py`
  Responsibility: introduce typography-aware standalone label detection and stop block flattening from collapsing those labels with following narrative text.
- Modify: `tests/test_text.py`
  Responsibility: add unit tests for label detection and paragraph-merge prevention.
- Modify: `tests/test_samples.py`
  Responsibility: verify sample-level body markdown behavior once the extractor logic changes.

## Chunk 1: Encode the Failing Cases

### Task 1: Add focused text-module regressions

**Files:**
- Modify: `tests/test_text.py`
- Read: `extractor/text.py`

- [ ] **Step 1: Write a failing classifier test**

Add a test for a line payload representing:
- `text`: `Configuration Parameters`
- `fontname`: `ABCDEE+Franklin Gothic Medium`
- `color`: `(0.047, 0.302, 0.635)`
- `size`: `11.04`

Assert that the new detector treats it as a standalone label line.

- [ ] **Step 2: Write a failing merge test**

Create two synthetic adjacent lines:
- blue Franklin label line
- black Times New Roman narrative line

Assert they do **not** merge into one paragraph block.

- [ ] **Step 3: Add a negative test**

Use two ordinary `Times New Roman` body lines at `11.04` with black color.

Assert they still merge when gap and indent indicate normal wrapped paragraph flow.

- [ ] **Step 4: Run tests to verify failure**

Run:
- `python3 -m unittest -v tests.test_text.TextModuleTests.<new_label_detector_test>`
- `python3 -m unittest -v tests.test_text.TextModuleTests.<new_label_merge_test>`

Expected: FAIL before implementation.

- [ ] **Step 5: Commit**

```bash
git add tests/test_text.py
git commit -m "test: cover typography-aware standalone body labels"
```

## Chunk 2: Add Typography-Aware Label Detection

### Task 2: Implement standalone label classification in `extractor/text.py`

**Files:**
- Modify: `extractor/text.py`
- Test: `tests/test_text.py`

- [ ] **Step 1: Add a focused helper**

Introduce a helper such as:
- `_looks_like_standalone_body_label(line: dict) -> bool`

Expected evidence-based rule:
- font family contains `Franklin Gothic Medium`
- dominant color is the observed blue `(0.047, 0.302, 0.635)` or within a small tolerance
- size is near body size `11.04`
- line is short enough to be a label, not a full sentence
- line text does not end with sentence punctuation

- [ ] **Step 2: Decide how to integrate the rule**

Preferred minimal approaches:
- treat these lines as a distinct `line_kind`, such as `body_label`
- or keep `paragraph` but prevent `_should_merge_paragraph_lines()` from merging when the previous line matches the label detector

Prefer the smaller change that preserves current behavior elsewhere.

- [ ] **Step 3: Keep existing heading behavior intact**

Do not replace `heading_levels`.
This change should only catch the `11.04` blue Franklin label lines that currently fall through as paragraphs.

- [ ] **Step 4: Run focused text tests**

Run:
- `python3 -m unittest -v tests.test_text.TextModuleTests`

Expected:
- new label tests pass
- wrapped body paragraph behavior still passes

- [ ] **Step 5: Commit**

```bash
git add extractor/text.py tests/test_text.py
git commit -m "fix: classify standalone blue body labels"
```

## Chunk 3: Validate on Real Samples

### Task 3: Re-run the real failing samples

**Files:**
- Test: `tests/test_samples.py`
- Inspect if needed: `samples/gold/raw-32-37/md/FGR-BC0008.md`
- Inspect if needed: `samples/gold/raw-38-50/md/FGR-BC0201.md`
- Inspect if needed: `samples/gold/raw-93-114/md/FGR-BC0401.md`

- [ ] **Step 1: Run the sample snapshot test**

Run:
`python3 -m unittest -v tests.test_samples.SampleRawDumpTests.test_sample_raw_dumps_parse_via_from_raw`

Focus on whether these lines stay separate:
- `Configuration Parameters`
- `Activation/Deactivation Parameters`
- `Feature Scope`
- `Non Stand Alone`
- `Stand Alone`
- `Preconditions`
- `Activation Procedure`
- `Deactivation Procedure`
- `RLC parameter adaptation for VoNR PUSCH repetition`

- [ ] **Step 2: Inspect remaining diffs**

If failures remain, classify them:
- same typography-aware label issue
- different body merge pattern
- fixture mismatch

Do not broaden rules until the remaining failure pattern is identified.

- [ ] **Step 3: Run supporting checks**

Run:
- `python3 -m unittest -v tests.test_pipeline.PipelineExtractionTests.test_pipeline_keeps_table_adjacent_body_text_outside_table_bbox`
- `python3 -m unittest -v tests.test_tables.TableModuleTests`

- [ ] **Step 4: Final commit**

```bash
git add extractor/text.py tests/test_text.py tests/test_samples.py
git commit -m "fix: preserve standalone body labels in sample markdown"
```

## Proposed Detection Method

When a line is adjacent to another body line and all of the following hold, treat it as a standalone label rather than a mergeable paragraph line:

1. `fontname` contains `Franklin Gothic Medium`
2. `color` is the observed blue used for body labels
3. `size` is around `11.04`, which is body-sized but not section-heading-sized
4. text is short and label-like
5. text does not end in sentence punctuation
6. next line is ordinary black `Times New Roman` body text at the same indent

## Success Criteria

- font size alone is no longer used as the only signal for these label lines
- blue Franklin `11.04` standalone labels remain on their own lines
- ordinary wrapped black body paragraphs still merge normally
- the three sample dumps no longer fail because of label-line flattening, or any remaining failures are clearly from a different pattern
