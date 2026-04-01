# Vertical-Axis Cross-Page Table Merge Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cross-page table merge를 PRD 기준으로 단순화해, row 내용이 아니라 vertical axis overlap과 intervening region 유무만으로 continuation을 판단한다.

**Architecture:** `extractor/pipeline.py`의 cross-page anchor selection을 구조 기반 판정으로 재정의한다. 내용 기반 continuation heuristic은 제거하고, 기존 region gap 차단 로직과 geometry merge 판정만 남긴다.

**Tech Stack:** Python 3, unittest, pdfplumber-based extraction pipeline

---

### Task 1: 구조 기반 merge 회귀 테스트 추가

**Files:**
- Modify: `tests/test_pipeline.py`
- Reference: `extractor/pipeline.py`

- [ ] **Step 1: Write the failing tests**

추가할 테스트:
- row 내용이 달라도 axes overlap과 empty gap만 맞으면 anchor selection이 가능해야 함
- axes overlap이 없으면 merge 후보가 아니어야 함
- gap에 다른 영역이 있으면 merge 후보가 아니어야 함

- [ ] **Step 2: Run focused tests to verify red**

Run: `python3 -m unittest -v tests.test_pipeline`
Expected: 새 테스트가 현재 content-based heuristic 때문에 실패

- [ ] **Step 3: Keep test fixture scope minimal**

테스트 입력은 synthetic bbox/axes/region_map과 간단한 rows로 제한한다. PDF fixture는 사용하지 않는다.

- [ ] **Step 4: Commit test changes when green later**

```bash
git add tests/test_pipeline.py
git commit -m "test: cover vertical-axis cross-page table merge"
```

### Task 2: content-based continuation heuristic 제거

**Files:**
- Modify: `extractor/pipeline.py`
- Reference: `docs/superpowers/specs/2026-04-01-vertical-axis-cross-page-table-merge-design.md`

- [ ] **Step 1: Remove row-content-based checks from anchor selection**

수정 대상:
- `_pick_cross_page_anchor()`
- `_looks_like_cross_page_continuation_rows()` 호출 제거
- header signature / first row signature / family-type 특례에 의존한 merge 판단 제거

- [ ] **Step 2: Keep structure-only merge checks**

유지할 조건:
- x overlap 또는 vertical axis overlap
- `_table_shapes_compatible()`
- `_has_cross_page_gap_blocked()`
- `_continuation_regions_should_merge()`

- [ ] **Step 3: Remove dead helper logic if no longer used**

비사용 상태가 되면 정리 대상:
- `_looks_like_cross_page_continuation_rows()`
- `_table_header_signature()`
- `_first_table_row_signature()`

- [ ] **Step 4: Run focused tests to get green**

Run: `python3 -m unittest -v tests.test_pipeline`
Expected: 새 구조 기반 테스트 통과

- [ ] **Step 5: Commit implementation**

```bash
git add extractor/pipeline.py tests/test_pipeline.py
git commit -m "refactor: use structural cross-page table merge"
```

### Task 3: sample 회귀 검증

**Files:**
- Verify only: `tests/test_samples.py`

- [ ] **Step 1: Run targeted sample test**

Run: `python3 -m unittest -v tests.test_samples.SampleRawDumpTests.test_sample_raw_dumps_parse_via_from_raw`
Expected: raw sample current failure set 재평가

- [ ] **Step 2: Inspect raw-93-114 table count**

Run a focused debug command if needed to compare:
- `artifacts/debug-raw-93-114/md/FGR-BC0401_tables.md`
- `samples/gold/raw-93-114/md/FGR-BC0401_tables.md`

- [ ] **Step 3: Record residual failures**

남은 실패가 있으면 body markdown diff와 table mismatch를 분리해 기록한다.

- [ ] **Step 4: Final verification**

Run:
```bash
python3 -m unittest -v tests.test_pipeline.TableModuleTests
python3 -m unittest -v tests.test_samples.SampleRawDumpTests.test_sample_raw_dumps_parse_via_from_raw
```

Expected:
- pipeline tests green
- sample failures가 있으면 정확히 어떤 sample/영역인지 확인 가능
