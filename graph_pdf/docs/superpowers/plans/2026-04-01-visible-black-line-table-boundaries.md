# Visible Black Line Table Boundaries Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 표 경계 추출을 “눈에 보이는 검은 선” 기준으로 재정의해, `linewidth=0` hairline/`rect_edge` 경계 때문에 생기는 가짜 column split을 제거한다.

**Architecture:** `extractor/tables.py`의 black-line 기반 grid 복원에서 edge object 종류가 아니라 `visible black line` 속성으로 경계 후보를 필터링한다. 구체적으로는 `stroking_color`가 검정 계열이고 `linewidth > 0`인 선만 row/column boundary 후보로 사용한다. 그 위에서 기존 row/column band, payload grid, table text 변환을 그대로 유지한다.

**Tech Stack:** Python 3, pdfplumber edge extraction, unittest

---

## Debug Facts

- `raw-93-114`의 문제 표에서 `Family Display Name`이 `Family Display | Name`으로 갈라진다.
- 문제 표 구간의 세로 경계 후보는 주로 `rect_edge`, `stroking_color=0`, `linewidth=0`이다.
- 이 edge들은 헤더뿐 아니라 데이터 row 구간까지 같은 x 좌표로 반복된다.
- 현재 로직은 이런 `linewidth=0` edge도 내부 column boundary로 사용한다.
- 사용자가 원하는 기준은 “렌더러에서 아주 얇게 보이는 hairline은 고려하지 않고, 눈에 보이는 검은 선만 쓴다”이다.

---

### Task 1: visible-line 기준 테스트 추가

**Files:**
- Modify: `tests/test_tables.py`
- Reference: `extractor/tables.py`

- [ ] **Step 1: Write the failing tests**

추가할 테스트:
- `linewidth=0`, `stroking_color=0`인 `rect_edge`는 black line segment로 취급하지 않아야 함
- `linewidth>0`, `stroking_color=0`인 선은 black line segment로 취급해야 함
- `linewidth=0` 내부 vertical edge만 있는 synthetic table은 추가 column split을 만들지 않아야 함

- [ ] **Step 2: Run focused tests to verify red**

Run:
```bash
python3 -m unittest -v tests.test_tables.TableModuleTests
```

Expected:
- 새 테스트가 현재 구현에서 실패

- [ ] **Step 3: Keep fixture scope minimal**

Synthetic `SimpleNamespace` page와 edge dict만 사용한다. 실제 PDF fixture는 사용하지 않는다.

### Task 2: black line 필터를 visible line 기준으로 수정

**Files:**
- Modify: `extractor/tables.py`

- [ ] **Step 1: Update black line predicate**

수정 대상:
- `_is_black_line_segment()`

변경 내용:
- `stroking_color`가 검정 계열인지 확인
- `linewidth`를 읽어서 `> 0`인 경우만 True
- `linewidth=0`은 object type과 무관하게 False
- `non_stroking_color`는 column/row boundary 판정에 사용하지 않음

- [ ] **Step 2: Keep downstream band logic unchanged**

유지할 것:
- `_extract_black_lines_for_table()`
- `_build_row_bands()`
- `_build_column_bands()`
- `_build_payload_grid()`

이 단계에서는 후처리 heuristic 추가 금지. 필터 기준만 바꾼다.

- [ ] **Step 3: Run focused tests to get green**

Run:
```bash
python3 -m unittest -v tests.test_tables.TableModuleTests
```

Expected:
- visible-line 기준 테스트 통과

### Task 3: sample 영향 검증

**Files:**
- Verify only: `tests/test_samples.py`

- [ ] **Step 1: Re-run raw sample snapshot test**

Run:
```bash
python3 -m unittest -v tests.test_samples.SampleRawDumpTests.test_sample_raw_dumps_parse_via_from_raw
```

- [ ] **Step 2: Inspect table markdown exact match**

특히 확인할 것:
- `raw-32-37` table markdown exact match 유지
- `raw-38-50` table markdown exact match 유지
- `raw-93-114`에서 `Table 8`, `9`, `11`, `13`, `20`의 `Family Display | Name` 분리가 사라졌는지

- [ ] **Step 3: Record residual failures**

만약 남는 실패가 있으면 다음 두 범주로 분리 기록:
- table boundary extraction failure
- body markdown formatting failure

### Task 4: final verification

**Files:**
- Verify only

- [ ] **Step 1: Run final checks**

Run:
```bash
python3 -m unittest -v tests.test_tables.TableModuleTests
python3 -m unittest -v tests.test_pipeline
python3 -m unittest -v tests.test_samples.SampleRawDumpTests.test_sample_raw_dumps_parse_via_from_raw
```

- [ ] **Step 2: Summarize outcome**

최종 보고에는 반드시 아래를 포함:
- visible-line 기준이 적용된 함수
- `linewidth=0` edge가 배제됐는지
- 각 sample의 table markdown exact 여부
- 남은 diff가 있으면 정확한 table 번호
