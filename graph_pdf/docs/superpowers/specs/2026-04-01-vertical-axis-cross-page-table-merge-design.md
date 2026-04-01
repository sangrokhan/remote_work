# Vertical-Axis Cross-Page Table Merge Design

## Goal

Cross-page table merge를 PRD 기준으로 단순화한다. 다음 페이지 첫 표와 이전 페이지 마지막 표를 병합할 때, 내용 기반 heuristic을 제거하고 다음 두 조건만 기준으로 삼는다.

1. 두 표의 `vertical axes`가 충분히 겹친다.
2. 두 표 사이에 `text`, `note`, `image`, `other table` 영역이 없다.

## Current Problem

현재 구현은 [extractor/pipeline.py](/home/han/.openclaw/workspace/remote_work/graph_pdf/extractor/pipeline.py)의 `_pick_cross_page_anchor()`와 `_looks_like_cross_page_continuation_rows()`를 통해 row 내용, header signature, family/type/description 예외 규칙까지 사용해 continuation 여부를 판단한다.

이 방식은 다음 문제를 만든다.

- PRD에 없는 내용 기반 규칙이 merge 판단에 개입한다.
- 표 끝에 note/prose row가 섞이면 대표 row 선택이 오염되어 merge가 실패한다.
- `raw-93-114`처럼 geometry는 맞지만 내용 heuristic이 실패하는 경우, 같은 계열 표가 분리되거나 중복 출력된다.

## Decision

Cross-page merge 판정은 구조 기반으로만 수행한다.

유지할 조건:

- 이전 페이지 마지막 표와 현재 페이지 첫 표 조합만 continuation 후보로 본다.
- 두 표의 `vertical axes` overlap이 있어야 한다.
- `region_map` 기준으로 두 표 사이에 `text`, `note`, `image`, `other table`가 있으면 merge하지 않는다.
- 페이지 경계 근처 표라는 기존 위치 제약은 유지한다.

제거할 조건:

- `_looks_like_cross_page_continuation_rows()`
- header signature 비교
- first row signature 비교
- family/type/description 전용 내용 heuristic
- row 내용 기반 continuation 판정 전체

## Implementation Shape

### 1. Merge anchor selection 단순화

[extractor/pipeline.py](/home/han/.openclaw/workspace/remote_work/graph_pdf/extractor/pipeline.py)의 `_pick_cross_page_anchor()`를 아래 기준으로 재작성한다.

- 후보 anchor는 `x overlap` 또는 `vertical axis overlap`이 있는 경우만 유지
- `_table_shapes_compatible()`는 유지하되, 내용 기반 `_looks_like_cross_page_continuation_rows()` 호출은 제거
- `_has_cross_page_gap_blocked()`와 `_continuation_regions_should_merge()`를 통과하면 anchor 후보로 인정
- 가장 최근 페이지, 가장 큰 overlap score를 우선 선택

### 2. Dead heuristic 제거

더 이상 cross-page merge에 사용하지 않는 아래 함수/로직을 제거하거나 비사용 상태로 정리한다.

- `_looks_like_cross_page_continuation_rows()`
- `_table_header_signature()`가 merge 판정 용도로만 쓰이던 분기
- family/type/description continuation 특례

### 3. Regression coverage

테스트는 구조 기준으로 다시 고정한다.

- axes가 겹치고 gap이 비어 있으면 merge
- axes가 안 겹치면 merge하지 않음
- gap 사이에 `text/note/image/other table`가 있으면 merge하지 않음
- row 내용이 달라도 위 구조 조건만 맞으면 merge 판정은 유지됨

## Expected Outcome

- `raw-93-114`의 merge 여부가 note/prose contamination에 영향받지 않는다.
- cross-page table merge 기준이 PRD와 일치한다.
- continuation merge 디버깅이 geometry 기반으로 단순해진다.
