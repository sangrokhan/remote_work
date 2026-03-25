# raw-38-50 노트/표 분류 메모

## 대상 샘플
- `samples/raw-38-50.dump` (재생성 PDF 13페이지)

## 의도 동작(확정 케이스)
- 8페이지: 노트 1개
- 9페이지: 노트 1개
- 10페이지: `Parameter`/`Description` 타입의 표 시작 1개
- 11페이지: 10페이지에서 시작한 표 연장 1개 + 11→12 시작 표 1개
- 12페이지: 11→12 연장 1개 + 12→13 시작 표 1개
- 13페이지: 12→13 연장 1개 + 단일 표 1개

## 구현 반영 내용
- `_estimate_region_kind()`에서 `파라미터 설명형` 레이아웃(`_is_parameter_description_layout`)은
  `note` 점수 비교로 들어가지 않도록 `reason="parameter_description_layout"`으로 바로 `table` 분류.
- `single-column` 조건을 넘은 후보는 기존처럼 `table` 경로로 유지하여
  `key/value` 열 분리를 가진 표가 노트로 오분류되지 않도록 조정.
