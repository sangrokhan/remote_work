# Extractor Refactor Removals

이번 리팩토링에서 `extractor.py`의 미사용/레거시 영역을 제거했다. 아래 항목은 현재 코드 기준으로 지원하지 않으며, 추후 필요하면 이 문서를 기준으로 다시 설계해야 한다.

## 제거된 공개/내부 helper

- `_normalize_body_lines`
  - 본문 줄 배열을 문장 단위로 다시 합치는 레거시 helper였다.
  - 현재 본문 정규화는 `_build_body_blocks` + `_join_non_heading_block_lines` 조합으로만 처리한다.

- `_normalize_list_block_lines`
  - bullet/list depth를 추정해 `-`, `*`, `+` 형태로 재구성하던 helper였다.
  - 현재 본문 추출은 heading/paragraph만 다루며 list 전용 block normalization은 지원하지 않는다.

- `_is_list_continuation_line`
  - bullet continuation alignment를 이용해 같은 list item 여부를 판정하던 helper였다.
  - 현재 구현에서는 본문 병합 기준을 line gap 하나로 단순화했다.

- `_looks_like_inline_term_continuation`
  - mixed style, single-token line, 강조된 첫 단어 등을 근거로 문장 continuation 여부를 판정하던 helper였다.
  - 현재 paragraph merge는 style heuristic을 사용하지 않는다.

- `_has_room_for_next_line_start`
  - 이전 줄의 남은 폭과 다음 줄 첫 단어 폭을 비교해 줄바꿈 의도를 추정하던 helper였다.
  - 현재 paragraph merge는 width-based heuristic을 사용하지 않는다.

- `_looks_like_table`
  - `_table_rejection_reason(...) is None`의 thin wrapper였다.
  - 실제 생산 코드에서 사용되지 않아 제거했다.

## 함께 제거된 테스트 기대값

이전 `tests/test_extractor.py`에서 `skip` 상태로만 남아 있던 아래 기대값은 모두 제거했다.

- hyphen-ended line을 다음 줄과 무조건 붙이는 규칙
- `o`, `?`, `◆` 같은 문자 기반 bullet depth 추정 규칙
- color/bold/style 변화에 따른 paragraph split 규칙
- bullet continuation indent에 따른 list block 유지/분리 규칙
- heading/paragraph/list 3분류 body block 기대값

## 재구현이 필요할 때 확인할 질문

- list를 실제 출력 포맷에 포함할지, 아니면 paragraph로 평탄화할지
- paragraph merge가 line gap 외에 style/indent/width 신호를 다시 써야 하는지
- body text가 downstream에서 문단 의미 보존이 중요한지, 아니면 검색용 평탄화가 우선인지
- legacy heuristic을 본문 전용으로 둘지, 표 셀 정규화에도 재사용할지

## 현재 유지하는 기준

- 본문: heading/paragraph만 구분
- paragraph merge: line gap 기준
- 표 셀: sentence ending + bullet line 유지 중심
- 표 판단: `_table_rejection_reason` 단일 기준
