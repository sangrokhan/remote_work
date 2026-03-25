# PDF 추출 성능 개선 TODO (병합 로직 유지 포함)

## 목적
기존 기능 변경은 유지하면서, 현재 `--region-log` 없이도 체감 속도를 줄이는 동작 최적화를 진행한다.

## 최적화 TODO (최우선)
- [ ] 1) `pipeline.py`에서 페이지당 텍스트 재추출 제거
  - [ ] `preview_markdown` 계산 결과를 캐싱해 같은 페이지에서 2회 이상 동일 `_extract_body_text` 호출이 발생하지 않게 통합.
  - [ ] `full_page_text`로 계산되는 미사용 변수(`full_page_text`) 제거.

- [ ] 2) 텍스트 영역 재사용 최적화
  - [ ] `_body_text_boxes`를 페이지 1회만 계산하고, `_gap_text_boxes_before_bbox/_after_bbox`에 전달해 추가 추출 호출 제거.
  - [ ] 가능한 곳에서 필터링 기준(표/도형/이미지 제외) 캐시를 재사용하도록 함수 시그니처 정리.

- [ ] 3) 페이지 간 병합 판정 루프 비용 상한화
  - [ ] `_has_intervening_regions_before/after`에서 현재/이전 페이지 전체 스캔 대신 `has_any` 조기 종료 경로를 강화.
  - [ ] `region_map` 조회는 존재할 때만 수행하고, `current_page`/`previous_page`가 비어 있으면 즉시 false 처리.

- [ ] 4) 디버그 경로 분리로 기본 경로 경량화
  - [ ] `debug`, `debug_watermark` 비활성 시 `debug_*` 수집 로직이 메모리/CPU 경로에 개입하지 않도록 조건 분기 정리.
  - [ ] `debug`가 false면 `table_debug_pages`, `edge_debug_pages`의 객체 생성/JSON 조합을 전면 생략.

- [ ] 5) 이미지 참조 수집 단계 재검토
  - [ ] `_collect_embedded_image_refs`의 사전 페이지 스캔 비용을 줄이기 위해, 텍스트/테이블/이미지 추출과 동시 사용 가능한 입력 최소화.
  - [ ] `extract_pdf_to_outputs`에서 실제 필요 페이지만 탐색하도록 보장해 불필요한 PDF 순회 제거.

- [ ] 6) 병합 판정에서 불필요한 정렬/리스트 조작 제거
  - [ ] `tables = sorted(...)` 전/후 정렬 경로를 리뷰하여 이미 정렬된 후보 순서를 재사용할 수 있는지 점검.
  - [ ] `gap_text_boxes` 병합 시 중복 리스트 생성 횟수를 최소화.

- [ ] 7) 결과 직렬화 비용 최적화
  - [ ] 기본 경로에서 `region_log`를 쓰지 않을 때 `region_map`은 최소 데이터만 유지.
  - [ ] `region_log` 사용 시에도 큰 JSON 객체를 한 번에 만들지 않고 배치 기반/요약 기반 기록 방식을 검토.

- [ ] 8) 병합 관련 성능 계측기 추가(검증용)
  - [ ] 페이지 단위 타이밍(log): `detect body`, `extract tables`, `extract text`, `cross-page merge`, `serialize`.
  - [ ] `topK` 지연 페이지와 큰 텍스트/표 페이지에서 어떤 단계가 병목인지 1회 리포트.

## 후속 체크 (병합 정확도 검증)
- [ ] 9) 최적화 전후 결과 일치성 체크 추가
  - [ ] `table count`, `document count`, `table_count`, `summary` 키를 유지 비교.
  - [ ] 성능 패치 이후 동일 입력에서 결과 변화가 없음을 기준 기준으로 검증.
