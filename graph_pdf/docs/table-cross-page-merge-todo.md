# PDF 추출 성능 개선 TODO (병합 로직 유지 포함)

## 목적
기존 기능 변경은 유지하면서, 현재 `--region-log` 없이도 체감 속도를 줄이는 동작 최적화를 진행한다.

## 최적화 TODO (최우선)
- [x] 1) `pipeline.py`에서 페이지당 텍스트 재추출 제거
  - [x] `preview_markdown` 계산 결과를 재사용해, 동일 페이지에서 필요 없는 추가 `_extract_body_text` 경로를 최소화.
  - [x] `full_page_text`로 계산되는 미사용 변수(`full_page_text`) 제거.

- [x] 2) 텍스트 영역 재사용 최적화
  - [x] `_body_text_boxes`를 페이지 1회 계산으로 정리하고 후속 병합 판단에서 재사용.
  - [x] 표 배치 bbox 목록 재생성 중복을 줄여 제외 필터 입력을 공유.

- [x] 3) 페이지 간 병합 판정 루프 비용 상한화
  - [x] `_has_intervening_regions_before/after`에서 조기 종료 경로 강화 및 region 타입 처리 분기 단순화.
  - [x] `region_map` 조회는 존재할 때만 수행하고, `current_page`/`previous_page` 상태별 early path 정리.

- [x] 4) 디버그 경로 분리로 기본 경로 경량화
  - [x] `debug`, `debug_watermark` 비활성 시 `debug_*` 수집 버퍼를 할당하지 않음.
  - [x] `debug`가 false면 `table_debug_pages`, `edge_debug_pages`의 객체 생성/JSON 조합을 생략.

- [ ] 5) 이미지 참조 수집 단계 재검토
  - [ ] `_collect_embedded_image_refs`의 사전 페이지 스캔 비용을 줄이기 위해, 텍스트/테이블/이미지 추출과 동시 사용 가능한 입력 최소화.
  - [ ] `extract_pdf_to_outputs`에서 실제 필요 페이지만 탐색하도록 보장해 불필요한 PDF 순회 제거.

- [ ] 6) 병합 판정에서 불필요한 정렬/리스트 조작 제거
  - [ ] `tables = sorted(...)` 전/후 정렬 경로를 리뷰하여 이미 정렬된 후보 순서를 재사용할 수 있는지 점검.
  - [x] `gap_text_boxes` 병합 시 중복 리스트 생성 횟수를 최소화.

- [x] 7) 결과 직렬화 비용 최적화
  - [x] 기본 경로에서 `region_log` 미사용 시 `region_map`은 최소 데이터만 유지.
  - [ ] `region_log` 사용 시에도 큰 JSON 객체를 한 번에 만들지 않고 배치 기반/요약 기반 기록 방식을 검토.

- [ ] 8) 병합 관련 성능 계측기 추가(검증용)
  - [ ] 페이지 단위 타이밍(log): `detect body`, `extract tables`, `extract text`, `cross-page merge`, `serialize`.
  - [ ] `topK` 지연 페이지와 큰 텍스트/표 페이지에서 어떤 단계가 병목인지 1회 리포트.

## 후속 체크 (병합 정확도 검증)
- [ ] 9) 최적화 전후 결과 일치성 체크 추가
  - [ ] `table count`, `document count`, `table_count`, `summary` 키를 유지 비교.
  - [ ] 성능 패치 이후 동일 입력에서 결과 변화가 없음을 기준 기준으로 검증.
