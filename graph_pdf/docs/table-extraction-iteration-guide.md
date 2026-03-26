# Table Extraction Iteration Guide (raw dump 기반)

## 목적
- 3개 덤프(`raw-32-37`, `raw-38-50`, `raw-93-114`)에 대해 모든 코드 변경을 검증한다.
- `md` 추출 결과와 시각 PDF(`--from-raw`로 복원한 PDF)를 교차 확인해, 헤더/내용 분리, 노트 판정, 이미지 추출까지 반복 보정한다.
- 변경이 특정 덤프에서 해결될 때, 다른 덤프로 회귀가 퍼졌는지 명확히 보고한다.

## 보호 대상 산출물
- 사용자 제공 기준 산출물인 `samples/raw-93-114.md`와 `samples/raw-93-114_table.md`는 절대 수정하지 않는다.
- 위 두 파일은 `raw-93-114 dump`의 기대 출력으로 간주하고, 비교/검증 기준으로만 사용한다.
- 작업 중 두 파일의 수정이 필요해 보일 경우에도 임의로 변경하지 말고, 반드시 사용자에게 수정 필요 사유를 먼저 보고하고 승인 대기한다.
- `samples/raw-93-114.md`의 image reference baseline은 실제 이미지 2개(`image 1`, `image 2`)다.

## 1) 기준선(Baseline) 고정
1. 변경 전 현재 상태를 기준선 디렉터리로 저장한다.
   1. `artifacts/iter/baseline/raw_visuals/pdfs`
   2. `artifacts/iter/baseline/raw_visuals/parse`
2. 다음 명령으로 덤프별 PDF 복원 + 파싱 + 레이아웃 분석을 수행한다.
   - `PYTHONPATH=. python3 scripts/replay_samples.py --samples-dir samples --pattern "raw-*.dump" --pdf-dir artifacts/iter/baseline/raw_visuals/pdfs --out-dir artifacts/iter/baseline/raw_visuals/parse --force --analyze-layout`
3. 생성 산출물 기준은 아래를 모두 보존한다.
   - `<dump>.pdf`
   - `<dump>_debug.json`
   - `<dump>_md` 디렉터리 결과 (`.md`, `.txt`, `_table.md`)
   - `sample_raw_visualization_report.json`

## 2) 1회 변경 후 실행 루틴
1. 코드 변경 후 같은 명령을 새 라벨로 재실행한다.
   - `PYTHONPATH=. python3 scripts/replay_samples.py --samples-dir samples --pattern "raw-*.dump" --pdf-dir artifacts/iter/<run>/raw_visuals/pdfs --out-dir artifacts/iter/<run>/raw_visuals/parse --force --analyze-layout`
2. 이전 라벨과 비교한다.
   1. table 수, note/ table region 수, 페이지별 후보 수, 텍스트 길이, 이미지 파일 수 비교
   2. `report` JSON의 `table_count_pdf`, `text_chars_pdf`, `notes/tables` 계수, 경계 박스 관련 디버그 필드 비교
   3. `md` 본문의 `[Image reference: ...]` 개수와 위치 변화를 반드시 비교한다.
   4. `raw-93-114`는 image reference가 정확히 2개(`image 1`, `image 2`)인지 함께 확인한다.
3. 시각 확인은 `raw_visuals/pdfs`의 PDF를 열어 아래를 페이지별로 대조한다.
   - 표 헤더/본문 행 구분
   - 헤더만 있는 단락형 표(헤더 반복/누락)
   - 노트 후보 경계선(파란/검은 선), 문단 박스, 이미지 추출 구간
   - 인접 표 병합(특히 페이지 경계 부근)

## 3) 영향 범위 보고 포맷 (반드시 작성)
모든 변경 후 아래 형식으로 보고한다.

- 변경 영향 요약
  - 변경 파일/함수
  - 변경 의도
  - 변경 전제 조건

- 덤프별 차이 요약
  - `raw-32-37`: 변경 페이지, 변경 표 수, 변경 note 수, 변경 이미지 수, 변경 image reference 수
  - `raw-38-50`: 변경 페이지, 변경 표 수, 변경 note 수, 변경 이미지 수, 변경 image reference 수
  - `raw-93-114`: 변경 페이지, 변경 표 수, 변경 note 수, 변경 이미지 수, 변경 image reference 수

- 근거 첨부
  - 새/구 `debug.json`의 `pages[*].detected_tables`, `pages[*].strategy_debug`, `pages[*].tables`
  - `table_markdown` diff
  - 필요 시 `md` 텍스트 diff
  - `[Image reference: ...]` diff 및 개수 비교
  - 대상 PDF 페이지 번호(시각 확인된 페이지)

## 4) 승인 게이트 (필수)
- 변경이 특정 덤프 한정 문제만 개선했더라도 아래 규칙이 만족되어야 다음 단계로 진행.
  - 다른 덤프의 회귀가 0건인지 확인
  - 회귀가 있다면, 회귀 항목, 페이지, 원인 가설까지 보고하고 사용자 승인 대기

## 5) 비교 리포트 생성용 최소 체크리스트
1. `table_count`가 바뀐 덤프를 우선 점검한다.
2. `page` 단위로 변경된 `detected_tables` 개체 수를 비교한다.
3. `note`와 `table` 종류가 뒤바뀐 케이스를 추출한다.
4. 이미지 누락/과다 여부:
   5. `note`로 분류된 영역이 이미지로 추출되는지 여부
   6. 표 영역이 image로 남아 있지 않은지 확인
7. `md` 내 `[Image reference: ...]`가 새로 생성되거나 위치가 바뀌었는지 확인한다.
8. 이미지 파일 검증과 별개로, image reference 증감은 독립적인 회귀 항목으로 기록한다.
9. `raw-93-114`에서는 baseline인 `image 1`, `image 2` 외 추가 reference 생성 또는 번호 불일치를 회귀로 본다.

## 6) 시각/텍스트 동시 비교 사이클 예시
1. `md`에서 표/문장 단위를 확인한다.
2. PDF의 같은 페이지에서 표 경계선/셀 분할을 본다.
3. 문제 원인 태그를 붙인다 (`오탐 노트`, `헤더 분리 실패`, `열 분리`, `연속 페이지 병합 실패`).
4. 원인 한 가지씩 우선순위 반영해서 한 번에 한 규칙만 수정한다.
5. 다시 `replay_samples.py` 전체 3덤프 재실행.
6. 변경이 특정 덤프에서 해결되면, 영향이 없는지 다른 두 덤프에 대해 동일 방식 검증.

## 7) 사용자 확인이 필요한 보고 항목
- 변경 결과 보고 시 다음을 명확히 전달한다.
  - 변경된 코드 위치(파일/함수/로직)
  - 해결된 덤프와 페이지
  - 회귀한 덤프와 페이지
  - 허용 가능한 부작용(있다면)
  - 다음 실행 제안

## 권장 실행 루틴 요약
1. 코드 변경 1개 단위
2. 3덤프 전체 재파싱
3. 덤프별 차이표 생성
4. PDF 시각 확인
5. 승인 의사 결정 후 다음 수정
