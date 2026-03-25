# Region Extraction Baseline Compatibility Check

## 목표
영역 로그(`--region-log`) 추가 전/후 기존 출력 산출물 동일성 확인

## 실행 환경
- Repository: `/home/han/.openclaw/workspace/remote_work/graph_pdf`
- 입력: `sample.pdf`
- 대상 페이지: `1-3`
- 샘플 스템: `sample`

## 비교 실험
- Baseline: `python3 -m extractor sample.pdf --out-md-dir /tmp/region-check-baseline/md --out-image-dir /tmp/region-check-baseline/images --stem sample --pages 1-3`
- Region Log 케이스: `python3 -m extractor sample.pdf --out-md-dir /tmp/region-check-logged/md --out-image-dir /tmp/region-check-logged/images --stem sample --pages 1-3 --region-log /tmp/region-check-logged/md/sample_regions.json`

## 해시 비교 결과
- `sample.txt`: 동일
- `sample.md`: 동일
- `sample_table.md`: 동일

(두 경우 모두 SHA-256 동일하게 확인됨)

## 생성 로그
- Baseline 산출물: `/tmp/region-check-baseline/md/`
- Region 로그 산출물: `/tmp/region-check-logged/md/sample_regions.json`

## Header/푸터 마진 민감도 점검 (TODO-12 미응답 확인)
- 목표: `--region-log`에 body bounds를 남기고, 헤더/푸터 마진 변경이 표 병합/출력에 미치는 영향을 확인.
- 추가 실행(임시):
  - `header_margin=90, footer_margin=40` (기본)
  - `header_margin=20, footer_margin=20`
  - `header_margin=0, footer_margin=0`
- 확인 결과(샘플 `sample.pdf`):
  - 각 케이스 `summary.table_count=4`, `table_markdown` 길이 동일
  - 페이지 1의 `body_top/body_bottom`도 동일(`72.0/722.0`)
- 확인 결과(덤프 기반 3종: `raw-32-37`, `raw-38-50`, `raw-93-114`):
  - `table_count`: 기본/20&20/0&0 모두 동일
  - `table_markdown` 동일 (예: raw-93-114 `23`, raw-38-50 `5`, raw-32-37 `5`)
- 결론:
  - 현재 샘플군에서는 header/footer 제외(본문 바운드 설정)가 cross-page 병합 오탐/미탐에 직접적인 영향 신호를 보이지 않음.
  - 다만 `region_log`에 `body_top/body_bottom/header_margin/footer_margin`를 남기도록 해 추후 문서별 추적을 가능하게 함.
