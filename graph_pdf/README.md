# graph_pdf

`pdfplumber` 기반으로 PDF에서 본문 텍스트, 표 markdown, 이미지 파일을 추출합니다.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI
```bash
python3 -m extractor /path/to/document.pdf \
  --out-md-dir artifacts/manual/md \
  --out-image-dir artifacts/manual/images
```

엔트리 파일을 직접 실행해도 됩니다.

```bash
python3 extractor/__main__.py /path/to/document.pdf \
  --out-md-dir artifacts/manual/md \
  --out-image-dir artifacts/manual/images
```

## Options
- `--pages 1,3,5-8`: 추출할 페이지 범위 지정
- `--force-table`: 표 탐지 실패 시 더 공격적인 추출 허용
- `--debug`: 표 구조와 drawing 디버그 JSON 생성
- `--debug-watermark`: 회전 문자 디버그 JSON 생성
- `--profile-fonts`: body text 기준 `font_size + font_color` 프로파일 JSON/CSV 생성
- `--heading-profile <path>`: 외부 JSON의 heading 규칙 적용
- `--raw <path>`: 입력 PDF를 최소 raw dump로 저장
- `--from-raw <path>`: raw dump를 임시 PDF로 복원해 같은 파이프라인 실행
- `--page-write`: markdown 출력에 페이지 주석 추가

## Heading Profile
기본 heading 규칙 예시는 `fixtures/font_heading_profile.json`에 있습니다. `heading_rules[].match.font_size`와 `heading_rules[].assign.tag` 또는 `markdown_prefix`를 사용합니다.

```bash
python3 -m extractor /path/to/document.pdf \
  --out-md-dir artifacts/manual/md \
  --out-image-dir artifacts/manual/images \
  --heading-profile fixtures/font_heading_profile.json
```

## Raw Replay
문서 전체를 raw dump로 저장하려면 `--raw`를 사용합니다.

```bash
python3 -m extractor /path/to/document.pdf \
  --raw artifacts/manual/raw/document.raw.dump
```

raw dump에는 `schema_version`, `document_pdf_base64`만 저장됩니다.

raw dump를 입력으로 재실행하려면 `--from-raw`를 사용합니다.

```bash
python3 -m extractor \
  --from-raw artifacts/manual/raw/document.raw.dump \
  --out-md-dir artifacts/manual/md \
  --out-image-dir artifacts/manual/images
```

## Outputs
- `artifacts/manual/md/output.txt`: 본문 텍스트
- `artifacts/manual/md/output.md`: 본문 markdown
- `artifacts/manual/md/output_tables.md`: 표 markdown
- `artifacts/manual/md/output_summary.json`: 추출 요약
- `artifacts/manual/md/output_font_profile.json`: 폰트 프로파일 JSON
- `artifacts/manual/md/output_font_profile.csv`: 폰트 프로파일 CSV
- `artifacts/manual/images/output_images/*`: 추출 이미지

## Modules
- `extractor/__init__.py`: 공개 진입점 export
- `extractor/__main__.py`: CLI entrypoint
- `extractor/font_profile.py`: body text font profile 생성
- `extractor/pipeline.py`: 전체 추출 orchestration
- `extractor/raw.py`: raw dump export/import helper
- `extractor/text.py`: body text 추출과 정규화
- `extractor/tables.py`: 표 탐지와 table markdown 렌더링
- `extractor/debug.py`: 디버그 payload 생성
- `extractor/images.py`: embedded image 저장
- `extractor/shared.py`: 공통 타입과 geometry helper

## Tests
- `tests/test_text.py`: 본문/워터마크/바운드 계산
- `tests/test_tables.py`: 표 탐지/병합/세그먼트 처리
- `tests/test_pipeline.py`: 파이프라인/문서 분리/표 병합 관련 테스트
- `tests/test_font_profile.py`: font profile CLI와 출력 포맷
- `tests/test_public_api.py`: 공개 API 경계 확인
- `tests/test_refactor_boundaries.py`: 공개 진입점 smoke test
