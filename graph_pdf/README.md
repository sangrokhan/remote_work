# graph_pdf

`graph_pdf`는 `pdfplumber` 기반 PDF 추출기입니다. 본문 markdown, 표 markdown, 이미지 파일, 요약 JSON을 한 번에 생성합니다.

현재 구현은 단순 텍스트 덤프보다 구조 보존에 더 초점을 둡니다.

- 본문은 header/footer 바깥의 body 영역만 추출합니다.
- 표는 black line/grid 기반 구조를 우선 사용하고, 필요할 때만 fallback 경로를 탑니다.
- note 영역은 표가 아니라 `Note:` 문장으로 본문에 남깁니다.
- embedded image와 drawing 기반 도형 이미지를 따로 추출합니다.
- `## 문서ID ...` 형태 heading이 나오면 문서를 분리하고 출력 파일명도 해당 문서 ID를 따릅니다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 빠른 실행

기본 출력 경로는 CLI 기준으로 아래와 같습니다.

- markdown: `graph_pdf/artifacts/md`
- images: `graph_pdf/artifacts/images`

```bash
python3 -m extractor /path/to/document.pdf
```

출력 경로를 직접 지정하려면:

```bash
python3 -m extractor /path/to/document.pdf \
  --out-md-dir artifacts/manual/md \
  --out-image-dir artifacts/manual/images
```

엔트리 파일을 직접 실행해도 동일하게 동작합니다.

```bash
python3 extractor/__main__.py /path/to/document.pdf \
  --out-md-dir artifacts/manual/md \
  --out-image-dir artifacts/manual/images
```

## CLI 옵션

- `--pages 1,3,5-8`: 1-based 페이지 범위 선택
- `--force-table`: 표 탐지 실패 시 더 공격적인 표 추출 시도
- `--debug`: 표/전략/문서 텍스트 프로파일 디버그 JSON 생성
- `--debug-watermark`: 회전 watermark 문자 디버그 JSON 생성
- `--profile-fonts`: body text 기준 font size + color 프로파일만 생성
- `--heading-profile <path>`: 외부 heading 규칙 JSON 사용
- `--page-write`: 본문 markdown에 `[//]: # (Page N)` 주석 삽입
- `--raw <path>`: 입력 PDF를 raw dump로 저장
- `--from-raw <path>`: raw dump를 임시 PDF로 복원해 같은 파이프라인 실행
- `--region-log <path>`: 페이지별 tables/text/images/notes 영역 로그 JSON 저장

## 현재 파이프라인 동작

### 1. 본문 영역 계산

본문 추출은 고정 margin만 쓰지 않습니다.

- 페이지 상단/하단의 긴 horizontal divider가 있으면 그 선을 우선 body bound로 사용
- 상단 divider가 없으면 `Chapter`, `Section`, `Appendix` 같은 큰 heading을 body 시작점 후보로 사용
- 그래도 못 찾으면 `header_margin`, `footer_margin` 기본값으로 fallback

### 2. heading과 문서 분리

기본적으로 `fixtures/font_heading_profile.json`이 존재하면 자동 로드됩니다. 별도 `--heading-profile`을 주면 그 파일이 우선합니다.

- heading 규칙은 `font_size -> h1~h6` 또는 `markdown_prefix`로 매핑
- 본문 line을 heading/paragraph 블록으로 묶은 뒤 markdown heading으로 변환
- `## ...` 라인에서 문서 ID를 추출합니다
- 문서 ID는 heading 첫 토큰을 사용하며 파일명 안전 문자로 정규화됩니다

예:

- `## FGR-TEST01 Example Feature` -> 문서 ID `FGR-TEST01`
- 출력 파일: `FGR-TEST01.md`, `FGR-TEST01_tables.md`

문서 시작 전에 나온 본문은 첫 문서가 실제로 감지될 때까지 `output` 문서에 유지됩니다.

### 3. 표 추출

표 추출은 아래 순서로 진행됩니다.

- body 내부의 table region 후보 탐지
- 검은 선/사각형을 이용해 row/column band 구성
- word payload를 grid cell에 배정
- split row와 반복 header를 후처리
- markdown table로 렌더링

여러 페이지에 걸친 표는 별도 상태로 유지합니다.

- 직전 페이지 표의 축(axis) 정렬
- 열 개수 유사성
- 페이지 경계 근처 위치
- 중간 영역에 text/image/note가 끼어들지 않았는지

위 조건이 맞으면 cross-page table로 이어 붙입니다.

### 4. note 처리

note처럼 보이는 영역은 표로 남기지 않고 본문 reference line으로 넣습니다.

- note anchor 기반으로 후보 영역을 수집
- multi-anchor note는 anchor별로 분할
- 최종 본문에는 `Note: ...` 한 줄 텍스트로 삽입

표 bbox가 note 영역과 거의 동일하면 해당 표는 버립니다.

### 5. 이미지 처리

이미지는 두 종류를 다룹니다.

- embedded image: PDF 내부 image object
- drawing image: line/curve/rect/char 조합으로 이루어진 도형 묶음

본문 영역과 겹치는 것만 추출하고, note marker와 겹치는 영역은 제외합니다. drawing 이미지의 경우 실제 렌더 bbox를 연결된 도형 기준으로 확장한 뒤 PNG로 저장합니다.

## 출력 파일

기본 stem은 CLI에서 항상 `output`입니다. 다만 실제 문서 분리가 일어나면 문서별 파일명은 문서 ID 기준으로 생성됩니다.

### 문서별 산출물

각 문서마다:

- `<document_id>.txt`
- `<document_id>.md`
- `<document_id>_tables.md`
- `<document_id>_summary.json`
- `<document_id>_images/`

예:

- `output.md`
- `output_tables.md`
- `output_summary.json`
- `output_images/`

또는 문서 분리 시:

- `FGR-TEST01.md`
- `FGR-TEST01_tables.md`
- `FGR-TEST01_summary.json`
- `FGR-TEST01_images/`

### 전체 실행 기준 산출물

실행 전체 기준 summary도 따로 생성합니다.

- `<stem>_summary.json`

이 파일에는 아래가 들어갑니다.

- 입력 PDF 경로
- 대표 문서 파일 경로
- 전체 table 수
- 전체 document 수
- 문서별 summary 목록

### 선택적 디버그 산출물

- `<stem>_debug.json`
- `<stem>_edges_debug.json`
- `<stem>_watermark_debug.json`
- `--region-log`로 지정한 JSON 파일

## heading profile 형식

예시 파일: `fixtures/font_heading_profile.json`

핵심 필드:

- `heading_rules[].match.font_size`
- `heading_rules[].match.max_x0` (선택)
- `heading_rules[].assign.tag`
- `heading_rules[].assign.markdown_prefix`

예:

```bash
python3 -m extractor /path/to/document.pdf \
  --heading-profile fixtures/font_heading_profile.json
```

## font profile 모드

표 추출 결과를 제외한 body text 기준으로 font style 빈도를 분석합니다.

```bash
python3 -m extractor /path/to/document.pdf \
  --profile-fonts \
  --out-md-dir artifacts/manual/md
```

생성 파일:

- `output_font_profile.json`
- `output_font_profile.csv`

## raw dump / replay

원본 PDF 또는 선택 페이지 subset을 base64 raw dump로 저장할 수 있습니다.

```bash
python3 -m extractor /path/to/document.pdf \
  --raw artifacts/manual/raw/document.raw.dump
```

현재 raw payload 핵심 필드:

- `schema_version`
- `document_pdf_base64`

raw dump를 다시 실행하려면:

```bash
python3 -m extractor \
  --from-raw artifacts/manual/raw/document.raw.dump \
  --out-md-dir artifacts/manual/md \
  --out-image-dir artifacts/manual/images
```

## Python API

주요 공개 진입점은 `extractor.extract_pdf_to_outputs`와 `extractor.profile_pdf_fonts`입니다.

```python
from pathlib import Path
from extractor import extract_pdf_to_outputs

result = extract_pdf_to_outputs(
    pdf_path=Path("sample.pdf"),
    out_md_dir=Path("artifacts/md"),
    out_image_dir=Path("artifacts/images"),
    stem="output",
)

print(result["md_file"])
print(result["table_md_file"])
print(result["summary"])
```

## 모듈 구조

- `extractor/__main__.py`: CLI entrypoint
- `extractor/__init__.py`: 공개 API export
- `extractor/pipeline.py`: 문서 분할, note/table/image 조합, 최종 산출물 생성
- `extractor/text.py`: body bound 계산, line/block 정규화, heading 처리
- `extractor/tables.py`: 표 탐지, black line grid 분석, markdown 렌더링
- `extractor/notes.py`: note 후보 수집과 `Note:` 텍스트 변환
- `extractor/images.py`: embedded/drawing image 추출
- `extractor/font_profile.py`: body text font profile 생성
- `extractor/raw.py`: raw dump 저장/복원
- `extractor/debug.py`: 디버그 payload 생성
- `extractor/shared.py`: 공통 geometry/text helper

## 테스트

현재 `tests/` 아래 pytest 기반 테스트를 사용합니다.

- `tests/test_pipeline.py`: 문서 분리, cross-page table, 산출물 조합
- `tests/test_tables.py`: 표 region, grid row 구성, markdown 렌더링
- `tests/test_text.py`: body bound, heading/paragraph 병합, watermark 처리
- `tests/test_images.py`: 이미지 crop/추출
- `tests/test_font_profile.py`: font profile 출력
- `tests/test_public_api.py`: 공개 API 경계
- `tests/test_refactor_boundaries.py`: 패키지 진입점 smoke test

전체 수집 개수는 현재 기준 `85개`입니다.
