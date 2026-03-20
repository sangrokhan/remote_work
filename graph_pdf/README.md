# graph_pdf

pdfplumber 기반으로 헤더/푸터/워터마크를 제거한 본문 텍스트와
페이지별 표 추출, 페이지 이미지를 분리 저장하는 데모입니다.

## 실행 방법
```bash
# 가상환경
python3 -m venv .venv
source .venv/bin/activate

# 패키지 설치
pip install -r graph_pdf/requirements.txt

# 데모 실행
.venv/bin/python graph_pdf/run_demo.py

# 검증 실행
.venv/bin/python graph_pdf/verify.py
```

## 직접 실행 방법
샘플 생성 없이 임의의 PDF를 바로 추출하려면 `extractor` 모듈을 직접 실행하면 됩니다.

```bash
python3 -m extractor sample.pdf \
  --out-md-dir artifacts/manual/md \
  --out-image-dir artifacts/manual/images \
  --stem sample
```

또는 엔트리 파일을 직접 실행할 수도 있습니다.

```bash
python3 extractor/__main__.py sample.pdf \
  --out-md-dir artifacts/manual/md \
  --out-image-dir artifacts/manual/images \
  --stem sample
```

### 주요 옵션
- `--pages 1,3,5-8`: 추출할 페이지 범위 지정
- `--force-table`: 표 영역 탐지 실패 시 더 공격적인 페이지 전체 표 추출 허용
- `--debug`: 표 구조/edge 디버그 JSON 생성
- `--debug-watermark`: 회전 문자 디버그 JSON 생성

### 직접 실행 예시
```bash
python3 -m extractor sample.pdf \
  --out-md-dir artifacts/manual/md \
  --out-image-dir artifacts/manual/images \
  --stem sample \
  --pages 1-2 \
  --debug \
  --debug-watermark
```

### 직접 실행 산출물
- `artifacts/manual/md/sample.txt`: 본문 텍스트
- `artifacts/manual/md/sample.md`: 본문 markdown
- `artifacts/manual/md/sample_table.md`: 표 markdown
- `artifacts/manual/md/sample_summary.json`: 추출 요약
- `artifacts/manual/md/sample_debug.json`: 표 구조 + 원본 drawing 객체 + 텍스트 폰트 크기 프로파일 디버그 (`--debug`)
- `artifacts/manual/md/sample_edges_debug.json`: edge 분해 결과 디버그 (`--debug`)
- `artifacts/manual/md/sample_watermark_debug.json`: 회전 문자 디버그 (`--debug-watermark`)
- `artifacts/manual/images/*`: body 영역 이미지 추출 결과

## 파일별 역할
- `run_demo.py`: 샘플 PDF를 생성하고 추출 파이프라인을 실행하는 진입 스크립트
- `verify.py`: 데모 산출물이 기대한 shape인지 확인하는 검증 스크립트
- `sample_generator.py`: 본문, 표, 워터마크, 이미지가 포함된 테스트용 샘플 PDF 생성기
- `sample_fixture.py`: 샘플 PDF 검증용 fixture 로더
- `fixtures/demo_document.json`: 샘플 문서의 기대 본문/표 데이터 fixture
- `extractor/__init__.py`: 외부에서 사용하는 공개 진입점 export
- `extractor/__main__.py`: CLI 실행용 entrypoint
- `extractor/pipeline.py`: 전체 PDF 추출 orchestration, 페이지 순회, cross-page table merge, 결과 파일 기록
- `extractor/text.py`: 워터마크/레이아웃 artifact 제거, body bounds 계산, 본문 line 추출과 정규화
- `extractor/tables.py`: 표 영역 탐지, 표 추출, 셀 정규화, 페이지 간 표 continuation merge 판단, markdown table 렌더링
- `extractor/debug.py`: 표 선분/그리드, 원본 drawing 객체, 텍스트 스타일/폰트 크기 디버그 payload 생성
- `extractor/images.py`: body 영역과 겹치는 embedded image만 저장
- `extractor/shared.py`: 공통 타입, 상수, geometry/segment helper
- `tests/test_text.py`: 본문/워터마크/바운드 계산 관련 테스트
- `tests/test_tables.py`: 표 탐지/병합/세그먼트 처리 관련 테스트
- `tests/test_pipeline.py`: end-to-end 추출 결과와 debug/image 산출물 테스트
- `tests/test_public_api.py`: 제거된 레거시 helper가 공개 API로 다시 노출되지 않는지 확인
- `tests/test_refactor_boundaries.py`: 공개 진입점과 모듈 경계 smoke test
- `docs/extractor-refactor-removals.md`: 이번 리팩토링에서 제거한 레거시 동작 기록
- `docs/extractor-sequence.md`: extractor 패키지의 실행 시퀀스 정리 문서

## 현재 데모 동작 요약
- 헤더/푸터 제거: 페이지 상단/하단 마진 기반 제거
- 워터마크 제외: 텍스트 패턴 기반 필터 적용
- 표 추출:
  - `horizontal_edges`로 표 영역 탐지
  - 영역별 `explicit_vertical_lines` 보정으로 좌우 외곽선 없는 테이블 처리
  - 워터마크 문자 필터링 후 셀 정제
  - 셀 내부 자동 줄바꿈은 한 줄로 병합, bullet/의도된 멀티라인은 같은 행 블록 안에서 유지
  - 두 페이지에 걸친 동일 표를 하나로 병합하는 후처리
- 이미지 분리: 페이지별 PNG(`artifacts/.../images/*.png`)로 저장

## 샘플/검증 커버리지
- 본문 멀티라인 및 들여쓰기 텍스트
- 표 3개 컬럼, 좌측 컬럼 병합 형태(빈 셀로 표현)
- 표 셀 내 3라인 텍스트 및 bullet 포함
- 여러 크기의 테이블(작은/큰/컴팩트)
- 페이지 경계에서 분할되는 표를 하나의 표로 병합 처리

## pipeline 흐름
1. `extract_pdf_to_outputs(...)`가 PDF를 열고 출력 디렉터리를 준비합니다.
2. 선택된 페이지 범위가 있으면 그 페이지들만 순회합니다.
3. `debug=True`면 표 구조, 원본 drawing 객체, 텍스트 폰트 크기 프로파일, edge 디버그 payload를 수집합니다.
4. `debug_watermark=True`면 회전된 문자 디버그 payload를 수집합니다.
5. `extractor.tables._extract_tables(...)`가 현재 페이지의 표 후보를 찾고 행 데이터를 정규화합니다.
6. `extractor.text._extract_body_text(...)`가 전체 body text를 구합니다.
7. 표 bbox를 제외한 body text를 다시 계산해 최종 본문 markdown에 사용합니다.
8. 표가 있으면 이전 페이지의 pending table과 이어붙일 수 있는지 검사합니다.
9. 이어붙일 수 있으면 pending table을 확장하고, 아니면 이전 pending table을 flush한 뒤 현재 표를 새 pending 상태로 둡니다.
10. 모든 페이지를 처리한 뒤 남은 pending table을 flush합니다.
11. 본문 markdown, table markdown, summary json, optional debug json을 기록합니다.
12. 마지막으로 body 영역과 겹치는 embedded image만 별도 파일로 저장합니다.

## 산출물 예시 위치
- 텍스트/마크다운: `graph_pdf/artifacts/run_demo/md/demo.txt`, `demo.md`
- 이미지: `graph_pdf/artifacts/run_demo/images/demo_page_01.png`, `demo_page_02.png`

`demo.txt`의 표 출력은 마크다운 테이블이 아니라 아래와 같은 행 블록 구조입니다.

```text
### Page 1 table 1
- Row 1
  Item: Laptop
  Qty: 12
  Price: $120
  - line 1
```

## 결과 (verify.py)
- PASS
- 추출 텍스트/마크다운 파일 생성 확인
- 페이지별 이미지 저장 확인
- 표 병합 + 멀티라인/바디 포맷 검증 확인
