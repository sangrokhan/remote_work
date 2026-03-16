# graph_pdf

pdfplumber 기반으로 헤더/푸터/워터마크를 제거한 본문 텍스트와
페이지별 테이블 추출, 페이지 이미지를 분리 저장하는 데모입니다.

## 구조
- `sample_generator.py`: 테스트용 샘플 PDF 생성기
- `extractor.py`: 본문 텍스트 정제, 표 추출, 페이지 이미지 추출 로직
- `run_demo.py`: 샘플 PDF 생성 + 추출 파이프라인 실행
- `verify.py`: 간단한 검증 스크립트
- `requirements.txt`: 실행에 필요한 라이브러리 목록

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

## 현재 데모 동작 요약
- 헤더/푸터 제거: 페이지 상단/하단 마진 기반 제거
- 워터마크 제외: 텍스트 패턴 기반 필터 적용
- 표 추출:
  - `horizontal_edges`로 표 영역 탐지
  - 영역별 `explicit_vertical_lines` 보정으로 좌우 외곽선 없는 테이블 처리
  - `lines` 기반 2-pass 추출 + fallback
- 이미지 분리: 페이지별 PNG(`artifacts/.../images/*.png`)로 저장

## 산출물 예시 위치
- 텍스트/마크다운: `graph_pdf/artifacts/run_demo/md/demo.txt`, `demo.md`
- 이미지: `graph_pdf/artifacts/run_demo/images/demo_page_01.png`, `demo_page_02.png`

## 결과 (verify.py)
- PASS
- 추출 텍스트 파일 생성 확인
- 테이블 2개 분리 추출 확인
- 페이지 이미지 2개 저장 확인
