# Extractor Sequence

현재 `extractor` 패키지는 아래 순서로 동작한다.

## 모듈 책임

- `extractor/pipeline.py`
  - 전체 추출 orchestration
  - 페이지 순회, cross-page table merge, 파일 출력

- `extractor/text.py`
  - 워터마크/레이아웃 artifact 제거
  - body bounds 계산
  - 본문 line payload 추출과 paragraph 단위 정규화

- `extractor/tables.py`
  - 표 후보 영역 계산
  - 표 추출, 셀 정규화, 페이지 간 continuation merge 판단
  - markdown table 렌더링

- `extractor/debug.py`
  - 회전 문자 디버그
  - 표 선분/그리드 디버그
  - edge selection 디버그

- `extractor/images.py`
  - body 영역과 겹치는 embedded image만 추출

- `extractor/shared.py`
  - 공통 타입, 상수, geometry/segment helper

## 실행 시퀀스

1. `extract_pdf_to_outputs(...)`가 PDF를 연다.
2. 페이지를 순회하면서 선택되지 않은 페이지는 건너뛴다.
3. `debug=True`면 선분/표 구조 디버그 payload를 수집한다.
4. `debug_watermark=True`면 회전된 문자 정보를 수집한다.
5. `tables._extract_tables(...)`가 현재 페이지의 표 후보와 표 데이터를 추출한다.
6. `text._extract_body_text(...)`가 전체 본문 텍스트를 뽑는다.
7. 표 bbox를 제외한 본문 텍스트를 다시 계산해 페이지 본문 출력에 쓴다.
8. 표가 있으면 이전 페이지의 pending table과 이어붙일 수 있는지 검사한다.
9. 이어붙일 수 있으면 pending table을 확장한다.
10. 이어붙일 수 없으면 이전 pending table을 flush하고 현재 표를 새 pending 상태로 둔다.
11. 모든 페이지를 돌고 나면 남아 있는 pending table을 flush한다.
12. 본문 markdown, table markdown, summary json을 쓴다.
13. `images._extract_embedded_images(...)`가 body 영역 이미지 파일만 저장한다.
14. 요청된 debug 파일들을 쓴 뒤 결과 dict를 반환한다.

## 리뷰 포인트

- 본문 품질 이슈는 먼저 `text.py`
- 표 병합/셀 정규화 이슈는 먼저 `tables.py`
- 디버그 JSON shape 변경은 `debug.py`와 `pipeline.py`
- 이미지 누락/과다 추출은 `images.py`
