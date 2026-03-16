# Fixture-Driven PDF Roundtrip Design

**Context**

현재 샘플 구성은 `sample_generator.py` 안에 하드코딩된 텍스트와 표 데이터를 직접 그려서 `sample.pdf`를 만들고, `extractor.py`의 결과를 별도 로직으로 확인하는 방식이다. 이 구조에는 세 가지 문제가 있다.

1. PDF 생성에 사용된 원본 정보와 검증 기준이 코드 여러 곳에 흩어져 있다.
2. 시각 샘플을 바꾸는 과정에서 검증 기준까지 흔들리기 쉽다.
3. 실제 목표인 "입력 정보가 PDF를 거쳐 다시 동일한 구조로 추출되는가"를 직접 검증하지 못한다.

**Goal**

PDF 생성과 추출 검증의 기준을 하나의 fixture 파일로 통합한다. 이 fixture는 사람이 읽을 수 있는 `json`, `md`, 또는 `txt` 형식 중 하나를 사용하며, 최소한 아래 정보를 포함해야 한다.

- 문서 본문 텍스트
- 표 정의
- 병합 셀 정보
- 페이지를 넘는 표 continuation 기대 형태
- 긴 단일 row가 다음 페이지로 이어지는 기대 형태

최종적으로는 다음 흐름이 성립해야 한다.

1. fixture 파일을 읽는다.
2. `sample_generator.py`가 fixture만을 입력으로 받아 PDF를 생성한다.
3. `extractor.py`가 PDF를 다시 추출한다.
4. 검증 단계에서 추출 결과를 fixture의 원본 정보와 비교한다.
5. 원본과 추출 결과가 구조적으로 동일하다고 판단되면 통과한다.

**Recommended Format**

기본 포맷은 `JSON`을 권장한다.

이유:

- 표, 셀, 병합 범위, 페이지 단위 기대값을 구조적으로 표현하기 쉽다.
- PDF 생성 입력과 검증 입력을 같은 객체 모델로 재사용할 수 있다.
- 추후 fixture를 여러 개로 늘릴 때 자동화에 유리하다.

권장 파일 예시:

- `fixtures/demo_document.json`

권장 최상위 필드:

```json
{
  "document": {
    "title": "sample",
    "pagesize": "letter"
  },
  "body": [
    {"text": "Chapter 1: Deep Structure Verification", "indent": 0}
  ],
  "tables": [
    {
      "id": "stage",
      "columns": ["Stage", "Team", "Notes"],
      "rows": [
        ["Phase A", "Discovery", "Kickoff scope lock"],
        ["", "Design", "UX skeleton review"]
      ],
      "merged_cells": [
        {"column": 0, "start_row": 0, "end_row": 6}
      ]
    }
  ],
  "expectations": {
    "page_count": 3,
    "table_headers_repeat_on_page_break": true,
    "merged_first_column_has_no_left_border": true,
    "split_rows": [
      {
        "table_id": "stage",
        "row_key": {"column": 1, "value": "Legal"},
        "starts_on_page": 2,
        "continues_on_page": 3
      }
    ]
  }
}
```

**Architecture**

구조는 세 레이어로 나눈다.

1. Fixture loader
   - fixture 파일을 읽고 정규화된 Python 데이터 구조로 변환한다.
   - 문서 본문, 표, 병합 정보, 검증 기대값을 한 곳에서 제공한다.

2. PDF builder
   - `sample_generator.py`는 하드코딩된 데모 상수 대신 fixture 데이터만 사용한다.
   - 표는 자동 페이지네이션으로 렌더링한다.
   - 표가 페이지를 넘기면 헤더를 다시 출력한다.
   - 병합 셀은 페이지가 바뀌어도 continuation 형태로 유지한다.
   - 긴 단일 row는 남은 공간이 부족할 때 row 내부 line 단위로 분할 렌더링한다.

3. Roundtrip verifier
   - `extractor.py` 결과를 fixture와 비교 가능한 구조로 정규화한다.
   - `verify.py`는 "텍스트가 일부 포함되는가" 수준이 아니라 fixture 기반 구조 비교를 수행한다.
   - 표의 continuation row나 페이지 분할 row는 비교 전에 다시 병합한다.

**Data Model Requirements**

fixture는 아래 정보를 표현할 수 있어야 한다.

- 셀 텍스트
- intentional multiline 내용
- 병합된 셀 범위
- 표 식별자
- 페이지 분할이 필요한 긴 row 식별자
- 페이지별 기대 조건

표 row 비교는 "보이는 조각"이 아니라 "논리 row" 기준이어야 한다. 예를 들어 `Legal` row가 2페이지와 3페이지에 나뉘어 보이더라도, 검증 단계에서는 하나의 row로 다시 합쳐서 fixture 원본 row와 비교한다.

**Rendering Rules**

이 문서의 기준 렌더링 규칙은 다음과 같다.

- 표는 자동 페이지네이션으로만 분할한다.
- 표를 맞추기 위해 수동으로 페이지 fragment를 하드코딩하지 않는다.
- 표가 다음 페이지로 이어지면 새 페이지 첫 부분에 헤더를 다시 그린다.
- 병합된 첫 컬럼 셀은 continuation 페이지에서도 같은 병합 영역으로 취급한다.
- 병합된 첫 컬럼 구간에서는 왼쪽 외곽 세로선을 그리지 않는다.
- 긴 단일 row는 가능한 한 row 내부 line 단위로 분할하고, 다음 페이지에서 같은 논리 row의 나머지 내용이 이어져야 한다.

**Extraction Rules**

추출 결과는 사람이 바로 사용할 수 있는 표 형식으로 정리되어야 한다.

- 최종 표 출력은 Markdown table 형태를 유지한다.
- layout wrap으로 생긴 줄바꿈은 cell 내부에서 다시 병합한다.
- intentional multiline 또는 bullet 성격의 내용은 `<br>`로 유지할 수 있다.
- 페이지 경계에서 잘린 row는 검증 전에 이전 row와 다시 병합한다.
- 페이지마다 반복된 header row는 중복 제거 후 하나의 논리 표로 합친다.

**Verification Rules**

검증은 두 단계로 나눈다.

1. Visual verification
   - 렌더된 PDF 페이지 이미지를 확인해 헤더 재출력, 병합 셀 continuation, 경계선 규칙, row split을 시각적으로 점검한다.

2. Structural verification
   - fixture 원본과 extractor 출력의 구조를 비교한다.
   - 비교 대상:
     - 본문 주요 라인
     - 표 수
     - 표 컬럼 정의
     - 각 논리 row의 cell 값
     - 병합 continuation으로 인해 분할된 row의 재조합 결과

**Non-Goals**

- fixture를 기반으로 임의의 복잡한 문서 레이아웃 편집기까지 만드는 것
- OCR 수준의 완전 일반화된 PDF roundtrip
- 시각 렌더링 차이를 픽셀 단위로 완전히 동일하게 맞추는 것

**Migration Plan**

1. fixture 파일을 새로 만든다.
2. 기존 `DEMO_*` 상수를 fixture loader 기반 구조로 옮긴다.
3. `create_demo_pdf()`가 fixture를 읽도록 바꾼다.
4. `verify.py`를 fixture 기반 구조 비교로 바꾼다.
5. 테스트는 fixture를 기준으로 PDF 생성과 추출 결과를 검증하도록 정리한다.

**Acceptance Criteria**

- PDF 생성 입력이 fixture 파일 하나로 수렴한다.
- `sample_generator.py`가 fixture만으로 `sample.pdf`를 생성한다.
- `extractor.py` 출력이 fixture 기준 논리 데이터와 일치한다.
- `Legal` 같은 split row는 추출 후 이전 row와 재병합되어 원본 fixture row와 동일하게 비교된다.
- 병합 셀, 헤더 반복, 왼쪽 외곽선 생략 규칙이 시각적으로 확인된다.
