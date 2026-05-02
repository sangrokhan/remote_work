# Word 문서 파서 설계 (DocxParser)

**날짜:** 2026-05-02  
**범위:** Task 1.1 — 문서 유형별 파서 개발 (Word .docx)  
**상태:** 설계 확정

---

## 1. 목표

Samsung RAN 문서 중 Word(.docx) 형식을 파싱하여:
- 메인 Markdown 파일 출력 (섹션 텍스트 + placeholder)
- 표 → 개별 CSV 파일
- 이미지 → 개별 이미지 파일 (PNG/JPEG 등)

각 표/이미지 파일은 소속 heading 컨텍스트 기반 명칭 부여 및 메타데이터 기록.

---

## 2. 라이브러리 선택

**`python-docx`** (직접 파싱)

- `paragraph.style.name` 으로 `"Heading 1"` ~ `"Heading N"` 직접 접근 → heading depth 커스텀 가능
- 표: `doc.tables` 직접 순회
- 이미지: `paragraph._element.xpath('.//a:blip')` + `doc.part.related_parts` 바이너리 추출
- 의존성 최소, 경량 단일 모듈 (기존 `excel_loader.py` 패턴 일치)

---

## 3. 설정 (DocxParseConfig)

```python
@dataclass
class DocxParseConfig:
    heading_depth: int = 2        # 섹션 경계로 인식할 최대 heading 레벨 (1~3)
    output_dir: Path = Path("output")
    slugify_max_len: int = 30     # 파일명에 사용하는 섹션 타이틀 최대 문자수
```

`heading_depth=2` → `Heading 1`, `Heading 2` 만 섹션 경계로 취급.  
`Heading 3` 이하는 Markdown `###` 으로 출력하되 파일 귀속 섹션은 변경하지 않음.

---

## 4. 출력 구조

```
{output_dir}/
├── {doc_stem}.md           # 메인 Markdown
├── tables/
│   ├── Table_{section_slug}_{seq}.csv
│   └── ...
└── images/
    ├── Fig_{section_slug}_{seq}.{ext}
    └── ...
```

### 4.1 파일명 규칙

- `section_slug`: 현재 섹션 heading 타이틀을 slugify (공백→`-`, 특수문자 제거, `slugify_max_len` truncate)
- `seq`: 해당 섹션 내 표/이미지 순서 (1-based). 섹션 변경 시 리셋.
- 예: `Table_System-Overview_1.csv`, `Fig_Intro_2.png`

### 4.2 CSV 파일 헤더 메타데이터

각 CSV 첫 두 줄:
```
# section: Introduction > System Overview
# source: MyDocument.docx
```

### 4.3 이미지 파일 메타데이터

동명 `.meta` 파일 (또는 EXIF 불가 시 `.txt` sidecar):
```
section: Introduction > System Overview
source: MyDocument.docx
seq: 1
```

---

## 5. 파싱 흐름

```
Document.load(path)
    │
    ├── paragraph 순회 (문서 순서 유지)
    │     ├── style == "Heading N" (N ≤ heading_depth)
    │     │     → 섹션 컨텍스트 갱신, seq 리셋, Markdown heading 출력
    │     ├── style == "Heading N" (N > heading_depth)
    │     │     → Markdown heading 출력만 (섹션 컨텍스트 유지)
    │     ├── 이미지 포함 paragraph (a:blip xpath)
    │     │     → 이미지 추출 → 파일 저장 → placeholder 삽입
    │     └── 일반 텍스트
    │           → Markdown 본문 append
    │
    └── doc.tables 위치 매핑
          → 각 table을 paragraph 순서상 위치에 삽입
          → CSV 저장 → placeholder 삽입
```

### 5.1 테이블 위치 매핑

`python-docx`는 `doc.paragraphs`와 `doc.tables`가 별도 리스트. XML 순서 유지를 위해:
```python
# document body의 child element 순서로 통합 순회
for child in doc.element.body:
    if child.tag ends with 'p':   # paragraph
    if child.tag ends with 'tbl': # table
```

### 5.2 Markdown placeholder

```markdown
<!-- TABLE: Table_System-Overview_1 -->

<!-- IMAGE: Fig_Intro_2 -->
```

---

## 6. 모듈 구조

```
src/spar/parsers/
├── __init__.py
├── docx_config.py      # DocxParseConfig dataclass
└── docx_parser.py      # DocxParser 클래스

scripts/
└── parse_docx.py       # CLI: --file, --output, --heading-depth
```

### DocxParser 인터페이스

```python
class DocxParser:
    def __init__(self, config: DocxParseConfig) -> None: ...
    def parse(self, docx_path: Path) -> ParseResult: ...

@dataclass
class ParseResult:
    markdown: str
    tables: list[ExtractedTable]    # path, section_path, seq
    images: list[ExtractedImage]    # path, section_path, seq, ext
```

---

## 7. CLI

```bash
python scripts/parse_docx.py \
  --file path/to/document.docx \
  --output output/ \
  --heading-depth 2
```

출력:
```
Parsed: output/document.md
Tables: 5 → output/tables/
Images: 12 → output/images/
```

---

## 8. 테스트 전략

- fixture: 최소 `.docx` (heading 2레벨, 표 1개, 이미지 1개) `python-docx`로 프로그래매틱 생성
- `test_heading_depth`: depth 변경 시 섹션 경계 변화 검증
- `test_table_csv`: CSV 내용 + 메타데이터 헤더 검증
- `test_image_extract`: 이미지 파일 존재 + sidecar 메타 검증
- `test_seq_reset`: 섹션 변경 시 seq=1 리셋 검증
- `test_placeholder`: Markdown 내 `<!-- TABLE: ... -->` 삽입 위치 검증

---

## 9. 미결 사항

- merged cell 표: 현재 설계는 단순 행/열 순회. 복잡한 merged cell은 빈 셀로 처리 (추후 개선)
- 스캔 이미지 내 텍스트: OCR 불포함 (Task 1.1 별도 항목)
- Heading style 이름이 커스텀된 문서 (`"제목 1"` 등): `heading_depth` 외 `heading_style_names` 설정 추가 검토
