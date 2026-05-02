# parsers/ — 문서 파서 모듈

## 역할

Samsung RAN 문서 유형별 구조화 파서.  
각 파서는 `parse_*()` 함수를 노출하고 `*ParseResult` dataclass로 청크 리스트를 반환.

## 파일 맵

| 파일 | 입력 | 출력 dataclass |
|---|---|---|
| `docx_parser.py` | Word (.docx) | `DocxParseResult` |
| `docx_config.py` | — | `DocxConfig` (헤딩 스타일, 컬럼 매핑) |
| `parameter_ref_parser.py` | Parameter Ref Excel | `ParameterRefParseResult` + `ParameterRecord` |
| `counter_ref_parser.py` | Counter Ref Excel | `CounterRefParseResult` + `CounterRecord` |
| `alarm_ref_parser.py` | Alarm Ref Excel | `AlarmRefParseResult` + `AlarmRecord` |
| `extractor/` | PDF | `ExtractionResult` (see below) |

## extractor/ 서브패키지

PDF 구조 추출 전용 패키지. `extractor.pipeline` 진입점.

| 파일 | 역할 |
|---|---|
| `pipeline.py` | 오케스트레이터 — `extract(pdf_path)` 반환 `ExtractionResult` |
| `text.py` | 폰트 메트릭 기반 헤딩 감지 + 워터마크 필터 |
| `tables.py` | 교차 페이지 테이블 병합 |
| `images.py` | 드로잉 클러스터 → 그림 bbox 검출 |
| `font_profile.py` | 문서 폰트 통계 수집 (헤딩 레벨 분류 기준) |
| `notes.py` | NOTE/WARNING/CAUTION 블록 추출 |
| `raw.py` | 디버그용 원시 페이지 데이터 덤프 |
| `shared.py` | 공통 bbox 지오메트리 + 텍스트 정제 유틸 |
| `debug.py` | HTML/JSON 렌더링 (개발 전용) |

## ParameterRecord 스키마

```python
@dataclass
class ParameterRecord:
    param_name: str       # kebab-case (e.g. handover-preparation-timer)
    yang_path: str        # YANG 경로 (/ 구분)
    feature_name: str     # FGR-XX0000 형식
    type: str
    default: str
    min: str
    max: str
    description: str

    @property
    def mo_path(self) -> list[str]: ...   # yang_path.split('/')
    @property
    def leaf_mo(self) -> str: ...         # mo_path[-1]
    def to_chunk_text(self) -> str: ...   # Milvus ingest용 포맷 텍스트
```

## 규약

- 모든 파서: 헤더 행 자동 감지 (최대 10행 탐색) — 실제 Samsung Excel 변종 대응
- `_COLUMN_ALIASES` 딕셔너리로 컬럼명 변형 흡수
- 샘플 파일: `data/samples/` (`git add -f` 필요 — root `.gitignore`의 `data/` 규칙 때문)
- 새 파서 추가 시 `data/samples/`에 샘플 xlsx + `tests/parsers/test_*.py` 동시 작성
