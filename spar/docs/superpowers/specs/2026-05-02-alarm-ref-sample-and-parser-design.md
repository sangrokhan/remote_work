# Alarm Reference Sample + Parser + Search Integration — Design

**Date**: 2026-05-02
**Branch**: `feat/alarm-ref-sample-and-parser`
**Status**: Draft

---

## 1. 목적

Samsung RAN 알람 도메인 처리를 위한 첫 단계 구축:

- 알람 레퍼런스 엑셀(키 ID + 명칭) 샘플 생성
- 해당 엑셀을 파싱하는 모듈 추가 (`alarm_ref_parser`)
- alarm_id 기반 검색/매칭 전략을 라우터/리트리버 레이어에 wiring (in-memory `AlarmIndex`)
- 향후 알람 PDF(설명/조치) 인제스트 시 join 키가 될 alarm_id 표준화

**비범위**: PDF 파서, Milvus 실제 인제스트 스크립트 수정, LLM 라우터 변경.

---

## 2. 산출물

| # | 경로 | 설명 |
|---|---|---|
| 1 | `data/samples/alarm_excel_ref_sample.xlsx` | 12행 알람 레퍼런스 샘플 |
| 2 | `src/spar/parsers/alarm_ref_parser.py` | 엑셀 파서 + `AlarmRecord` dataclass |
| 3 | `src/spar/retrieval/alarm_index.py` | in-memory `AlarmIndex` + 싱글톤 `get_alarm_index()` |
| 4 | `src/spar/retrieval/routing.py` | 기존 파일 — alarm_code → AlarmIndex 직접 lookup 분기 추가 |
| 5 | `tests/unit/parsers/test_alarm_ref_parser.py` | 파서 round-trip / 헤더 별칭 / skip / keywords 형태 |
| 6 | `tests/unit/retrieval/test_alarm_index.py` | 인덱스 lookup / not-found / 케이스 정규화 |
| 7 | `scripts/gen_alarm_sample.py` | 샘플 xlsx 생성 스크립트 (재현 가능) |
| 8 | 문서 업데이트 | `README.md`, `AGENTS.md`, `docs/prd.md` (CLAUDE.md 규약) |

---

## 3. 엑셀 스키마

| 헤더 (Excel) | 필드 (Python) | 예시 | 비고 |
|---|---|---|---|
| Alarm ID | `alarm_id` | `ALM-1003` | 필수. regex_router 정규식 `\bALM-(\d+)\b` 호환 |
| Alarm Name | `alarm_name` | `Cell Down` | 필수. 사람이 읽는 명칭 |
| Severity | `severity` | `Critical` | Critical/Major/Minor/Warning |
| Category | `category` | `Radio` | 도메인 그룹 (Radio/Transport/HW/SW/...) |
| Module | `module` | `gNB-DU` | 발생 노드/서브시스템 |

**별칭 매핑** (`_COLUMN_ALIASES`, lowercase 비교):
- `alarm id`, `id`, `alarm code` → `alarm_id`
- `alarm name`, `name` → `alarm_name`
- `severity`, `level` → `severity`
- `category`, `group` → `category`
- `module`, `node`, `subsystem` → `module`

**필수 컬럼**: `{alarm_id, alarm_name}`

**예약 필드 (코드에는 두되 엑셀에는 비움)**: `pdf_ref` — 향후 알람 PDF 파서가 alarm_id로 join 시 채울 자리.

---

## 4. AlarmRecord API

```python
@dataclass
class AlarmRecord:
    alarm_id: str          # 정규화: 대문자 (ALM-1003)
    alarm_name: str
    severity: str = ""
    category: str = ""
    module: str = ""
    pdf_ref: str = ""

    def to_chunk_text(self) -> str: ...
    def to_keywords(self) -> list[str]: ...
    def to_dict(self) -> dict[str, Any]: ...
```

`to_chunk_text()` 출력 예:
```
Alarm: ALM-1003 — Cell Down
Severity: Critical
Category: Radio
Module: gNB-DU
```

`to_keywords()` 출력 예:
```python
["ALM-1003", "Cell Down", "Critical", "Radio", "gNB-DU"]
```
(빈 값은 제외; Milvus 기존 `keywords` ARRAY 필드에 그대로 주입 가능)

---

## 5. AlarmIndex API

```python
class AlarmIndex:
    def __init__(self, records: list[AlarmRecord]): ...
    def lookup(self, alarm_id: str) -> AlarmRecord | None:
        """대소문자 무관, ALM-NNNN 정규화 후 dict lookup."""
    def search_by_name(self, query: str) -> list[AlarmRecord]:
        """alarm_name 부분일치 (case-insensitive). lexical fallback."""
    def __len__(self) -> int: ...

def get_alarm_index(path: str | Path | None = None) -> AlarmIndex:
    """싱글톤. 첫 호출 시 path 또는 기본 샘플 경로 로드."""
```

기본 경로: 환경변수 `SPAR_ALARM_REF_PATH` > `data/samples/alarm_excel_ref_sample.xlsx`.

---

## 6. routing.py 통합

`route_to_doc_type()` (또는 동급 함수) 내 `alarm_code` 엔티티 처리:

```python
if "alarm_code" in entities:
    rec = get_alarm_index().lookup(entities["alarm_code"])
    if rec:
        # STRUCTURED_LOOKUP 직답 후보로 우선 반환
        return RoutingResult(
            doc_type="alarm_ref",
            structured_record=rec.to_dict(),
            keywords=rec.to_keywords(),  # Milvus 보강 검색용
            ...
        )
```

기존 `RoutingResult` 스키마에 옵션 필드 추가:
- `structured_record: dict | None = None`
- `keywords: list[str] = field(default_factory=list)`

(이미 유사 필드 있으면 재사용 — 구현 시 확인.)

**Pipeline 영향**: `pipeline/` 노드 변경 없음 (이번 작업 비범위). 라우팅 결과만 풍부해지며 후속 노드에서 활용은 추후 작업.

---

## 7. 흐름도

```
query "ALM-1003 처리법"
  │
  ├─► regex_router → entities={alarm_code: "ALM-1003"}, route=DIAGNOSTIC
  │
  ├─► routing.py
  │     ├─► AlarmIndex.lookup("ALM-1003") → AlarmRecord
  │     └─► RoutingResult(doc_type=alarm_ref,
  │                        structured_record={...},
  │                        keywords=[...])
  │
  └─► (후속) Milvus hybrid search (keywords 필터) + reranker
```

---

## 8. 테스트 전략

### test_alarm_ref_parser.py

- `test_parse_sample_round_trip` — 샘플 xlsx 12행 모두 파싱
- `test_header_alias_resolution` — `Alarm Code` / `Name` 별칭으로도 인식
- `test_required_field_missing_skipped` — alarm_id 빈 행 skip + warning
- `test_to_keywords_excludes_blanks` — severity 빈 알람의 keywords 출력 검증
- `test_alarm_id_normalized_uppercase` — 소문자 입력도 ALM-NNNN 대문자로

### test_alarm_index.py

- `test_lookup_hit` / `test_lookup_miss`
- `test_lookup_case_insensitive`
- `test_search_by_name_partial`
- `test_singleton_caches`

---

## 9. 샘플 데이터 (12행 안)

| Alarm ID | Alarm Name | Severity | Category | Module |
|---|---|---|---|---|
| ALM-1001 | Cell Out of Service | Critical | Radio | gNB-DU |
| ALM-1002 | Link Down | Critical | Transport | gNB-CU |
| ALM-1003 | Cell Down | Critical | Radio | gNB-DU |
| ALM-1004 | High Temperature | Major | HW | RU |
| ALM-1005 | Fan Failure | Major | HW | BBU |
| ALM-1006 | Clock Sync Loss | Major | Transport | gNB-DU |
| ALM-1007 | License Expiring | Minor | SW | OAM |
| ALM-1008 | High CPU Usage | Minor | SW | gNB-CU |
| ALM-1009 | PRACH Anomaly | Minor | Radio | gNB-DU |
| ALM-1010 | Backup Failed | Warning | SW | OAM |
| ALM-1011 | Config Mismatch | Warning | SW | gNB-CU |
| ALM-1012 | Power Redundancy Lost | Major | HW | BBU |

(vendor-neutral, 일반 LTE/NR 알람 도메인. 실제 Samsung 알람 코드와 무관.)

---

## 10. PRD 갱신

`docs/prd.md`에 Phase 1 산출물로 추가:
- Task 1.1.x — alarm_ref Excel parser + AlarmIndex (이 PR)
- 향후 Task — alarm_ref PDF parser (alarm_id join)

CLAUDE.md 규약대로 README/AGENTS 디렉토리 맵 갱신.

---

## 11. Open Questions (없음)

설계상 결정사항:
- alarm_id 형식 = `ALM-NNNN` 고정 → regex_router 호환 ✅
- AlarmIndex 싱글톤 + 환경변수 override ✅
- pdf_ref 예약 필드 ✅
- LLM 라우터/pipeline 변경 없음 ✅
