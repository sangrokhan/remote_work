# Design: Entity Glossary + Two-Pass Ingest Pipeline

**Date:** 2026-05-03  
**Scope:** Task 1.3 (Reference 청커) + Task 1.6 확장 (Samsung 엔티티 사전)  
**Status:** Draft

---

## 1. 문제 정의

### 현재 파이프라인 (Single-Pass)

```
parse → chunk → tag_keywords(acronyms.json) → encode → Milvus
```

**문제점 3가지:**

1. **`acronyms.json`에 `keywords` 섹션 없음**  
   `load_keywords()` → 빈 set 반환 → `tag_keywords()` 아무 작동 안 함  
   청크의 `keywords[]` 필드 = 항상 빈 배열

2. **Samsung 도메인 엔티티명 사전에 없음**  
   파라미터명(`nrDlCellMaxTxPower`), 카운터명(`pmRrcConnEstabAtt`), 카운터 그룹명은  
   실제 Excel 파일 안에 있음 — ingest 전에 알 수 없음 (Chicken-and-egg)

3. **Task 1.3 Reference 문서 청커 미구현**  
   Parameter/Counter/Alarm Excel: 현재 ingest 경로 없음  
   파서(`to_chunk_text()`)는 이미 있으나 Chunk 변환 + Milvus 적재 미연결

---

## 2. 해결 설계: Two-Pass Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Pass A: Entity Glossary Build (최초 1회, 문서 추가 시 재실행)    │
│                                                                 │
│  scripts/build_entity_glossary.py                              │
│      ├─ scan parameter_ref Excel → param_names, yang_paths     │
│      ├─ scan counter_ref Excel → counter_names, groups          │
│      ├─ scan alarm_ref Excel → alarm_codes, alarm_names         │
│      └─ → dictionary/samsung_entities.json                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │ enriched entity set
┌─────────────────────────▼───────────────────────────────────────┐
│ Pass B: Main Ingest (기존 파이프라인, 엔리치드 tagging)          │
│                                                                 │
│  parse → chunk → tag_keywords(acronyms + samsung_entities)     │
│      → encode → Milvus                                          │
│                                                                 │
│  Reference 문서는 별도 경로:                                     │
│  Excel → parse_records → chunk_records (1 record = 1 chunk)    │
│      → tag_keywords → encode → Milvus                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 데이터 구조

### `dictionary/samsung_entities.json`

```json
{
  "parameter_names": ["nrDlCellMaxTxPower", "rachRootSequenceIndex", "..."],
  "yang_paths":      ["NRCellDU/nrDlCellMaxTxPower", "..."],
  "feature_names":   ["MIMO", "CA", "DSS", "..."],
  "counter_names":   ["pmRrcConnEstabAtt", "pmHoExecSuccNrNr", "..."],
  "counter_groups":  ["RRC", "Mobility", "Throughput", "..."],
  "alarm_codes":     ["4050", "4051", "..."],
  "alarm_names":     ["Cell Unavailable", "..."]
}
```

- 파일 생성 위치: `dictionary/samsung_entities.json`
- 생성 도구: `scripts/build_entity_glossary.py`
- 갱신 조건: 새 Excel 파일 추가 시 재실행

### Chunk 데이터 구조 (변경 없음)

기존 Milvus schema의 `keywords[]` ARRAY 필드 그대로 사용.  
tag_keywords 결과에 Samsung 엔티티명도 포함됨.

---

## 4. 컴포넌트 설계

### 4.1 `scripts/build_entity_glossary.py` (NEW)

```python
# 입력: data/ 하위 Excel 파일들 (parameter_ref, counter_ref, alarm_ref)
# 출력: dictionary/samsung_entities.json

def scan_parameter_refs(paths: list[Path]) -> dict[str, list[str]]
def scan_counter_refs(paths: list[Path]) -> dict[str, list[str]]
def scan_alarm_refs(paths: list[Path]) -> dict[str, list[str]]
def build_and_write(data_dir: Path, output_path: Path) -> None
```

- 기존 파서 재사용: `ParameterRefParser`, `CounterRefParser`, `AlarmRefParser`
- 노이즈 필터: 길이 < 3인 토큰 제외, 숫자만인 토큰 제외
- 중복 제거: set → sorted list

### 4.2 `src/spar/preprocessing/abbrev_mapper.py` (MODIFY)

추가할 함수:
```python
def load_entity_glossary(path: Path) -> dict[str, list[str]]:
    """samsung_entities.json 로드. 파일 없으면 빈 dict 반환 (graceful)."""

def get_all_keywords(acronyms: dict, entities: dict) -> set[str]:
    """acronyms.json의 global 키 + samsung_entities의 모든 엔티티명 합집합."""
```

변경 동작:
- `load_keywords(acronyms)` → `acronyms.get("keywords", {})` 대신 `get_all_keywords()` 사용
- `extract_terms()` 기존 로직 유지 (정규식 word boundary 매칭)

### 4.3 `src/spar/chunkers/reference_chunker.py` (NEW)

```python
def chunk_parameter_ref(records: list[ParameterRecord], source_doc: str) -> list[Chunk]
def chunk_counter_ref(records: list[CounterRecord], source_doc: str) -> list[Chunk]
def chunk_alarm_ref(records: list[AlarmRecord], source_doc: str) -> list[Chunk]
```

각 record → 1 Chunk 규칙:
- `text` = `record.to_chunk_text()`
- `doc_type` = `"parameter_ref"` | `"counter_ref"` | `"alarm_ref"`
- `keywords` = [] (tag_keywords가 이후 채움)
- `source_doc` = 파일명
- 파라미터: `mo_name` = `record.leaf_mo`, `yang_path` = `record.yang_path`
- 카운터: `mo_name` = `record.mid_group`, `section` = `record.large_group`

### 4.4 `src/spar/ingest/chunker.py` (MODIFY)

`chunk_document()` 디스패치 로직에 Reference 유형 추가:
```python
match doc_type:
    case "parameter_ref": return chunk_parameter_ref(records, source_doc)
    case "counter_ref":   return chunk_counter_ref(records, source_doc)
    case "alarm_ref":     return chunk_alarm_ref(records, source_doc)
    case _:               # 기존 md/fixed 청킹 유지
```

---

## 5. 파이프라인 변경 요약

### Before

```
run_ingest.py
    parse_document() → raw text (DOCX/PDF만 실질 지원)
    chunk_document() → Chunk[] (3GPP md만 완전 동작)
    tag_keywords()   → 아무것도 안 함 (keywords 빈 set)
```

### After

```
# Pass A (최초/문서 변경 시)
build_entity_glossary.py → dictionary/samsung_entities.json

# Pass B
run_ingest.py
    parse_document() → records (Excel) | raw text (DOCX/PDF)
    chunk_document() → Chunk[] (Excel: 1 record=1 chunk, 기존 유형 유지)
    tag_keywords(acronyms + entities) → keywords[] 실제 채워짐
    encode → Milvus
```

---

## 6. PRD 영향

| Task | 변경 내용 |
|------|-----------|
| Task 1.3 | `reference_chunker.py` 추가 → Reference 유형 청킹 완료 |
| Task 1.6 | `build_entity_glossary.py` 추가 → Samsung 엔티티 사전 구축 |
| Task 1.6 | `abbrev_mapper.py` 업데이트 → `get_all_keywords()` 실제 동작 |

---

## 7. 의존성 & 실행 순서

```
Excel 파일 배치
    → scripts/build_entity_glossary.py  (Pass A)
    → scripts/run_ingest.py <excel_files>  (Pass B — Reference)
    → scripts/run_ingest.py <docx/pdf_files>  (Pass B — 서술형 문서)
```

Pass A는 Pass B보다 반드시 선행. 단, 기존 3GPP md ingest는 Pass A 없이도 동작 유지 (graceful fallback).

---

## 8. 미결 결정사항

| 항목 | 옵션 A | 옵션 B | 현재 결정 |
|------|--------|--------|-----------|
| entity 파일 위치 | `acronyms.json`의 `keywords` 섹션에 병합 | 별도 `samsung_entities.json` | **B** — 관심사 분리 |
| 노이즈 필터 | 길이 ≥ 3 + 영문자 포함 | LLM 기반 필터 | **규칙 기반** (빠름) |
| Pass A 트리거 | 수동 실행 | `run_ingest.py`에서 자동 감지 | **수동** (Phase 1 범위) |
