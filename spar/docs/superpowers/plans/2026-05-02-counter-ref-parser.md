# Counter Reference Parser 구현 계획

**작성일**: 2026-05-02  
**연관 Task**: Task 1.1 (문서 유형별 파서), Task 1.3 (Reference 청킹), Task 3.1 (Counter DB)

---

## 1. 문서 구조 분석

Samsung RAN Counter Reference Excel 문서의 컬럼 구조:

```
| 대 그룹명 | 중 그룹명 | 중 그룹 ID | 카운터명 | 설명 | 측정 주기 | 단위 | 값 범위(Min) | 값 범위(Max) |
```

### 핵심 파싱 난점: 병합 셀 (Merged Cells)

- **대 그룹명**: 기술 계층/그룹 (예: RRC, MAC, PHY) — 해당 그룹 전체 행에 걸쳐 세로 병합
- **중 그룹명 + 중 그룹 ID**: 하위 카운터 묶음 — 카운터 N개 행에 걸쳐 병합
- **개별 카운터 행**: 카운터명 / 설명 / 주기 / 단위 / 범위 — 각 행마다 존재

병합 셀의 특성:
- openpyxl `read_only=True`에서 병합 비-최상단 셀 → `None`
- `read_only=False` + `ws.merged_cells` 활용하거나, **"sticky" carry-forward** 전략 사용

**채택 전략**: `read_only=False` 로드 후 병합 셀 범위 기준으로 값 전파 (가장 안정적)  
fallback: None → 직전 비어있지 않은 값 carry-forward

---

## 2. 데이터 모델

```python
@dataclass
class CounterRecord:
    # 그룹 컨텍스트 (병합 셀에서 상속)
    large_group: str       # 대 그룹명 (기술 계층)
    mid_group: str         # 중 그룹명
    mid_group_id: str      # 중 그룹 ID
    # 개별 카운터 필드
    counter_name: str      # 카운터명 (필수)
    description: str       # 설명
    period: str            # 측정 주기 (예: 1분, 15분, ROP)
    unit: str              # 단위 (예: count, %, dBm)
    min_val: str           # 최솟값
    max_val: str           # 최댓값

    def to_chunk_text(self) -> str:
        """RAG ingest용 텍스트. 그룹 계층 포함."""

    def to_dict(self) -> dict:
        """DB/JSON 직렬화."""
```

### RAG chunk 텍스트 예시

```
Counter: CELL.UE.MaxConnectedNbr
Group: RRC > UE Statistics [ID: G-0042]
Description: Maximum number of simultaneously connected UEs during the measurement period.
Period: 15 min
Unit: count
Value Range: 0 ~ 1024
```

---

## 3. 컬럼 헤더 별칭 매핑

헤더명이 문서 버전마다 다를 수 있으므로 alias 매핑:

```python
_COLUMN_ALIASES = {
    # 대 그룹
    "대분류": "large_group", "large group": "large_group",
    "layer": "large_group", "technology group": "large_group",
    "group category": "large_group",
    # 중 그룹명
    "중분류": "mid_group", "group name": "mid_group",
    "counter group": "mid_group", "category": "mid_group",
    "sub group": "mid_group",
    # 중 그룹 ID
    "id": "mid_group_id", "group id": "mid_group_id",
    "counter id": "mid_group_id", "no": "mid_group_id",
    # 카운터명
    "counter name": "counter_name", "name": "counter_name",
    "카운터명": "counter_name", "counter": "counter_name",
    # 설명
    "description": "description", "desc": "description", "설명": "description",
    # 주기
    "period": "period", "measurement period": "period",
    "주기": "period", "interval": "period",
    # 단위
    "unit": "unit", "단위": "unit",
    # 범위
    "min": "min_val", "minimum": "min_val", "최솟값": "min_val",
    "max": "max_val", "maximum": "max_val", "최댓값": "max_val",
    "value range": "value_range",  # min/max 합산 컬럼 처리
}

REQUIRED_FIELDS = {"counter_name"}
GROUP_FIELDS = {"large_group", "mid_group", "mid_group_id"}
```

---

## 4. 파서 구현 전략

### 4-1. 병합 셀 처리

```python
def _expand_merged_cells(ws) -> dict[tuple, Any]:
    """병합 범위의 최상단 값을 모든 셀에 전파 → {(row, col): value}"""
    cell_values = {}
    for row in ws.iter_rows():
        for cell in row:
            cell_values[(cell.row, cell.column)] = cell.value

    for merge_range in ws.merged_cells.ranges:
        top_left_val = cell_values.get((merge_range.min_row, merge_range.min_col))
        for row in range(merge_range.min_row, merge_range.max_row + 1):
            for col in range(merge_range.min_col, merge_range.max_col + 1):
                cell_values[(row, col)] = top_left_val

    return cell_values
```

### 4-2. 헤더 탐색 (최대 10행)

- 헤더 행: `counter_name` 필드 매핑 존재하는 첫 행
- GROUP_FIELDS는 헤더에 없을 수도 있음 → 컬럼 위치로 fallback (1~3번 컬럼)

### 4-3. 데이터 행 파싱

```python
for row_idx in data_rows:
    counter_name = expanded[row_idx, col_map["counter_name"]]
    if not counter_name:
        continue  # 빈 행 스킵

    large_group = expanded.get((row_idx, col_map.get("large_group")), "")
    mid_group   = expanded.get((row_idx, col_map.get("mid_group")), "")
    mid_group_id = expanded.get((row_idx, col_map.get("mid_group_id")), "")
    # ...
```

### 4-4. value_range 단일 컬럼 처리

일부 문서는 "0 ~ 1024" 형식 단일 컬럼:
```python
def _parse_value_range(raw: str) -> tuple[str, str]:
    """'0 ~ 1024' → ('0', '1024')"""
    if "~" in raw:
        parts = raw.split("~", 1)
        return parts[0].strip(), parts[1].strip()
    return raw.strip(), ""
```

---

## 5. RAG 통합 설계

### 5-1. Milvus 메타데이터 필드 추가

`milvus_client.py` 스키마에 추가:
```python
FieldSchema("large_group",  DataType.VARCHAR, max_length=128),
FieldSchema("mid_group",    DataType.VARCHAR, max_length=256),
FieldSchema("mid_group_id", DataType.VARCHAR, max_length=64),
FieldSchema("counter_name", DataType.VARCHAR, max_length=256),
FieldSchema("period",       DataType.VARCHAR, max_length=64),
FieldSchema("unit",         DataType.VARCHAR, max_length=64),
```

### 5-2. 청킹 전략

Task 1.3 기준: **카운터 1개 = chunk 1개**

```python
# chunkers.py doc_type dispatch 추가
elif doc_type == "counter_ref":
    yield from _counter_ref_chunks(records)
```

chunk 텍스트는 `CounterRecord.to_chunk_text()` — 그룹 계층 포함으로 BM25 검색 시 `"RRC counter"`, `"G-0042"` 등 부분 매칭 가능.

### 5-3. BM25 키워드 활용

- `counter_name` → `keywords` ARRAY 자동 주입 (term_tagger.py)
- `mid_group_id` → keywords에 포함 (ID로 직접 검색 지원)

### 5-4. 라우터 연동

`regex_router.py` 패턴 추가:
```python
# 카운터 그룹 ID 패턴 (예: G-0042, C-123)
r"[A-Z]-\d{3,5}" → STRUCTURED_LOOKUP
# 카운터명 패턴 (점 구분 계층 이름)
r"\b[A-Z]{2,}(?:\.[A-Z]+){1,}\b" → STRUCTURED_LOOKUP (counter_name entity)
```

---

## 6. 샘플 파일

`data/samples/counter_ref_sample.xlsx` — openpyxl로 생성:

```
| 대 그룹명 | 중 그룹명          | 중 그룹 ID | Counter Name              | Description                    | Period | Unit  | Min | Max   |
|-----------|-------------------|-----------|---------------------------|--------------------------------|--------|-------|-----|-------|
| RRC       | UE Statistics     | G-0042    | CELL.UE.MaxConnectedNbr   | Max connected UEs in ROP       | 15 min | count | 0   | 1024  |
|           |                   |           | CELL.UE.AvgConnectedNbr   | Average connected UEs in ROP   | 15 min | count | 0   | 1024  |
|           |                   |           | CELL.UE.TotalSetupAttempt | Total RRC setup attempts       | 15 min | count | 0   | -     |
|           | Handover Stats    | G-0043    | CELL.HO.IntraFreqSucc     | Intra-freq HO success count    | 15 min | count | 0   | -     |
|           |                   |           | CELL.HO.IntraFreqFail     | Intra-freq HO failure count    | 15 min | count | 0   | -     |
| MAC       | PDSCH Throughput  | G-0101    | CELL.DL.SchThroughput     | Scheduled DL throughput        | 15 min | kbps  | 0   | -     |
|           |                   |           | CELL.DL.AvgMCS            | Average MCS index for PDSCH    | 15 min | -     | 0   | 28    |
```

병합 셀 재현 (대 그룹/중 그룹/ID 영역) — 파서가 실제 문서와 동일한 조건에서 검증 가능.

---

## 7. 산출물 목록

| 파일 | 설명 |
|------|------|
| `data/samples/counter_ref_sample.xlsx` | 병합 셀 포함 샘플 Excel |
| `src/spar/parsers/counter_ref_parser.py` | 파서 (CounterRecord, parse_counter_ref_excel) |
| `tests/parsers/test_counter_ref_parser.py` | 파서 유닛 테스트 |
| `scripts/ingest_counter_ref.py` | CLI: Excel → Milvus ingest |

추후 (Task 3.1 연동):
- `db/schema.sql` counters 테이블에 mid_group_id, period, unit 컬럼 추가
- KG 노드: Counter → (related_to) → Parameter, KPI

---

## 8. 구현 순서

1. `counter_ref_sample.xlsx` 생성 (병합 셀 포함, 검증 기준)
2. `counter_ref_parser.py` — `_expand_merged_cells` + `parse_counter_ref_excel`
3. `test_counter_ref_parser.py` — 병합 셀 carry-forward, value_range 분리, 빈 행 스킵
4. `chunkers.py` — `counter_ref` doc_type 디스패치 추가
5. `ingest_counter_ref.py` — CLI (parameter_ref 패턴 참조)
6. `milvus_client.py` / `routing.py` — 메타데이터 필드 + expr 필터 확장
7. `regex_router.py` — 카운터명/그룹 ID 패턴 추가
