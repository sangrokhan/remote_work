# Entity Glossary + Two-Pass Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Samsung 도메인 엔티티(파라미터/카운터/알람)를 ingest 전에 사전으로 구축하고, Reference Excel을 1 record = 1 chunk로 Milvus에 적재하며, tag_keywords()가 실제로 동작하도록 한다.

**Architecture:** 두 단계 파이프라인. Pass A(`build_entity_glossary.py`)가 Excel 파일을 스캔해 `dictionary/samsung_entities.json`을 생성. Pass B(기존 `run_ingest.py`)가 enriched 사전으로 tagging하며 Reference Excel도 청킹·적재. `abbrev_mapper.py`는 두 사전을 합쳐 실제 keyword 매칭을 수행.

**Tech Stack:** Python 3.12, openpyxl (기존 파서), pytest, 기존 `ParameterRefParser`/`CounterRefParser`/`AlarmRefParser`

**Spec:** `docs/superpowers/specs/2026-05-03-entity-glossary-two-pass-ingest-design.md`

---

## File Map

| 파일 | 역할 | 상태 |
|------|------|------|
| `scripts/build_entity_glossary.py` | Pass A: Excel 스캔 → `samsung_entities.json` 생성 | NEW |
| `dictionary/samsung_entities.json` | Samsung 도메인 엔티티 사전 (generated) | NEW |
| `src/spar/preprocessing/abbrev_mapper.py` | `load_entity_glossary()` + `get_all_keywords()` 추가 | MODIFY |
| `src/spar/chunkers/reference_chunker.py` | Parameter/Counter/Alarm record → Chunk 변환 | NEW |
| `src/spar/ingest/chunker.py` | doc_type 디스패치에 reference 유형 추가 | MODIFY |
| `tests/unit/preprocessing/test_entity_glossary.py` | `load_entity_glossary`, `get_all_keywords` 테스트 | NEW |
| `tests/unit/chunkers/test_reference_chunker.py` | reference_chunker 테스트 | NEW |
| `tests/integration/test_build_entity_glossary.py` | build_entity_glossary end-to-end 테스트 | NEW |

---

## Task 1: `load_entity_glossary` + `get_all_keywords` (abbrev_mapper 확장)

**Files:**
- Modify: `src/spar/preprocessing/abbrev_mapper.py`
- Create: `tests/unit/preprocessing/test_entity_glossary.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/unit/preprocessing/test_entity_glossary.py
import json
from pathlib import Path
import pytest
from spar.preprocessing.abbrev_mapper import load_entity_glossary, get_all_keywords

def test_load_entity_glossary_returns_dict(tmp_path):
    entities = {
        "parameter_names": ["nrDlCellMaxTxPower", "rachRootSequenceIndex"],
        "counter_names": ["pmRrcConnEstabAtt"],
        "counter_groups": ["RRC"],
        "alarm_codes": ["4050"],
        "alarm_names": ["Cell Unavailable"],
        "yang_paths": ["NRCellDU/nrDlCellMaxTxPower"],
        "feature_names": ["MIMO"],
    }
    p = tmp_path / "samsung_entities.json"
    p.write_text(json.dumps(entities))
    result = load_entity_glossary(p)
    assert result["parameter_names"] == ["nrDlCellMaxTxPower", "rachRootSequenceIndex"]

def test_load_entity_glossary_missing_file_returns_empty(tmp_path):
    result = load_entity_glossary(tmp_path / "nonexistent.json")
    assert result == {}

def test_get_all_keywords_combines_acronyms_and_entities():
    acronyms = {
        "global": {
            "HO": {"expansion": "Handover", "variants": []},
            "CA": {"expansion": "Carrier Aggregation", "variants": []},
        }
    }
    entities = {
        "parameter_names": ["nrDlCellMaxTxPower"],
        "counter_names": ["pmRrcConnEstabAtt"],
        "counter_groups": ["RRC"],
        "alarm_codes": ["4050"],
        "alarm_names": [],
        "yang_paths": [],
        "feature_names": [],
    }
    keywords = get_all_keywords(acronyms, entities)
    assert "HO" in keywords
    assert "CA" in keywords
    assert "nrDlCellMaxTxPower" in keywords
    assert "pmRrcConnEstabAtt" in keywords
    assert "RRC" in keywords
    # 알람 코드는 숫자만 → 포함 안 됨
    assert "4050" not in keywords

def test_get_all_keywords_empty_entities():
    acronyms = {"global": {"HO": {"expansion": "Handover", "variants": []}}}
    keywords = get_all_keywords(acronyms, {})
    assert "HO" in keywords
```

- [ ] **Step 2: 실패 확인**

```bash
cd /home/han/.openclaw/workspace/remote_work/spar
python -m pytest tests/unit/preprocessing/test_entity_glossary.py -v 2>&1 | tail -20
```

Expected: ImportError 또는 AttributeError (`load_entity_glossary` 없음)

- [ ] **Step 3: `abbrev_mapper.py`에 두 함수 추가**

`src/spar/preprocessing/abbrev_mapper.py` 파일 끝 (또는 `load_keywords` 근처)에 추가:

```python
def load_entity_glossary(path: "Path") -> dict[str, list[str]]:
    """samsung_entities.json 로드. 파일 없으면 빈 dict 반환."""
    try:
        return cast("dict[str, list[str]]", json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return {}


_NOISE_PATTERN = re.compile(r"^\d+$|^.{1,2}$")


def get_all_keywords(acronyms: dict, entities: dict[str, list[str]]) -> set[str]:
    """acronyms global 키 + samsung_entities 모든 값 합집합. 노이즈(숫자만, 2자 이하) 제외."""
    keywords: set[str] = set(acronyms.get("global", {}).keys())
    for values in entities.values():
        for term in values:
            if term and not _NOISE_PATTERN.match(term):
                keywords.add(term)
    return keywords
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/unit/preprocessing/test_entity_glossary.py -v 2>&1 | tail -20
```

Expected: 4 PASSED

- [ ] **Step 5: 커밋**

```bash
git add src/spar/preprocessing/abbrev_mapper.py tests/unit/preprocessing/test_entity_glossary.py
git commit -m "feat(abbrev): add load_entity_glossary + get_all_keywords for Samsung entity support"
```

---

## Task 2: `reference_chunker.py` (Parameter/Counter/Alarm → Chunk 변환)

**Files:**
- Create: `src/spar/chunkers/reference_chunker.py`
- Create: `tests/unit/chunkers/test_reference_chunker.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/unit/chunkers/test_reference_chunker.py
import pytest
from spar.parsers.parameter_ref_parser import ParameterRecord
from spar.parsers.counter_ref_parser import CounterRecord
from spar.parsers.alarm_ref_parser import AlarmRecord
from spar.chunkers.reference_chunker import (
    chunk_parameter_ref,
    chunk_counter_ref,
    chunk_alarm_ref,
)


def test_chunk_parameter_ref_one_record_one_chunk():
    record = ParameterRecord(
        param_name="nrDlCellMaxTxPower",
        yang_path="NRCellDU/nrDlCellMaxTxPower",
        feature_name="MIMO",
        type="int32",
        default="23",
        min="0",
        max="40",
        description="Maximum downlink transmission power",
    )
    chunks = chunk_parameter_ref([record], source_doc="param_ref_v6.xlsx")
    assert len(chunks) == 1
    c = chunks[0]
    assert "nrDlCellMaxTxPower" in c.text
    assert c.doc_type == "parameter_ref"
    assert c.source_doc == "param_ref_v6.xlsx"
    assert c.mo_name == "NRCellDU"  # leaf_mo


def test_chunk_parameter_ref_empty_returns_empty():
    assert chunk_parameter_ref([], source_doc="x.xlsx") == []


def test_chunk_counter_ref_one_record_one_chunk():
    record = CounterRecord(
        counter_name="pmRrcConnEstabAtt",
        large_group="RRC",
        mid_group="Connection",
        mid_group_id="RRC.01",
        description="RRC connection establishment attempts",
        period="15min",
        unit="count",
        min_val="0",
        max_val="",
        value_range="",
    )
    chunks = chunk_counter_ref([record], source_doc="counter_ref_v6.xlsx")
    assert len(chunks) == 1
    c = chunks[0]
    assert "pmRrcConnEstabAtt" in c.text
    assert c.doc_type == "counter_ref"
    assert c.mo_name == "Connection"  # mid_group


def test_chunk_alarm_ref_one_record_one_chunk():
    record = AlarmRecord(
        alarm_code="4050",
        alarm_name="Cell Unavailable",
        severity="Critical",
        description="Cell is not available",
        probable_causes=["Hardware failure"],
        recommended_actions=["Check hardware"],
    )
    chunks = chunk_alarm_ref([record], source_doc="alarm_ref_v6.xlsx")
    assert len(chunks) == 1
    c = chunks[0]
    assert "4050" in c.text or "Cell Unavailable" in c.text
    assert c.doc_type == "alarm_ref"
```

- [ ] **Step 2: 실패 확인**

```bash
python -m pytest tests/unit/chunkers/test_reference_chunker.py -v 2>&1 | tail -20
```

Expected: ImportError (`reference_chunker` 없음)

- [ ] **Step 3: `reference_chunker.py` 구현**

알람 파서의 AlarmRecord 구조 먼저 확인:
```bash
grep -n "class AlarmRecord\|def to_chunk_text\|alarm_code\|alarm_name\|severity" src/spar/parsers/alarm_ref_parser.py | head -20
```

CounterRecord 구조 확인:
```bash
grep -n "class CounterRecord\|def to_chunk_text\|mid_group\|large_group" src/spar/parsers/counter_ref_parser.py | head -20
```

확인 후 작성:

```python
# src/spar/chunkers/reference_chunker.py
"""Reference 문서(Parameter/Counter/Alarm) Excel record → Chunk 변환. 1 record = 1 chunk."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spar.parsers.alarm_ref_parser import AlarmRecord
    from spar.parsers.counter_ref_parser import CounterRecord
    from spar.parsers.parameter_ref_parser import ParameterRecord


@dataclass
class Chunk:
    text: str
    doc_type: str
    source_doc: str
    mo_name: str = ""
    yang_path: str = ""
    section: str = ""
    keywords: list[str] = field(default_factory=list)


def chunk_parameter_ref(records: list[ParameterRecord], source_doc: str) -> list[Chunk]:
    return [
        Chunk(
            text=r.to_chunk_text(),
            doc_type="parameter_ref",
            source_doc=source_doc,
            mo_name=r.leaf_mo,
            yang_path=r.yang_path,
        )
        for r in records
    ]


def chunk_counter_ref(records: list[CounterRecord], source_doc: str) -> list[Chunk]:
    return [
        Chunk(
            text=r.to_chunk_text(),
            doc_type="counter_ref",
            source_doc=source_doc,
            mo_name=r.mid_group,
            section=r.large_group,
        )
        for r in records
    ]


def chunk_alarm_ref(records: list[AlarmRecord], source_doc: str) -> list[Chunk]:
    return [
        Chunk(
            text=r.to_chunk_text(),
            doc_type="alarm_ref",
            source_doc=source_doc,
        )
        for r in records
    ]
```

> **Note:** `CounterRecord`와 `AlarmRecord`에 `to_chunk_text()` 없으면 Task 2 Step 3 전에 추가해야 함. Step 2 실패 메시지 확인 후 판단.

- [ ] **Step 4: `CounterRecord.to_chunk_text()` 확인 및 추가 (필요 시)**

```bash
grep -n "to_chunk_text" src/spar/parsers/counter_ref_parser.py src/spar/parsers/alarm_ref_parser.py
```

없으면 각 파서에 추가:

`counter_ref_parser.py`의 `CounterRecord` 클래스:
```python
def to_chunk_text(self) -> str:
    lines = [f"Counter: {self.counter_name}"]
    if self.large_group:
        lines.append(f"Group: {self.large_group}")
    if self.mid_group:
        lines.append(f"Sub-group: {self.mid_group}")
    if self.description:
        lines.append(f"Description: {self.description}")
    if self.unit:
        lines.append(f"Unit: {self.unit}")
    if self.period:
        lines.append(f"Period: {self.period}")
    return "\n".join(lines)
```

`alarm_ref_parser.py`의 `AlarmRecord` 클래스:
```python
def to_chunk_text(self) -> str:
    lines = [
        f"Alarm Code: {self.alarm_code}",
        f"Alarm: {self.alarm_name}",
    ]
    if self.severity:
        lines.append(f"Severity: {self.severity}")
    if self.description:
        lines.append(f"Description: {self.description}")
    if self.probable_causes:
        lines.append("Probable Causes: " + "; ".join(self.probable_causes))
    if self.recommended_actions:
        lines.append("Recommended Actions: " + "; ".join(self.recommended_actions))
    return "\n".join(lines)
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
python -m pytest tests/unit/chunkers/test_reference_chunker.py -v 2>&1 | tail -20
```

Expected: 4 PASSED

- [ ] **Step 6: 커밋**

```bash
git add src/spar/chunkers/reference_chunker.py \
        src/spar/parsers/counter_ref_parser.py \
        src/spar/parsers/alarm_ref_parser.py \
        tests/unit/chunkers/test_reference_chunker.py
git commit -m "feat(chunkers): add reference_chunker for Parameter/Counter/Alarm Excel records"
```

---

## Task 3: `build_entity_glossary.py` (Pass A 스크립트)

**Files:**
- Create: `scripts/build_entity_glossary.py`
- Create: `tests/integration/test_build_entity_glossary.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/integration/test_build_entity_glossary.py
import json
from pathlib import Path
import pytest

# 샘플 Excel 파일 위치 (이미 존재)
PARAM_SAMPLE = Path("data/samples/parameter_ref_sample.xlsx")
COUNTER_SAMPLE = Path("data/samples/counter_ref_sample.xlsx")
ALARM_SAMPLE = Path("data/samples/alarm_excel_ref_sample.xlsx")


@pytest.mark.skipif(
    not PARAM_SAMPLE.exists(),
    reason="parameter_ref_sample.xlsx not found"
)
def test_build_entity_glossary_produces_valid_json(tmp_path):
    from scripts.build_entity_glossary import build_and_write

    output = tmp_path / "samsung_entities.json"
    build_and_write(
        param_paths=[PARAM_SAMPLE],
        counter_paths=[COUNTER_SAMPLE] if COUNTER_SAMPLE.exists() else [],
        alarm_paths=[ALARM_SAMPLE] if ALARM_SAMPLE.exists() else [],
        output_path=output,
    )
    assert output.exists()
    data = json.loads(output.read_text())
    assert "parameter_names" in data
    assert "counter_names" in data
    assert "alarm_codes" in data
    assert isinstance(data["parameter_names"], list)


@pytest.mark.skipif(
    not PARAM_SAMPLE.exists(),
    reason="parameter_ref_sample.xlsx not found"
)
def test_build_entity_glossary_no_noise_tokens(tmp_path):
    from scripts.build_entity_glossary import build_and_write

    output = tmp_path / "samsung_entities.json"
    build_and_write(
        param_paths=[PARAM_SAMPLE],
        counter_paths=[],
        alarm_paths=[],
        output_path=output,
    )
    data = json.loads(output.read_text())
    # 숫자만인 토큰 없어야 함
    for name in data.get("parameter_names", []):
        assert not name.isdigit(), f"noise token: {name}"
    # 2자 이하 토큰 없어야 함
    for name in data.get("parameter_names", []):
        assert len(name) > 2, f"too short: {name}"
```

- [ ] **Step 2: 실패 확인**

```bash
python -m pytest tests/integration/test_build_entity_glossary.py -v 2>&1 | tail -20
```

Expected: ImportError (`build_entity_glossary` 없음) 또는 skip

- [ ] **Step 3: `build_entity_glossary.py` 구현**

```python
#!/usr/bin/env python3
"""Pass A: Excel Reference 파일 스캔 → dictionary/samsung_entities.json 생성.

Usage:
    python scripts/build_entity_glossary.py \
        --param-dir data/parameter_refs/ \
        --counter-dir data/counter_refs/ \
        --alarm-dir data/alarm_refs/ \
        --output dictionary/samsung_entities.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from spar.parsers.parameter_ref_parser import parse_parameter_ref_excel
from spar.parsers.counter_ref_parser import parse_counter_ref_excel
from spar.parsers.alarm_ref_parser import parse_alarm_ref_excel

_NOISE = re.compile(r"^\d+$|^.{1,2}$")


def _clean(values: list[str]) -> list[str]:
    return sorted({v.strip() for v in values if v and not _NOISE.match(v.strip())})


def scan_parameter_refs(paths: list[Path]) -> dict[str, list[str]]:
    param_names, yang_paths, feature_names = [], [], []
    for p in paths:
        for r in parse_parameter_ref_excel(p):
            param_names.append(r.param_name)
            if r.yang_path:
                yang_paths.append(r.yang_path)
            if r.feature_name:
                feature_names.append(r.feature_name)
    return {
        "parameter_names": _clean(param_names),
        "yang_paths": _clean(yang_paths),
        "feature_names": _clean(feature_names),
    }


def scan_counter_refs(paths: list[Path]) -> dict[str, list[str]]:
    counter_names, counter_groups = [], []
    for p in paths:
        for r in parse_counter_ref_excel(p):
            counter_names.append(r.counter_name)
            if r.mid_group:
                counter_groups.append(r.mid_group)
            if r.large_group:
                counter_groups.append(r.large_group)
    return {
        "counter_names": _clean(counter_names),
        "counter_groups": _clean(counter_groups),
    }


def scan_alarm_refs(paths: list[Path]) -> dict[str, list[str]]:
    alarm_codes, alarm_names = [], []
    for p in paths:
        for r in parse_alarm_ref_excel(p):
            alarm_codes.append(r.alarm_code)
            alarm_names.append(r.alarm_name)
    return {
        "alarm_codes": _clean(alarm_codes),
        "alarm_names": _clean(alarm_names),
    }


def build_and_write(
    param_paths: list[Path],
    counter_paths: list[Path],
    alarm_paths: list[Path],
    output_path: Path,
) -> dict:
    entities: dict = {}
    entities.update(scan_parameter_refs(param_paths))
    entities.update(scan_counter_refs(counter_paths))
    entities.update(scan_alarm_refs(alarm_paths))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entities, ensure_ascii=False, indent=2))
    total = sum(len(v) for v in entities.values())
    print(f"Wrote {total} entities to {output_path}")
    return entities


def _collect_excel(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.xlsx")) + sorted(directory.glob("*.xls"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param-dir", type=Path, default=Path("data/parameter_refs"))
    parser.add_argument("--counter-dir", type=Path, default=Path("data/counter_refs"))
    parser.add_argument("--alarm-dir", type=Path, default=Path("data/alarm_refs"))
    parser.add_argument("--output", type=Path, default=Path("dictionary/samsung_entities.json"))
    # 단일 파일 지정 옵션 (테스트/샘플용)
    parser.add_argument("--param-file", type=Path, action="append", default=[])
    parser.add_argument("--counter-file", type=Path, action="append", default=[])
    parser.add_argument("--alarm-file", type=Path, action="append", default=[])
    args = parser.parse_args()

    param_paths = args.param_file or _collect_excel(args.param_dir)
    counter_paths = args.counter_file or _collect_excel(args.counter_dir)
    alarm_paths = args.alarm_file or _collect_excel(args.alarm_dir)

    build_and_write(param_paths, counter_paths, alarm_paths, args.output)
```

- [ ] **Step 4: 파서 함수명 확인**

```bash
grep -n "^def parse_\|^class " src/spar/parsers/parameter_ref_parser.py src/spar/parsers/counter_ref_parser.py src/spar/parsers/alarm_ref_parser.py
```

파서가 `parse_*_excel(path)` 형태가 아니면 script 내 import 수정.

- [ ] **Step 5: 샘플로 직접 실행 확인**

```bash
python scripts/build_entity_glossary.py \
    --param-file data/samples/parameter_ref_sample.xlsx \
    --output /tmp/test_samsung_entities.json
cat /tmp/test_samsung_entities.json | python3 -c "import json,sys; d=json.load(sys.stdin); print({k: len(v) for k,v in d.items()})"
```

Expected: `{'parameter_names': N, 'yang_paths': N, 'feature_names': N, ...}` (N > 0)

- [ ] **Step 6: 통합 테스트 통과 확인**

```bash
python -m pytest tests/integration/test_build_entity_glossary.py -v 2>&1 | tail -20
```

Expected: PASSED (또는 skip if sample missing)

- [ ] **Step 7: 커밋**

```bash
git add scripts/build_entity_glossary.py tests/integration/test_build_entity_glossary.py
git commit -m "feat(scripts): add build_entity_glossary.py — Pass A for Samsung entity pre-scan"
```

---

## Task 4: `chunker.py` 디스패치 통합 + 전체 파이프라인 연결

**Files:**
- Modify: `src/spar/ingest/chunker.py`
- Create: `tests/unit/ingest/test_chunker_dispatch.py`

- [ ] **Step 1: 현재 chunker.py 구조 확인**

```bash
cat src/spar/ingest/chunker.py
```

`chunk_document()` 시그니처와 doc_type 디스패치 로직 파악.

- [ ] **Step 2: 실패 테스트 작성**

```python
# tests/unit/ingest/test_chunker_dispatch.py
import pytest
from unittest.mock import patch, MagicMock
from spar.ingest.chunker import chunk_document


def test_chunk_document_dispatches_parameter_ref():
    mock_records = [MagicMock()]
    mock_records[0].to_chunk_text.return_value = "Parameter: foo"
    mock_records[0].leaf_mo = "NRCellDU"
    mock_records[0].yang_path = "NRCellDU/foo"

    with patch("spar.ingest.chunker.chunk_parameter_ref") as mock_chunker:
        mock_chunker.return_value = [MagicMock(doc_type="parameter_ref")]
        result = chunk_document(
            records=mock_records,
            doc_type="parameter_ref",
            source_doc="test.xlsx",
        )
    mock_chunker.assert_called_once_with(mock_records, source_doc="test.xlsx")
    assert len(result) == 1


def test_chunk_document_dispatches_counter_ref():
    mock_records = [MagicMock()]
    with patch("spar.ingest.chunker.chunk_counter_ref") as mock_chunker:
        mock_chunker.return_value = [MagicMock(doc_type="counter_ref")]
        result = chunk_document(
            records=mock_records,
            doc_type="counter_ref",
            source_doc="test.xlsx",
        )
    mock_chunker.assert_called_once_with(mock_records, source_doc="test.xlsx")
```

- [ ] **Step 3: 실패 확인**

```bash
python -m pytest tests/unit/ingest/test_chunker_dispatch.py -v 2>&1 | tail -20
```

Expected: 현재 `chunk_document` 시그니처 불일치 또는 ImportError

- [ ] **Step 4: `chunker.py` 디스패치 로직 추가**

`chunk_document()` 함수에 reference 유형 처리 추가 (기존 로직 유지):

```python
from spar.chunkers.reference_chunker import (
    chunk_parameter_ref,
    chunk_counter_ref,
    chunk_alarm_ref,
)

# chunk_document() 내 doc_type 처리 부분에:
if doc_type == "parameter_ref":
    return chunk_parameter_ref(records, source_doc=source_doc)
if doc_type == "counter_ref":
    return chunk_counter_ref(records, source_doc=source_doc)
if doc_type == "alarm_ref":
    return chunk_alarm_ref(records, source_doc=source_doc)
# 기존 로직 (md-aware / fixed-size) 이후
```

> 실제 편집 시 `src/spar/ingest/chunker.py` 전체 읽어보고 기존 구조에 맞춰 삽입.

- [ ] **Step 5: 테스트 통과 확인**

```bash
python -m pytest tests/unit/ingest/test_chunker_dispatch.py -v 2>&1 | tail -20
```

Expected: PASSED

- [ ] **Step 6: 전체 단위 테스트 회귀 확인**

```bash
python -m pytest tests/unit/ -v --tb=short 2>&1 | tail -30
```

Expected: 기존 테스트 모두 PASS, 새 테스트 PASS

- [ ] **Step 7: 커밋**

```bash
git add src/spar/ingest/chunker.py tests/unit/ingest/test_chunker_dispatch.py
git commit -m "feat(ingest): dispatch parameter_ref/counter_ref/alarm_ref to reference_chunker"
```

---

## Task 5: `abbrev_mapper.py` — `get_all_keywords` 실제 연결

**Files:**
- Modify: `src/spar/ingest/chunker.py` 또는 `scripts/run_ingest.py` (tag_keywords 호출 지점)

- [ ] **Step 1: tag_keywords 호출 지점 파악**

```bash
grep -rn "tag_keywords\|load_keywords\|get_all_keywords\|extract_terms" src/ scripts/ | grep -v ".pyc"
```

- [ ] **Step 2: 실패 테스트 작성**

```python
# tests/unit/preprocessing/test_tag_keywords_enriched.py
import json
from pathlib import Path
import pytest
from spar.preprocessing.abbrev_mapper import (
    load_acronyms,
    load_entity_glossary,
    get_all_keywords,
    extract_terms,
)


def test_extract_terms_finds_parameter_name(tmp_path):
    # samsung_entities.json에 파라미터명 있을 때
    entities = {"parameter_names": ["nrDlCellMaxTxPower"], "counter_names": [], "counter_groups": [], "alarm_codes": [], "alarm_names": [], "yang_paths": [], "feature_names": []}
    acronyms = {"global": {}}
    keywords = get_all_keywords(acronyms, entities)
    text = "The parameter nrDlCellMaxTxPower controls downlink power"
    found = extract_terms(text, keywords)
    assert "nrDlCellMaxTxPower" in found


def test_extract_terms_finds_counter_group(tmp_path):
    entities = {"parameter_names": [], "counter_names": [], "counter_groups": ["RRC", "Mobility"], "alarm_codes": [], "alarm_names": [], "yang_paths": [], "feature_names": []}
    acronyms = {"global": {}}
    keywords = get_all_keywords(acronyms, entities)
    text = "RRC connection counters show high values"
    found = extract_terms(text, keywords)
    assert "RRC" in found
```

- [ ] **Step 3: 실패 확인**

```bash
python -m pytest tests/unit/preprocessing/test_tag_keywords_enriched.py -v 2>&1 | tail -20
```

Expected: PASSED (이미 `extract_terms` 동작하므로) — 만약 FAIL이면 `get_all_keywords` 구현 재확인.

- [ ] **Step 4: ingest 파이프라인 연결 확인**

`run_ingest.py` 또는 `chunker.py`에서 `tag_keywords` 호출 시 `get_all_keywords` 사용하도록 확인:

```bash
grep -n "load_keywords\|keywords_set\|tag_keywords" scripts/run_ingest.py src/spar/ingest/chunker.py 2>/dev/null
```

`load_keywords(acronyms)` → `get_all_keywords(acronyms, entities)` 로 교체 필요한 위치 특정.  
`entities`는 `load_entity_glossary(Path("dictionary/samsung_entities.json"))` 로 로드.

- [ ] **Step 5: 교체 및 테스트**

```bash
python -m pytest tests/unit/ tests/integration/ -v --tb=short 2>&1 | tail -30
```

Expected: 모두 PASS

- [ ] **Step 6: 커밋**

```bash
git add -p  # 변경 파일 선택적 스테이징
git commit -m "feat(ingest): wire get_all_keywords into tag_keywords — Samsung entities now indexed"
```

---

## Task 6: PRD + 문서 갱신

**Files:**
- Modify: `docs/prd.md`
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/call_flow_preprocessing.md`

- [ ] **Step 1: prd.md Task 1.3 체크박스 갱신**

Task 1.3에 아래 항목 체크:
```markdown
- [x] **Reference 문서 (Parameter/Counter/Alarm)**: 항목 단위 청크 (parameter 1개 = chunk 1개)
```

산출물:
```markdown
- [x] 유형별 청커 모듈 (reference_chunker.py)
```

- [ ] **Step 2: prd.md Task 1.6 확장 항목 추가**

Task 1.6 완료 목록에 추가:
```markdown
- [x] Samsung 도메인 엔티티 사전 구축 (`dictionary/samsung_entities.json`)
- [x] `build_entity_glossary.py` — Pass A pre-scan 스크립트
- [x] `get_all_keywords()` — 3GPP 약어 + Samsung 엔티티 통합 keyword set
```

- [ ] **Step 3: call_flow_preprocessing.md 갱신**

Pass A 단계 추가:
```
scripts/build_entity_glossary.py  ← Pass A (최초/Excel 변경 시)
    ├─ scan parameter/counter/alarm Excel
    └─ → dictionary/samsung_entities.json

scripts/run_ingest.py  ← Pass B
    ├─ parse_document()
    ├─ chunk_document()  ← reference 유형: 1 record = 1 chunk
    ├─ tag_keywords(acronyms + samsung_entities)  ← 실제 동작
    ├─ encode_chunks()
    └─ ingest_to_milvus()
```

- [ ] **Step 4: AGENTS.md 디렉토리 맵 갱신**

```markdown
├── chunkers/        # 유형별 청킹 전략 — reference_chunker.py (Task 1.3 ✅)
├── dictionary/      # acronyms.json (3GPP 2503건) + samsung_entities.json (생성됨)
```

- [ ] **Step 5: 커밋**

```bash
git add docs/prd.md AGENTS.md README.md docs/call_flow_preprocessing.md
git commit -m "docs: update prd/agents/call_flow for two-pass ingest + reference chunker"
```

---

## 최종 검증

- [ ] 전체 테스트 통과

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -40
```

- [ ] Pass A 실제 실행 (샘플 파일 있으면)

```bash
python scripts/build_entity_glossary.py \
    --param-file data/samples/parameter_ref_sample.xlsx \
    --output dictionary/samsung_entities.json
python3 -c "
import json
d = json.load(open('dictionary/samsung_entities.json'))
print({k: len(v) for k, v in d.items()})
"
```

- [ ] mypy 확인

```bash
python -m mypy src/spar/preprocessing/abbrev_mapper.py src/spar/chunkers/reference_chunker.py --ignore-missing-imports 2>&1 | tail -10
```

---

## 의존성 그래프

```
Task 1 (abbrev_mapper 확장)
    ↓
Task 5 (get_all_keywords 연결)  ←── Task 3 (build_entity_glossary) ──→ Task 6 (문서)
Task 2 (reference_chunker)
    ↓
Task 4 (chunker 디스패치)
    ↓
Task 6 (문서)
```

Task 1, 2, 3은 병렬 진행 가능. Task 4는 Task 2 완료 후. Task 5는 Task 1, 3 완료 후.
