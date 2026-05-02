# Alarm Reference Sample + Parser + Search Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate alarm reference Excel sample, build parser + in-memory AlarmIndex, and wire alarm_id direct lookup into routing layer.

**Architecture:** Mirrors `parameter_ref_parser` pattern. New files only (parser + index + sample + tests + script). Single touch on `retrieval/routing.py` for alarm_code shortcut. PDF parsing intentionally out of scope.

**Tech Stack:** Python 3.12, openpyxl, pytest, pre-existing `regex_router` for `ALM-NNNN` extraction.

---

## File Structure

| Path | Purpose | Action |
|---|---|---|
| `data/samples/alarm_excel_ref_sample.xlsx` | 12-row sample workbook | Create (binary, via script) |
| `scripts/gen_alarm_sample.py` | Reproducible sample generator | Create |
| `src/spar/parsers/alarm_ref_parser.py` | `AlarmRecord` + `parse_alarm_ref_excel()` | Create |
| `src/spar/retrieval/alarm_index.py` | `AlarmIndex` + `get_alarm_index()` singleton | Create |
| `src/spar/retrieval/routing.py` | Add alarm_code → AlarmIndex lookup branch | Modify |
| `tests/unit/parsers/test_alarm_ref_parser.py` | Parser unit tests | Create |
| `tests/unit/retrieval/test_alarm_index.py` | Index unit tests | Create |
| `tests/unit/retrieval/test_routing_alarm.py` | Routing alarm wiring test | Create |
| `README.md`, `AGENTS.md`, `docs/prd.md` | Doc updates per CLAUDE.md | Modify |

---

## Task 1: Sample generator script + xlsx artifact

**Files:**
- Create: `scripts/gen_alarm_sample.py`
- Create: `data/samples/alarm_excel_ref_sample.xlsx` (output)

- [ ] **Step 1: Write generator script**

`scripts/gen_alarm_sample.py`:
```python
"""Reproducible generator for alarm_excel_ref_sample.xlsx.

Run: python scripts/gen_alarm_sample.py
Outputs: data/samples/alarm_excel_ref_sample.xlsx
"""
from __future__ import annotations

from pathlib import Path

import openpyxl

ROWS: list[tuple[str, str, str, str, str]] = [
    ("ALM-1001", "Cell Out of Service", "Critical", "Radio", "gNB-DU"),
    ("ALM-1002", "Link Down", "Critical", "Transport", "gNB-CU"),
    ("ALM-1003", "Cell Down", "Critical", "Radio", "gNB-DU"),
    ("ALM-1004", "High Temperature", "Major", "HW", "RU"),
    ("ALM-1005", "Fan Failure", "Major", "HW", "BBU"),
    ("ALM-1006", "Clock Sync Loss", "Major", "Transport", "gNB-DU"),
    ("ALM-1007", "License Expiring", "Minor", "SW", "OAM"),
    ("ALM-1008", "High CPU Usage", "Minor", "SW", "gNB-CU"),
    ("ALM-1009", "PRACH Anomaly", "Minor", "Radio", "gNB-DU"),
    ("ALM-1010", "Backup Failed", "Warning", "SW", "OAM"),
    ("ALM-1011", "Config Mismatch", "Warning", "SW", "gNB-CU"),
    ("ALM-1012", "Power Redundancy Lost", "Major", "HW", "BBU"),
]

HEADERS = ("Alarm ID", "Alarm Name", "Severity", "Category", "Module")


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "samples" / "alarm_excel_ref_sample.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "alarms"
    ws.append(HEADERS)
    for row in ROWS:
        ws.append(row)

    wb.save(out)
    print(f"wrote {out} ({len(ROWS)} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run script**

Run: `python scripts/gen_alarm_sample.py`
Expected: prints `wrote .../alarm_excel_ref_sample.xlsx (12 rows)`. File exists.

- [ ] **Step 3: Verify with openpyxl read-back**

Run:
```bash
python -c "import openpyxl; wb=openpyxl.load_workbook('data/samples/alarm_excel_ref_sample.xlsx'); ws=wb.active; print(ws.max_row, ws.max_column, [c.value for c in ws[1]])"
```
Expected: `13 5 ['Alarm ID', 'Alarm Name', 'Severity', 'Category', 'Module']`

- [ ] **Step 4: Commit**

```bash
git add scripts/gen_alarm_sample.py data/samples/alarm_excel_ref_sample.xlsx
git commit -m "feat(samples): add alarm_excel_ref_sample.xlsx generator + 12-row sample"
```

---

## Task 2: AlarmRecord + parser (TDD)

**Files:**
- Create: `tests/unit/parsers/test_alarm_ref_parser.py`
- Create: `src/spar/parsers/alarm_ref_parser.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/parsers/test_alarm_ref_parser.py`:
```python
from pathlib import Path

import openpyxl
import pytest

from spar.parsers.alarm_ref_parser import (
    AlarmRecord,
    parse_alarm_ref_excel,
)

SAMPLE = Path("data/samples/alarm_excel_ref_sample.xlsx")


def _write_xlsx(tmp_path: Path, rows: list[tuple]) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    for r in rows:
        ws.append(r)
    p = tmp_path / "t.xlsx"
    wb.save(p)
    return p


def test_parse_sample_round_trip():
    res = parse_alarm_ref_excel(SAMPLE)
    assert len(res.records) == 12
    ids = [r.alarm_id for r in res.records]
    assert ids[0] == "ALM-1001"
    assert ids[-1] == "ALM-1012"
    rec = res.records[2]
    assert rec.alarm_name == "Cell Down"
    assert rec.severity == "Critical"
    assert rec.category == "Radio"
    assert rec.module == "gNB-DU"


def test_header_alias_resolution(tmp_path):
    p = _write_xlsx(tmp_path, [
        ("Alarm Code", "Name", "Level", "Group", "Node"),
        ("alm-2001", "Test Alarm", "Major", "Radio", "gNB-DU"),
    ])
    res = parse_alarm_ref_excel(p)
    assert len(res.records) == 1
    assert res.records[0].alarm_id == "ALM-2001"  # uppercase normalized


def test_required_field_missing_skipped(tmp_path):
    p = _write_xlsx(tmp_path, [
        ("Alarm ID", "Alarm Name", "Severity"),
        ("", "Orphan", "Critical"),
        ("ALM-3001", "", "Critical"),
        ("ALM-3002", "Good", "Major"),
    ])
    res = parse_alarm_ref_excel(p)
    assert len(res.records) == 1
    assert res.records[0].alarm_id == "ALM-3002"
    assert res.skipped_rows == 2


def test_to_keywords_excludes_blanks():
    rec = AlarmRecord(alarm_id="ALM-9", alarm_name="X")
    assert rec.to_keywords() == ["ALM-9", "X"]
    rec2 = AlarmRecord(alarm_id="ALM-9", alarm_name="X", severity="Major", module="RU")
    assert rec2.to_keywords() == ["ALM-9", "X", "Major", "RU"]


def test_to_chunk_text_format():
    rec = AlarmRecord(alarm_id="ALM-1", alarm_name="Cell Down",
                      severity="Critical", category="Radio", module="gNB-DU")
    txt = rec.to_chunk_text()
    assert "Alarm: ALM-1 — Cell Down" in txt
    assert "Severity: Critical" in txt
    assert "Module: gNB-DU" in txt
```

- [ ] **Step 2: Run tests, expect failure**

Run: `pytest tests/unit/parsers/test_alarm_ref_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: spar.parsers.alarm_ref_parser`

- [ ] **Step 3: Implement parser**

`src/spar/parsers/alarm_ref_parser.py`:
```python
"""Samsung RAN 알람 레퍼런스 Excel 파서.

지원 컬럼 (헤더명 자동 탐색, 별칭 허용):
    Alarm ID, Alarm Name, Severity, Category, Module
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openpyxl

_COLUMN_ALIASES: dict[str, str] = {
    "alarm id": "alarm_id",
    "id": "alarm_id",
    "alarm code": "alarm_id",
    "code": "alarm_id",
    "alarm name": "alarm_name",
    "name": "alarm_name",
    "severity": "severity",
    "level": "severity",
    "category": "category",
    "group": "category",
    "module": "module",
    "node": "module",
    "subsystem": "module",
}

REQUIRED_FIELDS = {"alarm_id", "alarm_name"}


def _normalize_alarm_id(raw: str) -> str:
    return raw.strip().upper()


@dataclass
class AlarmRecord:
    alarm_id: str
    alarm_name: str
    severity: str = ""
    category: str = ""
    module: str = ""
    pdf_ref: str = ""

    def __post_init__(self) -> None:
        self.alarm_id = _normalize_alarm_id(self.alarm_id)

    def to_chunk_text(self) -> str:
        lines = [f"Alarm: {self.alarm_id} — {self.alarm_name}"]
        if self.severity:
            lines.append(f"Severity: {self.severity}")
        if self.category:
            lines.append(f"Category: {self.category}")
        if self.module:
            lines.append(f"Module: {self.module}")
        return "\n".join(lines)

    def to_keywords(self) -> list[str]:
        return [v for v in (self.alarm_id, self.alarm_name, self.severity,
                            self.category, self.module) if v]

    def to_dict(self) -> dict[str, Any]:
        return {
            "alarm_id": self.alarm_id,
            "alarm_name": self.alarm_name,
            "severity": self.severity,
            "category": self.category,
            "module": self.module,
            "pdf_ref": self.pdf_ref,
        }


@dataclass
class AlarmRefParseResult:
    records: list[AlarmRecord] = field(default_factory=list)
    skipped_rows: int = 0
    warnings: list[str] = field(default_factory=list)


def _resolve_header_row(ws) -> tuple[int, dict[str, int]]:
    for row_idx, row in enumerate(ws.iter_rows(max_row=10, values_only=True), start=1):
        col_map: dict[str, int] = {}
        for col_idx, cell in enumerate(row):
            if cell is None:
                continue
            alias = str(cell).strip().lower()
            field_name = _COLUMN_ALIASES.get(alias)
            if field_name and field_name not in col_map:
                col_map[field_name] = col_idx
        if REQUIRED_FIELDS.issubset(col_map):
            return row_idx, col_map
    raise ValueError(
        f"헤더 행 탐색 실패 — 필수 컬럼 없음: {REQUIRED_FIELDS}. "
        "컬럼명 확인 필요 (Alarm ID, Alarm Name)"
    )


def _cell_str(row: tuple, idx: int | None) -> str:
    if idx is None or idx >= len(row):
        return ""
    val = row[idx]
    if val is None:
        return ""
    return str(val).strip()


def parse_alarm_ref_excel(
    path: str | Path,
    sheet_name: str | None = None,
) -> AlarmRefParseResult:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet_name] if sheet_name else wb.active

    header_row, col_map = _resolve_header_row(ws)

    result = AlarmRefParseResult()
    for row_idx, row in enumerate(
        ws.iter_rows(min_row=header_row + 1, values_only=True),
        start=header_row + 1,
    ):
        if all(v is None for v in row):
            continue

        alarm_id = _cell_str(row, col_map.get("alarm_id"))
        alarm_name = _cell_str(row, col_map.get("alarm_name"))

        if not alarm_id or not alarm_name:
            result.skipped_rows += 1
            result.warnings.append(
                f"행 {row_idx}: alarm_id 또는 alarm_name 비어있음 — 스킵"
            )
            continue

        result.records.append(
            AlarmRecord(
                alarm_id=alarm_id,
                alarm_name=alarm_name,
                severity=_cell_str(row, col_map.get("severity")),
                category=_cell_str(row, col_map.get("category")),
                module=_cell_str(row, col_map.get("module")),
            )
        )

    wb.close()
    return result
```

- [ ] **Step 4: Run tests, expect pass**

Run: `pytest tests/unit/parsers/test_alarm_ref_parser.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/spar/parsers/alarm_ref_parser.py tests/unit/parsers/test_alarm_ref_parser.py
git commit -m "feat(parsers): add alarm_ref_parser for Samsung RAN alarm Excel"
```

---

## Task 3: AlarmIndex + singleton (TDD)

**Files:**
- Create: `tests/unit/retrieval/test_alarm_index.py`
- Create: `src/spar/retrieval/alarm_index.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/retrieval/test_alarm_index.py`:
```python
from pathlib import Path

import pytest

from spar.parsers.alarm_ref_parser import AlarmRecord
from spar.retrieval import alarm_index as ai_mod
from spar.retrieval.alarm_index import AlarmIndex, get_alarm_index

SAMPLE = Path("data/samples/alarm_excel_ref_sample.xlsx")


@pytest.fixture(autouse=True)
def _reset_singleton():
    ai_mod._SINGLETON = None
    yield
    ai_mod._SINGLETON = None


def test_lookup_hit():
    idx = AlarmIndex([
        AlarmRecord("ALM-1", "A"),
        AlarmRecord("ALM-2", "B"),
    ])
    rec = idx.lookup("ALM-2")
    assert rec is not None
    assert rec.alarm_name == "B"


def test_lookup_miss():
    idx = AlarmIndex([AlarmRecord("ALM-1", "A")])
    assert idx.lookup("ALM-999") is None


def test_lookup_case_insensitive():
    idx = AlarmIndex([AlarmRecord("ALM-1", "A")])
    assert idx.lookup("alm-1") is not None
    assert idx.lookup("Alm-1") is not None


def test_search_by_name_partial():
    idx = AlarmIndex([
        AlarmRecord("ALM-1", "Cell Down"),
        AlarmRecord("ALM-2", "Link Down"),
        AlarmRecord("ALM-3", "Fan Failure"),
    ])
    hits = idx.search_by_name("down")
    assert len(hits) == 2
    assert {r.alarm_id for r in hits} == {"ALM-1", "ALM-2"}


def test_singleton_loads_default_sample():
    idx = get_alarm_index()
    assert len(idx) == 12
    assert idx.lookup("ALM-1003").alarm_name == "Cell Down"


def test_singleton_caches():
    idx1 = get_alarm_index()
    idx2 = get_alarm_index()
    assert idx1 is idx2


def test_env_override(monkeypatch, tmp_path):
    # uses default sample if env points there
    monkeypatch.setenv("SPAR_ALARM_REF_PATH", str(SAMPLE))
    idx = get_alarm_index()
    assert len(idx) == 12
```

- [ ] **Step 2: Run tests, expect failure**

Run: `pytest tests/unit/retrieval/test_alarm_index.py -v`
Expected: FAIL — `ModuleNotFoundError: spar.retrieval.alarm_index`

- [ ] **Step 3: Implement AlarmIndex**

`src/spar/retrieval/alarm_index.py`:
```python
"""In-memory index of AlarmRecord for alarm_id direct lookup."""
from __future__ import annotations

import os
from pathlib import Path

from spar.parsers.alarm_ref_parser import AlarmRecord, parse_alarm_ref_excel

_DEFAULT_SAMPLE = Path("data/samples/alarm_excel_ref_sample.xlsx")
_SINGLETON: "AlarmIndex | None" = None


class AlarmIndex:
    def __init__(self, records: list[AlarmRecord]) -> None:
        self._by_id: dict[str, AlarmRecord] = {}
        for r in records:
            self._by_id[r.alarm_id.upper()] = r
        self._records: list[AlarmRecord] = list(records)

    def __len__(self) -> int:
        return len(self._records)

    def lookup(self, alarm_id: str) -> AlarmRecord | None:
        if not alarm_id:
            return None
        return self._by_id.get(alarm_id.strip().upper())

    def search_by_name(self, query: str) -> list[AlarmRecord]:
        q = query.strip().lower()
        if not q:
            return []
        return [r for r in self._records if q in r.alarm_name.lower()]


def _resolve_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("SPAR_ALARM_REF_PATH")
    if env:
        return Path(env)
    return _DEFAULT_SAMPLE


def get_alarm_index(path: str | Path | None = None) -> AlarmIndex:
    """Return process-wide AlarmIndex singleton.

    First call loads from ``path`` > ``$SPAR_ALARM_REF_PATH`` > default sample.
    Subsequent calls ignore arguments and return the cached instance.
    """
    global _SINGLETON
    if _SINGLETON is None:
        resolved = _resolve_path(path)
        result = parse_alarm_ref_excel(resolved)
        _SINGLETON = AlarmIndex(result.records)
    return _SINGLETON
```

- [ ] **Step 4: Run tests, expect pass**

Run: `pytest tests/unit/retrieval/test_alarm_index.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/spar/retrieval/alarm_index.py tests/unit/retrieval/test_alarm_index.py
git commit -m "feat(retrieval): add AlarmIndex singleton for alarm_id direct lookup"
```

---

## Task 4: Wire alarm_code into routing.py (TDD)

**Files:**
- Read: `src/spar/retrieval/routing.py` (understand current shape first)
- Modify: `src/spar/retrieval/routing.py`
- Create: `tests/unit/retrieval/test_routing_alarm.py`

- [ ] **Step 1: Read current routing.py to understand entry points**

Run: `cat src/spar/retrieval/routing.py`
Note the function signature(s) and `RoutingResult` (or equivalent) dataclass shape. The integration must be additive — existing tests must keep passing.

- [ ] **Step 2: Write failing test for alarm_code shortcut**

`tests/unit/retrieval/test_routing_alarm.py`:
```python
"""Verify routing layer uses AlarmIndex when alarm_code entity is present."""
from spar.retrieval import alarm_index as ai_mod
from spar.retrieval import routing


def setup_function(_):
    ai_mod._SINGLETON = None


def test_alarm_code_lookup_populates_structured_record():
    result = routing.resolve_alarm_entity({"alarm_code": "ALM-1003"})
    assert result is not None
    assert result["alarm_id"] == "ALM-1003"
    assert result["alarm_name"] == "Cell Down"
    assert "ALM-1003" in result["keywords"]
    assert "Cell Down" in result["keywords"]


def test_alarm_code_unknown_returns_none():
    result = routing.resolve_alarm_entity({"alarm_code": "ALM-9999"})
    assert result is None


def test_no_alarm_code_returns_none():
    result = routing.resolve_alarm_entity({})
    assert result is None
```

- [ ] **Step 3: Run test, expect failure**

Run: `pytest tests/unit/retrieval/test_routing_alarm.py -v`
Expected: FAIL — `AttributeError: module 'spar.retrieval.routing' has no attribute 'resolve_alarm_entity'`

- [ ] **Step 4: Add `resolve_alarm_entity()` helper to `routing.py`**

Append to `src/spar/retrieval/routing.py` (do not remove or rewrite existing code):
```python
# --- alarm_code direct lookup helper (Task: alarm-ref-sample-and-parser) ---

def resolve_alarm_entity(entities: dict) -> dict | None:
    """Resolve an extracted ``alarm_code`` entity against AlarmIndex.

    Returns a dict with keys ``alarm_id``, ``alarm_name``, ``severity``,
    ``category``, ``module``, ``keywords`` if a match is found; otherwise
    ``None``.

    This is the structured-lookup shortcut used before vector search
    when the regex router has identified an exact alarm code.
    """
    code = entities.get("alarm_code") if entities else None
    if not code:
        return None

    from spar.retrieval.alarm_index import get_alarm_index

    rec = get_alarm_index().lookup(code)
    if rec is None:
        return None

    payload = rec.to_dict()
    payload["keywords"] = rec.to_keywords()
    return payload
```

(Import at top of file is intentionally avoided to keep AlarmIndex lazy — singleton loads only on first alarm query.)

- [ ] **Step 5: Run new + existing routing tests**

Run: `pytest tests/unit/retrieval/ -v`
Expected: all tests pass (new 3 + any pre-existing routing tests still green).

- [ ] **Step 6: Commit**

```bash
git add src/spar/retrieval/routing.py tests/unit/retrieval/test_routing_alarm.py
git commit -m "feat(retrieval): wire alarm_code to AlarmIndex via resolve_alarm_entity"
```

---

## Task 5: Documentation updates

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/prd.md`

- [ ] **Step 1: Update `README.md` 디렉토리 구조 / 현 상태**

Add bullets under 현 상태:
- alarm_ref Excel parser + AlarmIndex 직접 lookup wiring (Task 1.1 부분)
- alarm 샘플: `data/samples/alarm_excel_ref_sample.xlsx`

- [ ] **Step 2: Update `AGENTS.md` directory map**

In Section 3 directory map, under `parsers/`:
```
│       ├── parsers/         # 문서 유형별 파서 — docx_parser.py, parameter_ref_parser.py, alarm_ref_parser.py (Task 1.1 ✅ Alarm Excel)
```

Under `retrieval/`:
```
│       ├── retrieval/       # ... + alarm_index.py (AlarmIndex 싱글톤, alarm_id 직접 lookup) ...
```

Add to "현 단계" line: `**Alarm Ref Excel parser + AlarmIndex ✅** (alarm_ref_parser.py, alarm_index.py, routing.resolve_alarm_entity, 12-row sample)`.

- [ ] **Step 3: Update `docs/prd.md`**

Locate Phase 1 / Task 1.1 (DOCX/PDF parsers). Append:
```
- [x] Alarm Reference Excel parser (`alarm_ref_parser.py`) — 2026-05-02 — branch feat/alarm-ref-sample-and-parser
  - 산출물: data/samples/alarm_excel_ref_sample.xlsx, src/spar/parsers/alarm_ref_parser.py,
            src/spar/retrieval/alarm_index.py, routing.resolve_alarm_entity
  - 후속: alarm_ref PDF 파서 (alarm_id join)
```

(If Phase/Task numbering differs, place the entry under the Alarm-related task that already exists; if none, add it under Task 1.1 as a sub-bullet.)

- [ ] **Step 4: Commit**

```bash
git add README.md AGENTS.md docs/prd.md
git commit -m "docs: record alarm_ref parser + AlarmIndex in README/AGENTS/prd"
```

---

## Task 6: End-to-end smoke

**Files:**
- (no changes; verification only)

- [ ] **Step 1: Run full new test set**

Run:
```bash
pytest tests/unit/parsers/test_alarm_ref_parser.py \
       tests/unit/retrieval/test_alarm_index.py \
       tests/unit/retrieval/test_routing_alarm.py -v
```
Expected: all pass (15 tests).

- [ ] **Step 2: Lint**

Run: `ruff check src/spar/parsers/alarm_ref_parser.py src/spar/retrieval/alarm_index.py src/spar/retrieval/routing.py scripts/gen_alarm_sample.py`
Expected: no errors. Fix any reported.

- [ ] **Step 3: Final repo status check**

Run: `git status` — should be clean. `git log --oneline -10` — should show the task commits.

- [ ] **Step 4: Done — branch ready for review**

No further commit. Branch `feat/alarm-ref-sample-and-parser` ready for PR.

---

## Self-Review Notes

- Spec coverage: §2 산출물 1–8 → Tasks 1–5; §3 schema → Task 2; §4–5 APIs → Tasks 2–3; §6 routing → Task 4; §8 tests → Tasks 2–4.
- Type consistency: `AlarmRecord` fields, `to_keywords()` order (`alarm_id, alarm_name, severity, category, module`), `lookup()` upper-case normalization — consistent across plan.
- No placeholders remain.
- Each task ≤ ~5 mins per step; commits per task; TDD red→green for code tasks.
