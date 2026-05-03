# Pass B: Excel Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `run_ingest.py`가 `.xlsx` 파일을 받아 parameter_ref/counter_ref/alarm_ref를 Milvus에 적재하고, Samsung 엔티티 키워드(entities)도 chunk tagging에 포함되도록 한다.

**Architecture:** `run_ingest.py`에 두 가지 수정. (1) `_find_chunk_keywords`를 `extract_terms(text, _KEYWORDS)` 기반으로 교체 — `_KEYWORDS`는 `get_all_keywords(acronyms, entities)` 결과. (2) `ingest_excel_file()` 추가 + `main()`에서 `.xlsx` 감지 시 라우팅. 기존 md/txt 경로는 변경 없음.

**Tech Stack:** Python 3.12, 기존 parsers (`parse_*_ref_excel`), 기존 `dispatch_records`, `extract_terms`, `embed_rows`

**Spec:** `docs/superpowers/specs/2026-05-03-entity-glossary-two-pass-ingest-design.md`

---

## File Map

| 파일 | 변경 |
|---|---|
| `scripts/run_ingest.py` | 수정 — import 추가, `_ENTITIES`/`_KEYWORDS` 전역, `_find_chunk_keywords` 시그니처 변경, `ingest_excel_file()` 추가, `main()` xlsx 라우팅 |
| `tests/integration/test_run_ingest_excel.py` | 신규 — dry-run 3종 통합 테스트 |
| `docs/prd.md` | 수정 — Task 1.3 체크박스 갱신 |

---

## Task 1: `_find_chunk_keywords` → Samsung entities 포함

**Files:**
- Modify: `scripts/run_ingest.py` (line ~1-30 imports, line ~50-60 globals, line ~89-100 function body, line ~160 call site)

- [ ] **Step 1: 현재 import 블록 확인**

```bash
cd /home/han/.openclaw/workspace/remote_work/spar
head -50 scripts/run_ingest.py
```

`from spar.preprocessing.abbrev_mapper import load_acronyms, map_abbreviations` 위치 파악.

- [ ] **Step 2: 실패 테스트 작성**

```python
# tests/unit/test_run_ingest_keywords.py
import importlib, sys
from pathlib import Path
import pytest

# run_ingest.py를 모듈로 임포트하기 위해 scripts/ 경로 추가
_SCRIPTS = Path(__file__).parent.parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def test_find_chunk_keywords_uses_entity_terms(monkeypatch, tmp_path):
    """_find_chunk_keywords가 Samsung entity 이름을 탐지해야 한다."""
    import run_ingest

    # _KEYWORDS에 파라미터명 포함 여부 검증
    # 현재 _KEYWORDS는 get_all_keywords 결과여야 함
    # 샘플 entity term이 _KEYWORDS에 있는지 확인
    # dictionary/samsung_entities.json이 없을 수도 있으므로 monkeypatch로 주입
    monkeypatch.setattr(run_ingest, "_KEYWORDS", {"nrDlCellMaxTxPower", "HO"})
    result = run_ingest._find_chunk_keywords(
        "The parameter nrDlCellMaxTxPower controls downlink power", run_ingest._KEYWORDS
    )
    assert "nrDlCellMaxTxPower" in result


def test_find_chunk_keywords_signature_accepts_set(monkeypatch):
    import run_ingest
    monkeypatch.setattr(run_ingest, "_KEYWORDS", {"CA", "RACH"})
    result = run_ingest._find_chunk_keywords("CA and RACH procedures", run_ingest._KEYWORDS)
    assert "CA" in result
    assert "RACH" in result
```

- [ ] **Step 3: 실패 확인**

```bash
python -m pytest tests/unit/test_run_ingest_keywords.py -v 2>&1 | tail -20
```

Expected: `TypeError` 또는 `AssertionError` — 현재 `_find_chunk_keywords(text, acronyms: dict)` 시그니처가 `set[str]`을 받지 않거나, `_KEYWORDS` 전역이 없음.

- [ ] **Step 4: `run_ingest.py` 수정**

`scripts/run_ingest.py` 파일을 Read 후 세 곳 수정:

**(a) import 교체** — `load_acronyms, map_abbreviations` 줄을:
```python
from spar.preprocessing.abbrev_mapper import (
    extract_terms,
    get_all_keywords,
    load_acronyms,
    load_entity_glossary,
    map_abbreviations,
)
```

**(b) `_ACRONYMS` 전역 선언 바로 아래에 추가**:
```python
_ENTITIES_PATH = Path(__file__).parent.parent / "dictionary" / "samsung_entities.json"
_ENTITIES: dict = load_entity_glossary(_ENTITIES_PATH)
_KEYWORDS: set[str] = get_all_keywords(_ACRONYMS, _ENTITIES)
```

**(c) `_find_chunk_keywords` 함수 교체**:
```python
def _find_chunk_keywords(text: str, keywords: set[str]) -> list[str]:
    return extract_terms(text, keywords)[:50]
```

**(d) `ingest_file()` 내 호출 지점 수정**:
```python
# 변경 전:
c["keywords"] = _find_chunk_keywords(c["text"], _ACRONYMS)
# 변경 후:
c["keywords"] = _find_chunk_keywords(c["text"], _KEYWORDS)
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
python -m pytest tests/unit/test_run_ingest_keywords.py -v 2>&1 | tail -20
```

Expected: 2 PASSED

- [ ] **Step 6: 기존 테스트 회귀 확인**

```bash
python -m pytest tests/unit/ -q --tb=short 2>&1 | tail -20
```

Expected: 모두 PASS

- [ ] **Step 7: 커밋**

```bash
git add scripts/run_ingest.py tests/unit/test_run_ingest_keywords.py
git commit -m "fix(ingest): wire Samsung entities into chunk keyword tagging via get_all_keywords"
```

---

## Task 2: `ingest_excel_file()` + CLI xlsx 라우팅

**Files:**
- Modify: `scripts/run_ingest.py`

- [ ] **Step 1: 현재 `ingest_file()` 와 `main()` 확인**

```bash
grep -n "def ingest_file\|def main\|ALLOWED_SUFFIXES\|input_file\|suffix" scripts/run_ingest.py | head -30
```

`main()`에서 `args.input_file` 처리 분기 위치 파악.

- [ ] **Step 2: 실패 테스트 작성**

```python
# tests/unit/test_run_ingest_excel_unit.py
import sys
from pathlib import Path
import pytest

_SCRIPTS = Path(__file__).parent.parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def test_ingest_excel_file_exists():
    import run_ingest
    assert hasattr(run_ingest, "ingest_excel_file"), "ingest_excel_file not found in run_ingest"


def test_ingest_excel_file_wrong_doc_type_raises(tmp_path):
    import run_ingest
    fake_xlsx = tmp_path / "test.xlsx"
    fake_xlsx.write_bytes(b"")  # 내용 무관 — doc_type 체크가 먼저
    with pytest.raises(SystemExit, match="not supported"):
        run_ingest.ingest_excel_file(None, fake_xlsx, "spec", force=False, dry_run=True)
```

- [ ] **Step 3: 실패 확인**

```bash
python -m pytest tests/unit/test_run_ingest_excel_unit.py -v 2>&1 | tail -20
```

Expected: `AssertionError` — `ingest_excel_file` 없음.

- [ ] **Step 4: `ingest_excel_file()` 구현 추가**

`scripts/run_ingest.py` 에서 `ingest_file()` 함수 정의 뒤에 추가:

```python
_EXCEL_DOC_TYPES = {"parameter_ref", "counter_ref", "alarm_ref"}


def ingest_excel_file(
    client: "SparMilvusClient | None",
    file_path: Path,
    doc_type: str,
    *,
    force: bool,
    dry_run: bool,
) -> int:
    from spar.ingest.chunkers import dispatch_records
    from spar.parsers.alarm_ref_parser import parse_alarm_ref_excel
    from spar.parsers.counter_ref_parser import parse_counter_ref_excel
    from spar.parsers.parameter_ref_parser import parse_parameter_ref_excel

    _PARSER_MAP = {
        "parameter_ref": parse_parameter_ref_excel,
        "counter_ref": parse_counter_ref_excel,
        "alarm_ref": parse_alarm_ref_excel,
    }
    if doc_type not in _PARSER_MAP:
        raise SystemExit(
            f"ERROR: doc_type '{doc_type}' not supported for Excel ingest. "
            f"Use one of: {sorted(_PARSER_MAP)}"
        )

    source_doc = file_path.name
    print(f"Processing: {file_path}  [doc_type={doc_type}]")

    result = _PARSER_MAP[doc_type](file_path)
    records = result.records
    print(f"  parsed: {len(records)} records")

    chunks = dispatch_records(records, source_doc=source_doc, doc_type=doc_type)
    for c in chunks:
        c["keywords"] = _find_chunk_keywords(c["text"], _KEYWORDS)
    print(f"  chunked: {len(chunks)} chunks")

    if not chunks:
        return 0

    rows = embed_rows(chunks, dry_run=dry_run)

    if dry_run:
        print(f"  [DRY RUN] would insert {len(rows)} chunks — skipping Milvus write")
        for r in rows[:2]:
            preview = r["text"][:80].replace("\n", " ")
            kw_preview = r.get("keywords", [])[:5]
            print(f"    chunk_id={r['chunk_id']}  section={r['section']!r}  keywords={kw_preview}  text={preview!r}")
        return len(rows)

    if client is None:
        raise RuntimeError("BUG: client is None after dry-run check")
    if force:
        client.delete_by_source(doc_type, source_doc)
        print(f"  deleted existing chunks for source_doc={source_doc!r}")
    client.insert(doc_type, rows)
    print(f"  inserted: {len(rows)} chunks → spar_{doc_type}")
    return len(rows)
```

- [ ] **Step 5: `main()` xlsx 라우팅 추가**

`main()`에서 `ingest_file(...)` 호출 직전을 찾아 xlsx 분기 추가:

```python
# 변경 전 (대략):
if args.input_file:
    ingest_file(client, args.input_file, ...)

# 변경 후:
if args.input_file:
    if args.input_file.suffix.lower() == ".xlsx":
        count = ingest_excel_file(
            client, args.input_file, args.doc_type,
            force=args.force, dry_run=args.dry_run,
        )
    else:
        count = ingest_file(
            client, args.input_file, args.doc_type,
            force=args.force, dry_run=args.dry_run,
        )
```

> 실제 편집 시 `main()` 전체 읽어보고 기존 분기 구조에 맞게 삽입.

- [ ] **Step 6: 테스트 통과 확인**

```bash
python -m pytest tests/unit/test_run_ingest_excel_unit.py -v 2>&1 | tail -20
```

Expected: 2 PASSED

- [ ] **Step 7: 커밋**

```bash
git add scripts/run_ingest.py tests/unit/test_run_ingest_excel_unit.py
git commit -m "feat(ingest): add ingest_excel_file() + xlsx CLI routing for Reference Excel Pass B"
```

---

## Task 3: 통합 테스트 (dry-run) + PRD 갱신

**Files:**
- Create: `tests/integration/test_run_ingest_excel.py`
- Modify: `docs/prd.md`

샘플 파일 위치:
- `data/samples/parameter_ref_sample.xlsx`
- `data/samples/counter_ref_sample.xlsx`
- `data/samples/alarm_excel_ref_sample.xlsx`

- [ ] **Step 1: 통합 테스트 작성**

```python
# tests/integration/test_run_ingest_excel.py
"""Pass B 통합 테스트 — dry-run 모드로 Milvus 미사용."""
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent.parent
_SAMPLES = _ROOT / "data" / "samples"


def _dry_run(sample: str, doc_type: str) -> str:
    result = subprocess.run(
        [
            sys.executable, "scripts/run_ingest.py",
            "--input-file", str(_SAMPLES / sample),
            "--doc-type", doc_type,
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )
    assert result.returncode == 0, f"STDERR:\n{result.stderr}"
    return result.stdout


def test_parameter_ref_dry_run():
    out = _dry_run("parameter_ref_sample.xlsx", "parameter_ref")
    assert "DRY RUN" in out
    assert "chunks" in out


def test_counter_ref_dry_run():
    out = _dry_run("counter_ref_sample.xlsx", "counter_ref")
    assert "DRY RUN" in out
    assert "chunks" in out


def test_alarm_ref_dry_run():
    out = _dry_run("alarm_excel_ref_sample.xlsx", "alarm_ref")
    assert "DRY RUN" in out
    assert "chunks" in out


def test_parameter_ref_chunk_has_text():
    """dry-run 출력에 청크 텍스트 미리보기가 있는지 확인."""
    out = _dry_run("parameter_ref_sample.xlsx", "parameter_ref")
    assert "chunk_id=" in out


def test_alarm_ref_wrong_doc_type_fails():
    """잘못된 doc_type → SystemExit (returncode != 0)."""
    result = subprocess.run(
        [
            sys.executable, "scripts/run_ingest.py",
            "--input-file", str(_SAMPLES / "alarm_excel_ref_sample.xlsx"),
            "--doc-type", "spec",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )
    assert result.returncode != 0
```

- [ ] **Step 2: 테스트 실행**

```bash
python -m pytest tests/integration/test_run_ingest_excel.py -v 2>&1 | tail -30
```

Expected: 5 PASSED

실패 시 확인 포인트:
- `parameter_ref_sample.xlsx` 파일 구조가 `parse_parameter_ref_excel`과 맞는지 (`data/samples/` 확인)
- `_find_chunk_keywords`의 `_KEYWORDS` 초기화 오류 여부

- [ ] **Step 3: 전체 단위 + 통합 테스트 회귀**

```bash
python -m pytest tests/unit/ tests/integration/ -q --tb=short 2>&1 | tail -30
```

Expected: 모두 PASS

- [ ] **Step 4: `docs/prd.md` Task 1.3 갱신**

Task 1.3 `- [ ] **Reference 문서 (Parameter/Counter/Alarm)**` → `- [x]` 로 변경.

Task 1.3 산출물:
```markdown
- [x] 유형별 청커 모듈 (`reference_chunker.py` — Parameter/Counter/Alarm)
- [x] Excel ingest 경로 (`run_ingest.py --input-file *.xlsx`)
```

- [ ] **Step 5: 커밋**

```bash
git add tests/integration/test_run_ingest_excel.py docs/prd.md
git commit -m "test(integration): Pass B dry-run for parameter/counter/alarm Excel ingest; docs: prd Task 1.3 complete"
```

---

## 최종 검증

```bash
# 전체 테스트
python -m pytest tests/ -q --tb=short --ignore=tests/integration/test_build_entity_glossary.py 2>&1 | tail -20

# Pass B 수동 스모크
python scripts/run_ingest.py \
  --input-file data/samples/parameter_ref_sample.xlsx \
  --doc-type parameter_ref \
  --dry-run
```

Expected 출력:
```
Processing: data/samples/parameter_ref_sample.xlsx  [doc_type=parameter_ref]
  parsed: N records
  chunked: N chunks
  [DRY RUN] would insert N chunks — skipping Milvus write
    chunk_id=...  section='...'  keywords=[...]  text='...'
```

---

## 의존성 그래프

```
Task 1 (keywords fix) → Task 2 (ingest_excel_file) → Task 3 (integration test + prd)
```

Task 1과 Task 2는 순서 필요 (Task 2가 `_KEYWORDS` 사용).
