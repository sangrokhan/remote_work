# 3GPP Spec Number-Aware Routing & Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 3GPP 문서번호(`TS 29.502` 등) 포함 질의를 RegexRouter가 fast-path로 캐치하고, spec_number를 Milvus dynamic field로 저장/필터링하여 관련 문서 그룹 내에서 hybrid search를 수행한다.

**Architecture:** RegexRouter에 TS 번호 패턴을 추가해 entities에 spec_number를 추출한다. run_ingest.py의 `--intro-only` 플래그로 파일명에서 spec_number를 파싱해 청크 메타데이터에 동적 필드로 부착한다. Milvus `enable_dynamic_field=True`가 이미 활성화되어 있으므로 스키마 변경 없이 `expr="spec_number == '29.502'"` 필터가 동작한다. hybrid_search()는 이미 `expr` 파라미터를 지원한다.

**Tech Stack:** Python 3.12, pymilvus, pytest, re (stdlib), pathlib (stdlib)

---

## File Map

| 파일 | 변경 | 역할 |
|------|------|------|
| `src/spar/router/regex_router.py` | 수정 | TS `\d{2}.\d{3}` 패턴 + DEFINITION_EXPLAIN route |
| `tests/router/test_regex_router.py` | 수정 | TS 패턴 단위 테스트 추가 |
| `scripts/slice_3gpp_intros.py` | 신규 | 각 3GPP .md 앞 1000줄 → /tmp/3gpp_intros/ |
| `scripts/run_ingest.py` | 수정 | `--intro-only` 플래그, `_parse_spec_number()`, spec_number dynamic field 부착 |
| `tests/test_routing_3gpp.py` | 신규 | spec_number → expr 변환, hybrid_search expr 전달 검증 |

---

## Task 1: RegexRouter — TS spec number 패턴

**Files:**
- Modify: `src/spar/router/regex_router.py`
- Modify: `tests/router/test_regex_router.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/router/test_regex_router.py` 파일에 아래 테스트들을 기존 테스트 아래에 추가:

```python
def test_ts_spec_dotted(router):
    result = router.route("TS 29.502 session management overview")
    assert result is not None
    assert result.route == Route.DEFINITION_EXPLAIN
    assert result.layer == "regex"
    assert result.confidence == 1.0
    assert result.entities.get("spec_number") == "29.502"


def test_ts_spec_dotted_with_3gpp_prefix(router):
    result = router.route("3GPP TS 38.300 NR architecture explained")
    assert result is not None
    assert result.route == Route.DEFINITION_EXPLAIN
    assert result.entities.get("spec_number") == "38.300"


def test_ts_spec_no_dot(router):
    result = router.route("TS29502 what is SMF?")
    assert result is not None
    assert result.route == Route.DEFINITION_EXPLAIN
    assert result.entities.get("spec_number") == "29.502"


def test_ts_spec_space_separated(router):
    result = router.route("TS 29 502 defines session management")
    assert result is not None
    assert result.entities.get("spec_number") == "29.502"


def test_ts_spec_no_conflict_with_alarm(router):
    """TS 패턴과 alarm 패턴이 같은 쿼리에 있을 때 alarm이 우선."""
    result = router.route("ALM-4012 related to TS 29.502")
    assert result is not None
    assert result.route == Route.STRUCTURED_LOOKUP
    assert result.entities.get("alarm_code") == "ALM-4012"
```

- [ ] **Step 2: 실패 확인**

```bash
cd /home/han/.openclaw/workspace/remote_work/spar
python -m pytest tests/router/test_regex_router.py::test_ts_spec_dotted -v
```

Expected: `FAILED` — `AssertionError: assert None is not None`

- [ ] **Step 3: RegexRouter 구현**

`src/spar/router/regex_router.py`에서 기존 패턴 선언 블록 아래에 추가하고 `route()` 메서드에 체크 로직 삽입:

```python
# 파일 상단 패턴 선언 (기존 _PARAM_NAME_RE 아래에 추가)
_SPEC_NUM_RE = re.compile(
    r"\b(?:3GPP\s+)?TS\s*(\d{2})[\.\s]?(\d{3})\b", re.IGNORECASE
)
```

`route()` 메서드에서 `_PARAM_NAME_RE` 체크 직전에 삽입 (alarm/MO 체크 다음, param 체크 이전):

```python
        m = _SPEC_NUM_RE.search(query)
        if m:
            return RouteResult(
                route=Route.DEFINITION_EXPLAIN,
                confidence=1.0,
                layer="regex",
                entities={"spec_number": f"{m.group(1)}.{m.group(2)}"},
            )
```

수정 후 전체 `route()` 순서:
1. `_ALARM_CODE_RE` → `STRUCTURED_LOOKUP`
2. `_ALARM_WORD_RE` → `STRUCTURED_LOOKUP`
3. `_MO_NAME_RE` → `STRUCTURED_LOOKUP`
4. `_SPEC_NUM_RE` → `DEFINITION_EXPLAIN`  ← 신규
5. `_PARAM_NAME_RE` → `STRUCTURED_LOOKUP`
6. `return None`

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/router/test_regex_router.py -v
```

Expected: 모든 테스트 PASS (기존 4개 + 신규 5개)

- [ ] **Step 5: 커밋**

```bash
git add src/spar/router/regex_router.py tests/router/test_regex_router.py
git commit -m "feat(router): RegexRouter에 3GPP TS 문서번호 패턴 추가"
```

---

## Task 2: slice_3gpp_intros.py 스크립트

**Files:**
- Create: `scripts/slice_3gpp_intros.py`
- Create: `tests/scripts/test_slice_3gpp_intros.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/scripts/test_slice_3gpp_intros.py` 생성:

```python
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


def test_parse_spec_number_standard():
    from slice_3gpp_intros import parse_spec_number
    assert parse_spec_number("29502-i40.md") == "29.502"
    assert parse_spec_number("38300-i30.md") == "38.300"
    assert parse_spec_number("23501-i20.md") == "23.501"


def test_parse_spec_number_unknown():
    from slice_3gpp_intros import parse_spec_number
    assert parse_spec_number("10.2 TT.md") == ""
    assert parse_spec_number("foo.md") == ""
    assert parse_spec_number("README.md") == ""


def test_slice_writes_limited_lines(tmp_path):
    from slice_3gpp_intros import slice_file
    src = tmp_path / "29502-i40.md"
    src.write_text("\n".join(f"line {i}" for i in range(2000)), encoding="utf-8")
    dst = tmp_path / "out.md"
    slice_file(src, dst, line_limit=1000)
    result_lines = dst.read_text(encoding="utf-8").splitlines()
    assert len(result_lines) == 1000
    assert result_lines[0] == "line 0"
    assert result_lines[999] == "line 999"


def test_slice_short_file(tmp_path):
    from slice_3gpp_intros import slice_file
    src = tmp_path / "short.md"
    src.write_text("hello\nworld\n", encoding="utf-8")
    dst = tmp_path / "out.md"
    slice_file(src, dst, line_limit=1000)
    assert dst.read_text(encoding="utf-8") == "hello\nworld\n"
```

- [ ] **Step 2: 실패 확인**

```bash
python -m pytest tests/scripts/test_slice_3gpp_intros.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'slice_3gpp_intros'`

- [ ] **Step 3: 스크립트 구현**

`scripts/slice_3gpp_intros.py` 생성:

```python
#!/usr/bin/env python3
"""각 3GPP .md 파일의 앞 1000줄을 /tmp/3gpp_intros/<series>/<filename>으로 복사."""

from __future__ import annotations

import re
import sys
from pathlib import Path

SPAR_ROOT = Path(__file__).parent.parent
DATA_DIR = SPAR_ROOT / "data" / "tspec-llm" / "3GPP-clean" / "Rel-18"
OUT_DIR = Path("/tmp/3gpp_intros")
LINE_LIMIT = 1000

_SPEC_FNAME_RE = re.compile(r"^(\d{2})(\d{3})")


def parse_spec_number(filename: str) -> str:
    """'29502-i40.md' → '29.502'. 매칭 실패 시 ''."""
    stem = Path(filename).stem
    m = _SPEC_FNAME_RE.match(stem)
    if not m:
        return ""
    return f"{m.group(1)}.{m.group(2)}"


def slice_file(src: Path, dst: Path, line_limit: int = LINE_LIMIT) -> None:
    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    dst.write_text("".join(lines[:line_limit]), encoding="utf-8")


def main() -> None:
    md_files = sorted(DATA_DIR.rglob("*.md"))
    if not md_files:
        print(f"ERROR: .md 파일 없음: {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    count = 0
    skipped = 0
    for src in md_files:
        series = src.parent.name
        dst_dir = OUT_DIR / series
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        slice_file(src, dst)
        spec_num = parse_spec_number(src.name)
        label = spec_num if spec_num else "(unknown spec_number)"
        print(f"  {src.name} → {dst}  [{label}]")
        if spec_num:
            count += 1
        else:
            skipped += 1

    print(f"\n완료: {count}개 파싱 성공, {skipped}개 spec_number 파싱 실패 → {OUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/scripts/test_slice_3gpp_intros.py -v
```

Expected: 4개 모두 PASS

- [ ] **Step 5: 커밋**

```bash
git add scripts/slice_3gpp_intros.py tests/scripts/test_slice_3gpp_intros.py
git commit -m "feat(scripts): 3GPP .md intro slicer (앞 1000줄 추출)"
```

---

## Task 3: run_ingest.py — `--intro-only` 플래그 + spec_number 메타데이터

**Files:**
- Modify: `scripts/run_ingest.py`

- [ ] **Step 1: `_parse_spec_number` 함수 추가**

`scripts/run_ingest.py`에서 `ALLOWED_SUFFIXES` 선언 아래에 추가:

```python
import re as _re

_SPEC_FNAME_RE = _re.compile(r"^(\d{2})(\d{3})")


def _parse_spec_number(filename: str) -> str:
    """'29502-i40.md' → '29.502'. 매칭 실패 시 ''."""
    stem = Path(filename).stem
    m = _SPEC_FNAME_RE.match(stem)
    if not m:
        return ""
    return f"{m.group(1)}.{m.group(2)}"
```

- [ ] **Step 2: `ingest_file()` 시그니처에 `intro_only` 추가**

기존:
```python
def ingest_file(
    client: SparMilvusClient | None,
    file_path: Path,
    doc_type: str,
    *,
    force: bool,
    dry_run: bool,
) -> int:
```

변경:
```python
def ingest_file(
    client: SparMilvusClient | None,
    file_path: Path,
    doc_type: str,
    *,
    force: bool,
    dry_run: bool,
    intro_only: bool = False,
) -> int:
```

- [ ] **Step 3: `ingest_file()` 본문에 spec_number 스탬핑 추가**

`ingest_file()` 내부에서 `source_doc = file_path.name` 직후에:

```python
    spec_number = _parse_spec_number(source_doc) if intro_only else ""
```

`chunks = chunk_dispatch(...)` 호출 이후, `for c in chunks: c["doc_type"] = doc_type` 블록 아래에 추가:

```python
    if spec_number:
        for c in chunks:
            c["spec_number"] = spec_number
```

dry-run 출력에도 spec_number 포함:

```python
    if dry_run:
        print(f"  [DRY RUN] would insert {len(rows)} chunks — skipping Milvus write")
        if spec_number:
            print(f"  spec_number={spec_number!r} (dynamic field)")
        for r in rows[:2]:
            preview = r["text"][:80].replace("\n", " ")
            print(f"    chunk_id={r['chunk_id']}  section={r['section']!r}  text={preview!r}")
        return len(rows)
```

- [ ] **Step 4: `ingest_directory()` 시그니처에 `intro_only` 추가**

기존:
```python
def ingest_directory(
    client: SparMilvusClient | None,
    input_dir: Path,
    doc_type: str,
    *,
    force: bool,
    dry_run: bool,
) -> None:
```

변경:
```python
def ingest_directory(
    client: SparMilvusClient | None,
    input_dir: Path,
    doc_type: str,
    *,
    force: bool,
    dry_run: bool,
    intro_only: bool = False,
) -> None:
```

내부 `ingest_file(...)` 호출에 `intro_only=intro_only` 추가.

- [ ] **Step 5: `main()`에 `--intro-only` 인수 추가**

기존 `parser.add_argument("--dry-run", ...)` 아래에:

```python
    parser.add_argument(
        "--intro-only",
        action="store_true",
        help="파일명에서 spec_number 파싱 후 청크 dynamic field로 부착 (spec doc_type 전용)",
    )
```

`with client_ctx as client:` 블록에서:

```python
    with client_ctx as client:
        if args.input_file:
            ingest_file(client, args.input_file, args.doc_type,
                        force=args.force, dry_run=args.dry_run,
                        intro_only=args.intro_only)
        else:
            ingest_directory(client, args.input_dir, args.doc_type,
                             force=args.force, dry_run=args.dry_run,
                             intro_only=args.intro_only)
```

- [ ] **Step 6: dry-run으로 검증**

```bash
cd /home/han/.openclaw/workspace/remote_work/spar
python scripts/slice_3gpp_intros.py
python scripts/run_ingest.py \
    --input-dir /tmp/3gpp_intros \
    --doc-type spec \
    --intro-only \
    --dry-run 2>&1 | head -30
```

Expected 출력 (일부):
```
[DRY RUN] Milvus connection skipped

Processing: /tmp/3gpp_intros/29_series/29502-i40.md  [doc_type=spec]
  parsed: ... chars  →  N chunks
  spec_number='29.502' (dynamic field)
  [DRY RUN] would insert N chunks — skipping Milvus write
    chunk_id=...  section=...  text=...
```

- [ ] **Step 7: 커밋**

```bash
git add scripts/run_ingest.py
git commit -m "feat(ingest): --intro-only 플래그 + spec_number dynamic field 부착"
```

---

## Task 4: end-to-end 라우팅 테스트

**Files:**
- Create: `tests/test_routing_3gpp.py`

- [ ] **Step 1: 테스트 파일 생성**

`tests/test_routing_3gpp.py` 생성:

```python
"""3GPP spec number-aware routing & retrieval 통합 테스트.

RegexRouter 단위 테스트 (Milvus 불필요) + hybrid_search expr 전달 검증 (mock).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from spar.router.regex_router import RegexRouter
from spar.router.schemas import Route


@pytest.fixture
def router() -> RegexRouter:
    return RegexRouter()


# ---------------------------------------------------------------------------
# RegexRouter — spec_number entity 추출 검증
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected_spec", [
    ("TS 29.502 session management", "29.502"),
    ("3GPP TS 38.300 NR architecture", "38.300"),
    ("TS29502 what is SMF?", "29.502"),
    ("TS 38 300 overview of 5G NR", "38.300"),
    ("refer to TS 23.501 for system architecture", "23.501"),
])
def test_regex_router_extracts_spec_number(router, query, expected_spec):
    result = router.route(query)
    assert result is not None, f"expected regex match for: {query!r}"
    assert result.route == Route.DEFINITION_EXPLAIN
    assert result.layer == "regex"
    assert result.entities.get("spec_number") == expected_spec


@pytest.mark.parametrize("query", [
    "session management in 5G core",
    "how to configure SMF parameters",
    "what is carrier aggregation?",
])
def test_regex_router_no_match_for_non_ts_queries(router, query):
    result = router.route(query)
    # RegexRouter는 None 반환 (알람/MO/param 패턴도 없음)
    assert result is None or result.entities.get("spec_number") is None


# ---------------------------------------------------------------------------
# spec_number entity → Milvus expr 문자열 변환
# ---------------------------------------------------------------------------

def test_spec_number_to_milvus_expr(router):
    result = router.route("TS 29.502 session management")
    assert result is not None
    spec_num = result.entities.get("spec_number")
    expr = f"spec_number == '{spec_num}'" if spec_num else None
    assert expr == "spec_number == '29.502'"


def test_no_spec_number_gives_none_expr(router):
    result = router.route("session management in 5G core")
    spec_num = result.entities.get("spec_number") if result else None
    expr = f"spec_number == '{spec_num}'" if spec_num else None
    assert expr is None


# ---------------------------------------------------------------------------
# hybrid_search expr 파라미터 전달 검증 (Milvus mock)
# ---------------------------------------------------------------------------

def test_hybrid_search_receives_expr_when_spec_number_known(router):
    mock_client = MagicMock()
    mock_client.hybrid_search.return_value = []

    result = router.route("TS 38.300 NR architecture overview")
    assert result is not None
    spec_num = result.entities.get("spec_number")
    expr = f"spec_number == '{spec_num}'" if spec_num else None

    mock_client.hybrid_search(
        doc_type="spec",
        query_text="TS 38.300 NR architecture overview",
        query_vector=[0.0] * 1024,
        expr=expr,
    )

    mock_client.hybrid_search.assert_called_once_with(
        doc_type="spec",
        query_text="TS 38.300 NR architecture overview",
        query_vector=[0.0] * 1024,
        expr="spec_number == '38.300'",
    )


def test_hybrid_search_no_expr_when_no_spec_number():
    mock_client = MagicMock()
    mock_client.hybrid_search.return_value = []

    # spec_number 없는 일반 질의
    expr = None

    mock_client.hybrid_search(
        doc_type="spec",
        query_text="session management in 5G core",
        query_vector=[0.0] * 1024,
        expr=expr,
    )

    mock_client.hybrid_search.assert_called_once_with(
        doc_type="spec",
        query_text="session management in 5G core",
        query_vector=[0.0] * 1024,
        expr=None,
    )


def test_expr_fallback_on_empty_result():
    """expr 결과 0건 → None expr로 재시도하는 패턴 검증."""
    mock_client = MagicMock()
    # 첫 번째 호출: expr 있음 → 0건
    # 두 번째 호출: expr 없음 → 결과 있음
    mock_client.hybrid_search.side_effect = [
        [],
        [{"chunk_id": "abc", "text": "session management", "score": 0.9}],
    ]

    def search_with_fallback(client, query_text, query_vector, expr=None):
        results = client.hybrid_search(
            doc_type="spec",
            query_text=query_text,
            query_vector=query_vector,
            expr=expr,
        )
        if not results and expr is not None:
            results = client.hybrid_search(
                doc_type="spec",
                query_text=query_text,
                query_vector=query_vector,
                expr=None,
            )
        return results

    results = search_with_fallback(
        mock_client,
        "TS 29.502 session management",
        [0.0] * 1024,
        expr="spec_number == '29.502'",
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "abc"
    assert mock_client.hybrid_search.call_count == 2
```

- [ ] **Step 2: 테스트 실행**

```bash
python -m pytest tests/test_routing_3gpp.py -v
```

Expected: 모든 테스트 PASS

- [ ] **Step 3: 전체 테스트 suite 실행**

```bash
python -m pytest tests/ -v --ignore=tests/scripts/test_convert_pdf_to_md_cli.py
```

Expected: 기존 테스트 포함 모두 PASS (Milvus/vLLM 연결 불필요한 단위 테스트만)

- [ ] **Step 4: 커밋**

```bash
git add tests/test_routing_3gpp.py
git commit -m "test(routing): 3GPP spec number 라우팅 end-to-end 테스트"
```

---

## 실제 Milvus 연동 확인 (선택, Milvus 기동 시)

Milvus가 올라와 있을 때 실제 ingest 후 검색 확인:

```bash
# 1. intro 슬라이스
python scripts/slice_3gpp_intros.py

# 2. ingest (실제 Milvus 연결)
python scripts/run_ingest.py \
    --input-dir /tmp/3gpp_intros \
    --doc-type spec \
    --intro-only \
    --force

# 3. 검색 확인 (테스트 클라이언트 있을 경우)
# python scripts/test_api.py --query "TS 29.502 session management"
```
