# Router Goldset + Eval Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `gen_router_goldset.py` (QA goldset → router goldset 변환) and `run_router_eval.py` (레이어별 라우터 평가 + Markdown 리포트).

**Architecture:** 두 개의 독립 CLI 스크립트. `gen_router_goldset.py`는 `retrieval_goldset.jsonl`의 `type` 필드를 `expected_route`로 매핑해 `router_goldset.jsonl` 생성. `run_router_eval.py`는 골드셋 로드 → 지정 레이어 라우터 실행 → per-route precision/recall/F1 + confusion matrix → Markdown 리포트 저장. 메트릭은 sklearn 없이 직접 계산.

**Tech Stack:** Python 3.12, pytest, `spar.router.{RegexRouter,EmbeddingRouter}`, `spar.encoder.registry.get_encoder`

---

## File Map

| 파일 | 액션 | 역할 |
|---|---|---|
| `scripts/gen_router_goldset.py` | 생성 | QA goldset → router goldset 변환 |
| `scripts/run_router_eval.py` | 생성 | 라우터 레이어별 평가 + Markdown 리포트 |
| `tests/scripts/test_gen_router_goldset.py` | 생성 | 변환 함수 단위 테스트 |
| `tests/scripts/test_run_router_eval.py` | 생성 | 평가 함수 단위 테스트 (regex + embedding layer) |
| `docs/prd.md` | 수정 | Task 2.3 체크박스 갱신 |

---

### Task 1: gen_router_goldset.py

**Files:**
- Create: `scripts/gen_router_goldset.py`
- Create: `tests/scripts/test_gen_router_goldset.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/scripts/test_gen_router_goldset.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


@pytest.fixture()
def qa_items():
    return [
        {"query_id": "Q0001", "query": "What is CA?", "answer": "ans", "type": "definition",
         "source_doc": "29502-i40.md", "spec_number": "29.502", "release": "Rel-18"},
        {"query_id": "Q0002", "query": "How to configure RACH?", "answer": "ans", "type": "procedural",
         "source_doc": "29502-i40.md", "spec_number": "29.502", "release": "Rel-18"},
        {"query_id": "Q0003", "query": "Why failure?", "answer": "ans", "type": "diagnostic",
         "source_doc": "29502-i40.md", "spec_number": "29.502", "release": "Rel-18"},
        {"query_id": "Q0004", "query": "Compare LTE NR", "answer": "ans", "type": "comparative",
         "source_doc": "29502-i40.md", "spec_number": "29.502", "release": "Rel-18"},
        {"query_id": "Q0005", "query": "Default maxTxPower?", "answer": "ans", "type": "lookup",
         "source_doc": "29502-i40.md", "spec_number": "29.502", "release": "Rel-18"},
    ]


def test_convert_type_mapping(qa_items):
    from gen_router_goldset import convert
    results, counts, skipped = convert(qa_items)
    assert len(results) == 5
    assert skipped == 0
    assert results[0]["expected_route"] == "definition_explain"
    assert results[1]["expected_route"] == "procedural"
    assert results[2]["expected_route"] == "diagnostic"
    assert results[3]["expected_route"] == "comparative"
    assert results[4]["expected_route"] == "structured_lookup"


def test_convert_query_id_format(qa_items):
    from gen_router_goldset import convert
    results, _, _ = convert(qa_items, start_id=1)
    assert results[0]["query_id"] == "RQ0001"
    assert results[4]["query_id"] == "RQ0005"


def test_convert_preserves_qa_query_id(qa_items):
    from gen_router_goldset import convert
    results, _, _ = convert(qa_items)
    assert results[0]["qa_query_id"] == "Q0001"
    assert results[4]["qa_query_id"] == "Q0005"


def test_convert_start_id_offset(qa_items):
    from gen_router_goldset import convert
    results, _, _ = convert(qa_items, start_id=10)
    assert results[0]["query_id"] == "RQ0010"
    assert results[4]["query_id"] == "RQ0014"


def test_convert_unknown_type_skipped():
    from gen_router_goldset import convert
    items = [
        {"query_id": "Q0001", "query": "q", "answer": "a", "type": "unknown_type",
         "source_doc": "x.md", "spec_number": "", "release": "Rel-18"},
    ]
    results, counts, skipped = convert(items)
    assert len(results) == 0
    assert skipped == 1


def test_convert_counts(qa_items):
    from gen_router_goldset import convert
    _, counts, _ = convert(qa_items)
    assert counts["definition_explain"] == 1
    assert counts["structured_lookup"] == 1


def test_cli_dry_run(tmp_path):
    from gen_router_goldset import convert
    input_path = tmp_path / "qa.jsonl"
    items = [
        {"query_id": "Q0001", "query": "What is CA?", "answer": "a", "type": "definition",
         "source_doc": "d.md", "spec_number": "29.502", "release": "Rel-18"},
    ]
    input_path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8")
    output_path = tmp_path / "router.jsonl"
    # dry-run: output file should not be created
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/gen_router_goldset.py",
         "--input", str(input_path), "--output", str(output_path), "--dry-run"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    assert result.returncode == 0
    assert not output_path.exists()


def test_cli_writes_output(tmp_path):
    import subprocess
    input_path = tmp_path / "qa.jsonl"
    items = [
        {"query_id": "Q0001", "query": "What is CA?", "answer": "a", "type": "definition",
         "source_doc": "d.md", "spec_number": "29.502", "release": "Rel-18"},
    ]
    input_path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8")
    output_path = tmp_path / "router.jsonl"
    result = subprocess.run(
        [sys.executable, "scripts/gen_router_goldset.py",
         "--input", str(input_path), "--output", str(output_path)],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    assert result.returncode == 0
    assert output_path.exists()
    lines = [json.loads(l) for l in output_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0]["expected_route"] == "definition_explain"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
.venv/bin/pytest tests/scripts/test_gen_router_goldset.py -v
```

Expected: `ModuleNotFoundError: No module named 'gen_router_goldset'`

- [ ] **Step 3: gen_router_goldset.py 구현**

```python
#!/usr/bin/env python3
"""QA goldset (retrieval_goldset.jsonl) → router goldset 변환.

사용법:
    python scripts/gen_router_goldset.py
    python scripts/gen_router_goldset.py --input data/goldsets/retrieval_goldset.jsonl --output data/goldsets/router_goldset.jsonl
    python scripts/gen_router_goldset.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SPAR_ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = SPAR_ROOT / "data" / "goldsets" / "retrieval_goldset.jsonl"
DEFAULT_OUTPUT = SPAR_ROOT / "data" / "goldsets" / "router_goldset.jsonl"

TYPE_TO_ROUTE: dict[str, str] = {
    "definition": "definition_explain",
    "procedural": "procedural",
    "diagnostic": "diagnostic",
    "comparative": "comparative",
    "lookup": "structured_lookup",
}


def convert(
    items: list[dict],
    start_id: int = 1,
) -> tuple[list[dict], dict[str, int], int]:
    """QA 항목 리스트 → router goldset 항목 리스트.

    Returns:
        (results, counts_per_route, skipped_count)
    """
    results: list[dict] = []
    counts: dict[str, int] = {}
    skipped = 0
    for item in items:
        qa_type = item.get("type", "")
        route = TYPE_TO_ROUTE.get(qa_type)
        if route is None:
            print(
                f"  WARN: 미매핑 type={qa_type!r}, query_id={item.get('query_id')} — 스킵",
                file=sys.stderr,
            )
            skipped += 1
            continue
        n = start_id + len(results)
        results.append({
            "query_id": f"RQ{n:04d}",
            "query": item["query"],
            "expected_route": route,
            "source_doc": item.get("source_doc", ""),
            "spec_number": item.get("spec_number", ""),
            "release": item.get("release", ""),
            "qa_query_id": item.get("query_id", ""),
        })
        counts[route] = counts.get(route, 0) + 1
    return results, counts, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="QA goldset → router goldset 변환")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, metavar="FILE")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, metavar="FILE")
    parser.add_argument("--append", action="store_true", help="이어쓰기 (기본: 덮어쓰기)")
    parser.add_argument("--dry-run", action="store_true", help="파일 미생성, 통계만 출력")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: 입력 파일 없음: {args.input}", file=sys.stderr)
        sys.exit(1)

    raw_lines = [l for l in args.input.read_text(encoding="utf-8").splitlines() if l.strip()]
    items = [json.loads(line) for line in raw_lines]
    print(f"QA goldset {len(items)}개 로드")

    start_id = 1
    if args.append and args.output.exists():
        existing = sum(1 for l in args.output.read_text(encoding="utf-8").splitlines() if l.strip())
        start_id = existing + 1
        print(f"기존 {existing}개 항목에 이어쓰기. RQ{start_id:04d}부터 시작")

    results, counts, skipped = convert(items, start_id)

    route_summary = ", ".join(f"{r}={c}" for r, c in sorted(counts.items()))
    print(f"매핑: {route_summary}")
    if skipped:
        print(f"미매핑(스킵): {skipped}개")

    if args.dry_run:
        print("[DRY RUN] 파일 미생성")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with open(args.output, mode, encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"router_goldset.jsonl → {len(results)}개 저장")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
.venv/bin/pytest tests/scripts/test_gen_router_goldset.py -v
```

Expected: 모든 테스트 PASS

- [ ] **Step 5: 커밋**

```bash
git add scripts/gen_router_goldset.py tests/scripts/test_gen_router_goldset.py
git commit -m "feat(scripts): add gen_router_goldset.py — QA goldset to router goldset converter"
```

---

### Task 2: run_router_eval.py (core + regex layer)

**Files:**
- Create: `scripts/run_router_eval.py`
- Create: `tests/scripts/test_run_router_eval.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/scripts/test_run_router_eval.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

ROUTES = [
    "structured_lookup",
    "definition_explain",
    "procedural",
    "diagnostic",
    "comparative",
    "default_rag",
]


@pytest.fixture()
def small_goldset():
    return [
        {"query_id": "RQ0001", "query": "ALM-1234 alarm", "expected_route": "structured_lookup",
         "source_doc": "d.md", "spec_number": "", "release": "Rel-18", "qa_query_id": "Q0001"},
        {"query_id": "RQ0002", "query": "What is Carrier Aggregation?", "expected_route": "definition_explain",
         "source_doc": "d.md", "spec_number": "", "release": "Rel-18", "qa_query_id": "Q0002"},
        {"query_id": "RQ0003", "query": "How to configure RACH parameters?", "expected_route": "procedural",
         "source_doc": "d.md", "spec_number": "", "release": "Rel-18", "qa_query_id": "Q0003"},
    ]


def test_compute_metrics_perfect():
    from run_router_eval import compute_metrics
    expected = ["structured_lookup", "definition_explain", "procedural"]
    predicted = ["structured_lookup", "definition_explain", "procedural"]
    metrics = compute_metrics(expected, predicted, ROUTES)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["per_route"]["structured_lookup"]["precision"] == pytest.approx(1.0)
    assert metrics["per_route"]["structured_lookup"]["recall"] == pytest.approx(1.0)


def test_compute_metrics_one_wrong():
    from run_router_eval import compute_metrics
    expected = ["structured_lookup", "definition_explain", "procedural"]
    predicted = ["structured_lookup", "structured_lookup", "procedural"]
    metrics = compute_metrics(expected, predicted, ROUTES)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["per_route"]["definition_explain"]["recall"] == pytest.approx(0.0)
    assert metrics["per_route"]["definition_explain"]["precision"] == pytest.approx(0.0)
    assert metrics["per_route"]["structured_lookup"]["precision"] == pytest.approx(0.5)


def test_compute_metrics_empty():
    from run_router_eval import compute_metrics
    metrics = compute_metrics([], [], ROUTES)
    assert metrics["accuracy"] == pytest.approx(0.0)


def test_build_confusion_matrix():
    from run_router_eval import build_confusion_matrix
    expected = ["structured_lookup", "definition_explain"]
    predicted = ["structured_lookup", "structured_lookup"]
    matrix = build_confusion_matrix(expected, predicted, ROUTES)
    assert matrix["structured_lookup"]["structured_lookup"] == 1
    assert matrix["definition_explain"]["structured_lookup"] == 1


def test_eval_regex_layer(small_goldset):
    from run_router_eval import eval_regex
    expected, predicted = eval_regex(small_goldset)
    # ALM-1234 → structured_lookup (regex matches alarm code)
    assert expected[0] == "structured_lookup"
    assert predicted[0] == "structured_lookup"
    assert len(expected) == len(predicted) == 3


def test_format_report_contains_accuracy():
    from run_router_eval import compute_metrics, format_report
    expected = ["structured_lookup"]
    predicted = ["structured_lookup"]
    metrics = compute_metrics(expected, predicted, ROUTES)
    matrix = {"structured_lookup": {"structured_lookup": 1}}
    report = format_report(metrics, matrix, layer="regex", date_str="2026-05-01")
    assert "accuracy" in report.lower()
    assert "regex" in report.lower()
    assert "structured_lookup" in report


def test_load_goldset(tmp_path):
    from run_router_eval import load_goldset
    items = [
        {"query_id": "RQ0001", "query": "q", "expected_route": "procedural",
         "source_doc": "d.md", "spec_number": "", "release": "Rel-18", "qa_query_id": "Q0001"},
    ]
    p = tmp_path / "goldset.jsonl"
    p.write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")
    loaded = load_goldset(p)
    assert len(loaded) == 1
    assert loaded[0]["expected_route"] == "procedural"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
.venv/bin/pytest tests/scripts/test_run_router_eval.py -v
```

Expected: `ModuleNotFoundError: No module named 'run_router_eval'`

- [ ] **Step 3: run_router_eval.py 구현 (core + regex)**

```python
#!/usr/bin/env python3
"""라우터 골드셋 평가 스크립트.

사용법:
    python scripts/run_router_eval.py --layer regex
    python scripts/run_router_eval.py --layer embedding --threshold 0.65
    python scripts/run_router_eval.py --layer hybrid
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

SPAR_ROOT = Path(__file__).parent.parent
DEFAULT_GOLDSET = SPAR_ROOT / "data" / "goldsets" / "router_goldset.jsonl"
DEFAULT_OUTPUT_DIR = SPAR_ROOT / "data" / "eval_results"

ROUTES = [
    "structured_lookup",
    "definition_explain",
    "procedural",
    "diagnostic",
    "comparative",
    "default_rag",
]


# ── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_goldset(path: Path) -> list[dict]:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [json.loads(l) for l in lines]


# ── 메트릭 계산 ──────────────────────────────────────────────────────────────

def compute_metrics(
    expected: list[str],
    predicted: list[str],
    routes: list[str],
) -> dict:
    """per-route precision/recall/F1 + overall accuracy."""
    if not expected:
        return {"accuracy": 0.0, "per_route": {r: {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for r in routes}}

    per_route: dict[str, dict] = {}
    for route in routes:
        tp = sum(1 for e, p in zip(expected, predicted) if e == route and p == route)
        fp = sum(1 for e, p in zip(expected, predicted) if e != route and p == route)
        fn = sum(1 for e, p in zip(expected, predicted) if e == route and p != route)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_route[route] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "support": tp + fn,
        }
    accuracy = sum(1 for e, p in zip(expected, predicted) if e == p) / len(expected)
    return {"accuracy": accuracy, "per_route": per_route}


def build_confusion_matrix(
    expected: list[str],
    predicted: list[str],
    routes: list[str],
) -> dict[str, dict[str, int]]:
    """matrix[actual][predicted] = count."""
    matrix: dict[str, dict[str, int]] = {r: {c: 0 for c in routes} for r in routes}
    for e, p in zip(expected, predicted):
        if e in matrix and p in matrix[e]:
            matrix[e][p] += 1
    return matrix


# ── 레이어별 평가 ─────────────────────────────────────────────────────────────

def eval_regex(goldset: list[dict]) -> tuple[list[str], list[str]]:
    """RegexRouter로 평가. 외부 서비스 불필요."""
    sys.path.insert(0, str(SPAR_ROOT / "src"))
    from spar.router.regex_router import RegexRouter
    from spar.router.schemas import Route

    router = RegexRouter()
    expected_list: list[str] = []
    predicted_list: list[str] = []
    for item in goldset:
        result = router.route(item["query"])
        predicted = result.route.value if result is not None else Route.DEFAULT_RAG.value
        expected_list.append(item["expected_route"])
        predicted_list.append(predicted)
    return expected_list, predicted_list


def eval_embedding(goldset: list[dict], threshold: float = 0.65) -> tuple[list[str], list[str]]:
    """EmbeddingRouter로 평가. ENCODER_URL 환경변수 필요."""
    sys.path.insert(0, str(SPAR_ROOT / "src"))
    from spar.encoder.registry import get_encoder
    from spar.router.embedding_router import EmbeddingRouter
    from spar.router.schemas import Route

    encoder = get_encoder()
    router = EmbeddingRouter(encoder=encoder, threshold=threshold)
    expected_list: list[str] = []
    predicted_list: list[str] = []
    for item in goldset:
        result = router.route(item["query"])
        predicted = result.route.value if result is not None else Route.DEFAULT_RAG.value
        expected_list.append(item["expected_route"])
        predicted_list.append(predicted)
    return expected_list, predicted_list


# ── 리포트 생성 ──────────────────────────────────────────────────────────────

def format_report(
    metrics: dict,
    confusion: dict[str, dict[str, int]],
    layer: str,
    date_str: str,
    coverage: dict | None = None,
) -> str:
    lines = [
        f"# Router Eval — {layer} layer ({date_str})",
        "",
        "## Overall",
        f"accuracy: {metrics['accuracy']:.1%} ({sum(v['tp'] for v in metrics['per_route'].values())}/{sum(v['support'] for v in metrics['per_route'].values())})",
        "",
        "## Per-route",
        "| route | precision | recall | F1 | support |",
        "|---|---|---|---|---|",
    ]
    for route, m in metrics["per_route"].items():
        if m["support"] == 0:
            continue
        lines.append(
            f"| {route} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']} |"
        )

    if coverage:
        lines += [
            "",
            "## Coverage",
            f"matched (≥{coverage['threshold']}): {coverage['matched']}/{coverage['total']} ({coverage['matched']/coverage['total']:.1%})",
            f"fallback: {coverage['fallback']}/{coverage['total']} ({coverage['fallback']/coverage['total']:.1%})",
        ]

    lines += ["", "## Confusion Matrix"]
    route_cols = [r for r in ROUTES if any(confusion.get(r2, {}).get(r, 0) for r2 in ROUTES) or any(confusion.get(r, {}).values())]
    if route_cols:
        header = "| actual \\ predicted | " + " | ".join(route_cols) + " |"
        sep = "|---|" + "---|" * len(route_cols)
        lines += [header, sep]
        for actual in ROUTES:
            row_vals = confusion.get(actual, {})
            if not any(row_vals.values()):
                continue
            row = f"| {actual} | " + " | ".join(str(row_vals.get(c, 0)) for c in route_cols) + " |"
            lines.append(row)

    return "\n".join(lines) + "\n"


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="라우터 골드셋 평가")
    parser.add_argument("--goldset", type=Path, default=DEFAULT_GOLDSET)
    parser.add_argument("--layer", choices=["regex", "embedding", "llm", "hybrid"], default="hybrid")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.goldset.exists():
        print(f"ERROR: goldset 없음: {args.goldset}", file=sys.stderr)
        sys.exit(1)

    goldset = load_goldset(args.goldset)
    print(f"goldset {len(goldset)}개 로드")

    today = date.today().isoformat()
    if args.output is None:
        args.output = DEFAULT_OUTPUT_DIR / f"router_eval_{today}.md"

    if args.layer == "regex":
        expected, predicted = eval_regex(goldset)
        coverage = None
    elif args.layer == "embedding":
        expected, predicted = eval_embedding(goldset, args.threshold)
        fallback_count = sum(1 for p in predicted if p == "default_rag")
        coverage = {
            "threshold": args.threshold,
            "matched": len(predicted) - fallback_count,
            "fallback": fallback_count,
            "total": len(predicted),
        }
    else:
        print(f"ERROR: --layer {args.layer} 는 아직 미구현. regex 또는 embedding 사용.", file=sys.stderr)
        sys.exit(1)

    metrics = compute_metrics(expected, predicted, ROUTES)
    confusion = build_confusion_matrix(expected, predicted, ROUTES)
    report = format_report(metrics, confusion, layer=args.layer, date_str=today, coverage=coverage)

    print(f"\nOverall accuracy: {metrics['accuracy']:.1%}")
    print(f"Per-route F1:")
    for route, m in metrics["per_route"].items():
        if m["support"] > 0:
            print(f"  {route:22s}: {m['f1']:.2f}  (support={m['support']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"\n리포트 저장: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
.venv/bin/pytest tests/scripts/test_run_router_eval.py -v
```

Expected: 모든 테스트 PASS

- [ ] **Step 5: 커밋**

```bash
git add scripts/run_router_eval.py tests/scripts/test_run_router_eval.py
git commit -m "feat(scripts): add run_router_eval.py — per-layer router evaluation with Markdown report"
```

---

### Task 3: embedding layer 테스트 + PRD 업데이트

**Files:**
- Modify: `tests/scripts/test_run_router_eval.py` (embedding layer 테스트 추가)
- Modify: `docs/prd.md` (Task 2.3 체크박스)

- [ ] **Step 1: embedding layer 테스트 추가**

`tests/scripts/test_run_router_eval.py` 파일 끝에 추가:

```python
import numpy as np
from unittest.mock import MagicMock, patch


def _make_mock_encoder(dim: int = 8):
    """test_embedding_router.py 와 동일한 stub encoder."""
    from spar.encoder.base import EncoderClient
    encoder = MagicMock(spec=EncoderClient)
    rng = np.random.default_rng(42)
    cache: dict[str, np.ndarray] = {}

    def _encode(texts: list[str], *, normalize: bool = True) -> np.ndarray:
        vecs = []
        for t in texts:
            if t not in cache:
                v = rng.random(dim).astype(np.float32)
                if normalize:
                    v /= np.linalg.norm(v)
                cache[t] = v
            vecs.append(cache[t])
        return np.array(vecs)

    encoder.encode.side_effect = _encode
    return encoder


def test_eval_embedding_layer_returns_routes(small_goldset):
    from run_router_eval import eval_embedding
    mock_enc = _make_mock_encoder()
    with patch("spar.encoder.registry.get_encoder", return_value=mock_enc):
        expected, predicted = eval_embedding(small_goldset, threshold=0.0)
    assert len(expected) == len(predicted) == 3
    assert all(r in ROUTES for r in predicted)


def test_eval_embedding_strict_threshold_all_fallback(small_goldset):
    from run_router_eval import eval_embedding
    mock_enc = _make_mock_encoder()
    with patch("spar.encoder.registry.get_encoder", return_value=mock_enc):
        _, predicted = eval_embedding(small_goldset, threshold=2.0)
    # threshold > max possible cosine similarity → 전부 default_rag
    assert all(p == "default_rag" for p in predicted)
```

- [ ] **Step 2: 테스트 통과 확인**

```bash
.venv/bin/pytest tests/scripts/test_run_router_eval.py -v
```

Expected: 모든 테스트 PASS (embedding 테스트 포함)

- [ ] **Step 3: docs/prd.md Task 2.3 체크박스 갱신**

`docs/prd.md`에서 Task 2.3 항목 찾아 완료 표시:

```markdown
### Task 2.3 — 라우터 골드셋 및 평가

- [x] 라우터 전용 골드셋 (QA goldset type→route 매핑으로 ~300개)
- [ ] Confusion matrix 분석 (어느 라우트끼리 헷갈리는지)
- [ ] 라우팅 정확도 측정 (overall + per-route)
- [x] **산출물**: `scripts/gen_router_goldset.py`
- [x] **산출물**: `scripts/run_router_eval.py`
- [ ] **산출물**: `router_goldset.jsonl` ← goldset 실행 후 생성
- [ ] **산출물**: `router_eval_report.md` ← eval 실행 후 생성
```

- [ ] **Step 4: 전체 테스트 실행**

```bash
.venv/bin/pytest tests/scripts/test_gen_router_goldset.py tests/scripts/test_run_router_eval.py -v
```

Expected: 모든 테스트 PASS

- [ ] **Step 5: 커밋**

```bash
git add tests/scripts/test_run_router_eval.py docs/prd.md
git commit -m "test(scripts): add embedding layer eval test; docs(prd): update Task 2.3 checkboxes"
```

---

## 실행 방법 (구현 완료 후)

```bash
# 1. QA goldset → router goldset 변환
python scripts/gen_router_goldset.py

# 2. regex layer 평가 (외부 서비스 불필요)
python scripts/run_router_eval.py --layer regex

# 3. embedding layer 평가 (ENCODER_URL 필요)
ENCODER_URL=http://localhost:8001 python scripts/run_router_eval.py --layer embedding --threshold 0.65
```
