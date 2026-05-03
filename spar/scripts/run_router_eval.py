#!/usr/bin/env python3
"""라우터 골드셋 평가 스크립트.

사용법:
    python scripts/run_router_eval.py --layer regex
    python scripts/run_router_eval.py --layer embedding --threshold 0.65
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SPAR_ROOT = Path(__file__).parent.parent
DEFAULT_GOLDSET = SPAR_ROOT / "data" / "goldsets" / "goldset.jsonl"
DEFAULT_OUTPUT_DIR = SPAR_ROOT / "output" / "router_eval"

ROUTES = [
    "structured_lookup",
    "definition_explain",
    "procedural",
    "diagnostic",
    "comparative",
    "default_rag",
]


def load_goldset(path: Path) -> list[dict]:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [json.loads(l) for l in lines]


def compute_metrics(
    expected: list[str],
    predicted: list[str],
    routes: list[str],
) -> dict:
    """per-route precision/recall/F1 + overall accuracy."""
    if not expected:
        return {
            "accuracy": 0.0,
            "per_route": {
                r: {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
                for r in routes
            },
        }

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


def extract_failures(
    goldset: list[dict],
    expected: list[str],
    predicted: list[str],
) -> list[dict]:
    """expected_route ≠ predicted_route 인 케이스 추출."""
    failures = []
    for item, exp, pred in zip(goldset, expected, predicted):
        if exp != pred:
            failures.append({
                "query_id": item.get("query_id", ""),
                "query": item["query"],
                "expected_route": exp,
                "predicted_route": pred,
                "source_doc": item.get("source_doc", ""),
                "type": item.get("type", ""),
            })
    return failures


def format_failures_md(failures: list[dict], max_samples: int = 5) -> str:
    if not failures:
        return "## Failures\n\nNone.\n"

    by_pair: dict[tuple, list] = defaultdict(list)
    for f in failures:
        by_pair[(f["expected_route"], f["predicted_route"])].append(f)

    lines = [f"## Failures ({len(failures)} total)", ""]

    lines += ["### Distribution", "", "| expected | predicted | count |", "|---|---|---|"]
    for (exp, pred), items in sorted(by_pair.items(), key=lambda x: -len(x[1])):
        lines.append(f"| {exp} | {pred} | {len(items)} |")

    lines += ["", "### Samples"]
    for (exp, pred), items in sorted(by_pair.items(), key=lambda x: -len(x[1])):
        lines.append(f"\n#### {exp} → {pred} ({len(items)}개)")
        for item in items[:max_samples]:
            qid = item["query_id"] or "-"
            lines.append(f"- [{qid}] {item['query']}")

    return "\n".join(lines) + "\n"


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

    import asyncio
    encoder = asyncio.run(get_encoder())
    router = EmbeddingRouter(encoder=encoder, threshold=threshold)
    expected_list: list[str] = []
    predicted_list: list[str] = []
    for item in goldset:
        result = router.route(item["query"])
        predicted = result.route.value if result is not None else Route.DEFAULT_RAG.value
        expected_list.append(item["expected_route"])
        predicted_list.append(predicted)
    return expected_list, predicted_list


def format_report(
    metrics: dict,
    confusion: dict[str, dict[str, int]],
    layer: str,
    date_str: str,
    coverage: dict | None = None,
    failures: list[dict] | None = None,
) -> str:
    total_support = sum(v["support"] for v in metrics["per_route"].values())
    correct = sum(v["tp"] for v in metrics["per_route"].values())
    lines = [
        f"# Router Eval — {layer} layer ({date_str})",
        "",
        "## Overall",
        f"accuracy: {metrics['accuracy']:.1%} ({correct}/{total_support})",
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

    lines += ["", format_failures_md(failures or [])]

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="라우터 골드셋 평가")
    parser.add_argument("--goldset", type=Path, default=DEFAULT_GOLDSET)
    parser.add_argument("--layer", choices=["regex", "embedding"], default="regex")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.goldset.exists():
        print(f"ERROR: goldset 없음: {args.goldset}", file=sys.stderr)
        sys.exit(1)

    goldset = load_goldset(args.goldset)
    print(f"goldset {len(goldset)}개 로드")

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    ts_str = now.strftime("%Y%m%d_%H%M%S")

    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR / ts_str
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coverage = None
    if args.layer == "regex":
        expected, predicted = eval_regex(goldset)
    elif args.layer == "embedding":
        expected, predicted = eval_embedding(goldset, args.threshold)
        fallback_count = sum(1 for p in predicted if p == "default_rag")
        coverage = {
            "threshold": args.threshold,
            "matched": len(predicted) - fallback_count,
            "fallback": fallback_count,
            "total": len(predicted),
        }

    metrics = compute_metrics(expected, predicted, ROUTES)
    confusion = build_confusion_matrix(expected, predicted, ROUTES)
    failures = extract_failures(goldset, expected, predicted)

    print(f"\nOverall accuracy: {metrics['accuracy']:.1%}")
    print("Per-route F1:")
    for route, m in metrics["per_route"].items():
        if m["support"] > 0:
            print(f"  {route:22s}: {m['f1']:.2f}  (support={m['support']})")
    print(f"\nFailures: {len(failures)}/{len(goldset)}")

    md_path = args.output_dir / f"router_eval_{args.layer}.md"
    json_path = args.output_dir / f"router_eval_{args.layer}.json"

    report = format_report(metrics, confusion, layer=args.layer, date_str=date_str, coverage=coverage, failures=failures)
    md_path.write_text(report, encoding="utf-8")

    json_data = {
        "layer": args.layer,
        "date": date_str,
        "goldset": str(args.goldset),
        "total": len(goldset),
        "accuracy": metrics["accuracy"],
        "per_route": metrics["per_route"],
        "confusion_matrix": confusion,
        "coverage": coverage,
        "failures": failures,
    }
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"MD  저장: {md_path}")
    print(f"JSON 저장: {json_path}")


if __name__ == "__main__":
    main()
