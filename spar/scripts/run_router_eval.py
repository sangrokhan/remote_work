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

    return "\n".join(lines) + "\n"


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
    print("Per-route F1:")
    for route, m in metrics["per_route"].items():
        if m["support"] > 0:
            print(f"  {route:22s}: {m['f1']:.2f}  (support={m['support']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"\n리포트 저장: {args.output}")


if __name__ == "__main__":
    main()
