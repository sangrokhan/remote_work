"""
평가 실행기 — Recall@5/10/50, MRR 측정

Usage:
    python -m spar.eval.run_eval \
        --goldset data/goldsets/retrieval_goldset.jsonl \
        --doc-type spec \
        --top-k 50 \
        --output data/eval_results/phase1_eval.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from spar.eval.metrics import compute_metrics, hit_rank


def _load_goldset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _collect_results(goldset: list[dict], doc_type: str, top_k: int) -> list[dict]:
    from spar.encoder.registry import get_encoder
    from spar.retrieval.milvus_client import SparMilvusClient

    encoder = asyncio.run(get_encoder())
    client = SparMilvusClient()

    results = []
    total = len(goldset)
    for i, gold in enumerate(goldset, 1):
        print(f"  [{i}/{total}] {gold['query_id']}: {gold['query'][:60]}", flush=True)
        vec = encoder.encode([gold["query"]])[0].tolist()
        retrieved = client.hybrid_search(
            doc_type=doc_type,
            query_text=gold["query"],
            query_vector=vec,
            top_k=top_k,
        )
        results.append({"gold": gold, "retrieved": retrieved})

    client.close()
    return results


def _print_summary(metrics: dict) -> None:
    print(f"\n{'='*50}")
    print(f"  n_queries : {metrics['n_queries']}")
    print(f"  MRR       : {metrics['mrr']:.4f}")
    for k in [5, 10, 50]:
        key = f"recall_at_{k}"
        if key in metrics:
            print(f"  Recall@{k:<3} : {metrics[key]:.4f}")
    print(f"{'='*50}")

    if metrics.get("by_type"):
        print("\n  By query type:")
        for qtype, m in metrics["by_type"].items():
            print(
                f"    {qtype:<15} n={m['n']:<4} "
                f"MRR={m['mrr']:.3f}  "
                f"R@5={m['recall_at_5']:.3f}  "
                f"R@10={m['recall_at_10']:.3f}"
            )


def _save_output(path: Path, metrics: dict, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metrics": metrics,
        "details": [
            {
                "query_id": r["gold"]["query_id"],
                "query": r["gold"]["query"],
                "type": r["gold"].get("type"),
                "expected_doc": r["gold"]["source_doc"],
                "expected_section": r["gold"]["section"],
                "hit_rank": hit_rank(r["retrieved"], r["gold"]),
                "top5": [
                    f"{c['source_doc']}#{c.get('section_num', '')}"
                    for c in r["retrieved"][:5]
                ],
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\n  Saved → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval 평가 — Recall@K / MRR")
    parser.add_argument("--goldset", required=True, type=Path, help="JSONL 골드셋 경로")
    parser.add_argument("--doc-type", default="spec", help="Milvus doc_type (default: spec)")
    parser.add_argument("--top-k", type=int, default=50, help="최대 검색 결과 수 (default: 50)")
    parser.add_argument("--output", type=Path, default=None, help="상세 결과 JSON 저장 경로")
    args = parser.parse_args()

    if not args.goldset.exists():
        print(f"Error: goldset not found: {args.goldset}", file=sys.stderr)
        sys.exit(1)

    goldset = _load_goldset(args.goldset)
    print(f"골드셋 로드: {len(goldset)}개 쿼리 (doc_type={args.doc_type}, top_k={args.top_k})")

    results = _collect_results(goldset, args.doc_type, args.top_k)
    metrics = compute_metrics(results)

    _print_summary(metrics)

    if args.output:
        _save_output(args.output, metrics, results)


if __name__ == "__main__":
    main()
