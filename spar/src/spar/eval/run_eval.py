# src/spar/eval/run_eval.py
"""
Performance evaluation runner — Recall@5/10/50, MRR via graph.ainvoke()

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

from spar.eval.metrics import compute_metrics
from spar.pipeline.config import PRESET_CONFIGS
from spar.pipeline.graph import build_graph


def _load_goldset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def _collect_results_via_graph(
    goldset: list[dict],
    doc_type: str,
    top_k: int,
    router,
    reranker,
    encoder,
    milvus,
    acronyms_path: Path | None = None,
) -> list[dict]:
    cfg = next(c for c in PRESET_CONFIGS if c.name == "full_retrieval")
    graph = build_graph(
        router=router, reranker=reranker, encoder=encoder, milvus=milvus,
        config=cfg, acronyms_path=acronyms_path,
    )
    # doc_type not yet wired into SparState; graph uses HybridRouter for routing
    _ = doc_type
    results = []
    for gold in goldset:
        try:
            state = await graph.ainvoke({"query": gold["query"], "top_k": top_k})
        except Exception as exc:
            print(f"[warn] {gold.get('query_id')} failed: {exc}", file=sys.stderr)
            continue
        reranked = state.get("reranked_chunks")
        retrieved = reranked if reranked is not None else state.get("raw_chunks", [])
        results.append({"gold": gold, "retrieved": retrieved})
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
                "expected_doc": r["gold"].get("source_doc"),
                "expected_section": r["gold"].get("section"),
                "retrieved_top3": r["retrieved"][:3],
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goldset", required=True, type=Path)
    parser.add_argument("--doc-type", default="spec")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.goldset.exists():
        print(f"Error: goldset not found: {args.goldset}", file=sys.stderr)
        sys.exit(1)

    from spar.encoder.registry import get_encoder
    from spar.reranker.registry import get_reranker
    from spar.retrieval.milvus_client import SparMilvusClient
    from spar.router.hybrid_router import HybridRouter

    encoder = get_encoder()
    reranker = get_reranker()
    router = HybridRouter()
    client = SparMilvusClient()
    client.connect()

    goldset = _load_goldset(args.goldset)
    print(f"Goldset: {len(goldset)} queries (doc_type={args.doc_type}, top_k={args.top_k})")

    try:
        results = asyncio.run(
            _collect_results_via_graph(
                goldset=goldset, doc_type=args.doc_type, top_k=args.top_k,
                router=router, reranker=reranker, encoder=encoder, milvus=client,
            )
        )
        metrics = compute_metrics(results)
        _print_summary(metrics)

        if args.output:
            _save_output(args.output, metrics, results)
    finally:
        client.close()


if __name__ == "__main__":
    main()
