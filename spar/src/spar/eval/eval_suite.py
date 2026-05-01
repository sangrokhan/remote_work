# src/spar/eval/eval_suite.py
"""
Multi-config performance suite — runs goldset through multiple GraphConfig variants
and prints a comparison table.

Usage:
    python -m spar.eval.eval_suite \
        --goldset data/goldsets/retrieval_goldset.jsonl \
        --configs baseline +reranker +qexpand full_retrieval \
        --top-k 10 \
        --output data/eval_results/suite_YYYYMMDD.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from spar.eval.metrics import (
    compute_faithfulness,
    compute_suite_metrics,
    recall_at_k,
    reciprocal_rank,
)
from spar.pipeline.config import GraphConfig, PRESET_CONFIGS
from spar.pipeline.graph import build_graph


def _load_goldset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def run_suite(
    configs: list[GraphConfig],
    goldset: list[dict],
    router,
    reranker,
    encoder,
    milvus,
    top_k: int = 10,
    llm_client=None,
    acronyms_path: Path | None = None,
) -> list[dict]:
    results = []
    for cfg in configs:
        graph = build_graph(
            router=router, reranker=reranker, encoder=encoder, milvus=milvus,
            config=cfg, acronyms_path=acronyms_path,
            llm=llm_client if getattr(cfg, "use_real_generate", False) else None,
        )
        per_query: list[dict[str, Any]] = []
        for gold in goldset:
            try:
                state = await graph.ainvoke({
                    "query": gold["query"],
                    "top_k": top_k,
                    "gold_chunks": [gold.get("section", "")],
                    "gold_answer": gold.get("gold_answer"),
                })
            except Exception as exc:
                per_query.append({"error": str(exc), "query_id": gold.get("query_id")})
                continue

            reranked = state.get("reranked_chunks")
            retrieved = reranked if reranked is not None else state.get("raw_chunks", [])
            pq: dict[str, Any] = {
                "query_id": gold.get("query_id"),
                "query": gold["query"],
                "recall_at_5": float(recall_at_k(retrieved, gold, 5)),
                "recall_at_10": float(recall_at_k(retrieved, gold, 10)),
                "mrr": reciprocal_rank(retrieved, gold),
                "latency_ms": sum(state["node_timings"].values()) if state.get("node_timings") else None,
                "faithfulness": None,
            }
            if (
                gold.get("gold_answer")
                and state.get("answer")
                and llm_client
                and getattr(cfg, "use_real_generate", False)
            ):
                pq["faithfulness"] = await compute_faithfulness(
                    answer=state["answer"],
                    context_chunks=retrieved,
                    gold_answer=gold["gold_answer"],
                    llm_client=llm_client,
                )
            per_query.append(pq)

        results.append({"config_name": cfg.name, "per_query": per_query})
    return results


def print_comparison_table(summary_rows: list[dict]) -> None:
    header = f"{'config':<18} | {'R@5':>6} | {'R@10':>6} | {'MRR':>6} | {'Faith':>6} | {'p50ms':>7}"
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}")
    print("-" * len(header))
    for row in summary_rows:
        faith = f"{row['faithfulness']:.3f}" if row["faithfulness"] is not None else "  -  "
        print(
            f"{row['config']:<18} | "
            f"{row['recall_at_5']:>6.3f} | "
            f"{row['recall_at_10']:>6.3f} | "
            f"{row['mrr']:>6.3f} | "
            f"{faith:>6} | "
            f"{row['p50_ms']:>7.0f}"
        )
    print(f"{sep}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goldset", required=True, type=Path)
    parser.add_argument("--configs", nargs="+", default=["baseline", "full_retrieval"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.goldset.exists():
        print(f"Error: goldset not found: {args.goldset}", file=sys.stderr)
        sys.exit(1)

    preset_map = {c.name: c for c in PRESET_CONFIGS}
    selected = []
    for name in args.configs:
        if name not in preset_map:
            print(f"Unknown config '{name}'. Available: {list(preset_map)}", file=sys.stderr)
            sys.exit(1)
        selected.append(preset_map[name])

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
    print(f"Goldset: {len(goldset)} queries | configs: {[c.name for c in selected]}")

    try:
        results = asyncio.run(
            run_suite(
                configs=selected, goldset=goldset, router=router, reranker=reranker,
                encoder=encoder, milvus=client, top_k=args.top_k,
            )
        )
        summary = compute_suite_metrics(results)
        print_comparison_table(summary)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"summary": summary, "details": results}, indent=2))
            print(f"Saved: {args.output}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
