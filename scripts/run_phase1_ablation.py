"""
Phase 1 Ablation Runner — Recall@5/10/50, MRR across pipeline presets.

Runs goldset through multiple GraphConfig presets and saves per-preset JSON results
for downstream report generation via gen_phase1_report.py.

Usage:
    python scripts/run_phase1_ablation.py \
        --goldset data/goldsets/goldset_en.jsonl \
        --presets baseline +reranker +qexpand full_retrieval \
        --top-k 50 \
        --limit 500 \
        --output-dir output/phase1_eval

Environment:
    ENCODER_URL, MILVUS_HOST, MILVUS_PORT, RERANKER_BACKEND, ACRONYMS_PATH
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

SPAR_ROOT = Path(__file__).parent.parent
DEFAULT_GOLDSET = SPAR_ROOT / "data" / "goldsets" / "goldset_en.jsonl"
DEFAULT_OUTPUT_DIR = SPAR_ROOT / "output" / "phase1_eval"

EVAL_PRESETS = ["baseline", "+reranker", "+qexpand", "full_retrieval"]


def _load_goldset(path: Path, limit: int | None = None) -> list[dict]:
    items = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return items[:limit] if limit else items


async def _run_preset(
    goldset: list[dict],
    preset_name: str,
    top_k: int,
    router,
    reranker,
    encoder,
    milvus,
    acronyms_path: Path | None,
) -> dict:
    """Run full goldset through one pipeline preset. Returns {metrics, details}."""
    from spar.eval.metrics import compute_metrics, hit_rank
    from spar.pipeline.config import PRESET_CONFIGS
    from spar.pipeline.graph import build_graph

    cfg = next((c for c in PRESET_CONFIGS if c.name == preset_name), None)
    if cfg is None:
        raise ValueError(f"Unknown preset: {preset_name!r}. Available: {[c.name for c in PRESET_CONFIGS]}")

    graph = build_graph(
        router=router,
        reranker=reranker,
        encoder=encoder,
        milvus=milvus,
        config=cfg,
        acronyms_path=acronyms_path,
    )

    results: list[dict] = []
    latencies: list[float] = []

    for gold in goldset:
        t0 = time.perf_counter()
        try:
            state = await graph.ainvoke({"query": gold["query"], "top_k": top_k})
        except Exception as exc:
            print(f"  [warn] {gold.get('query_id')} failed: {exc}", file=sys.stderr)
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        reranked = state.get("reranked_chunks")
        retrieved = reranked if reranked is not None else state.get("raw_chunks", [])
        results.append({"gold": gold, "retrieved": retrieved, "latency_ms": elapsed_ms})

    metrics = compute_metrics(results)

    # attach latency stats
    if latencies:
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        metrics["latency_p50_ms"] = sorted_lat[n // 2]
        metrics["latency_p95_ms"] = sorted_lat[int(n * 0.95)]
        metrics["latency_mean_ms"] = sum(latencies) / n

    details = [
        {
            "query_id": r["gold"].get("query_id", ""),
            "query": r["gold"]["query"],
            "type": r["gold"].get("type"),
            "expected_route": r["gold"].get("expected_route"),
            "needs_decomposition": r["gold"].get("needs_decomposition", False),
            "source_doc": r["gold"].get("source_doc"),
            "section": r["gold"].get("section"),
            "hit_rank": hit_rank(r["retrieved"], r["gold"]),
            "latency_ms": r["latency_ms"],
            "retrieved_top5": [
                {k: v for k, v in c.items() if k in ("source_doc", "section_num", "text")}
                for c in r["retrieved"][:5]
            ],
        }
        for r in results
    ]

    return {"preset": preset_name, "metrics": metrics, "details": details}


def _print_preset_summary(preset_name: str, metrics: dict) -> None:
    print(f"\n  [{preset_name}]")
    print(f"    n={metrics['n_queries']}  MRR={metrics['mrr']:.4f}  "
          f"R@5={metrics.get('recall_at_5', 0):.4f}  "
          f"R@10={metrics.get('recall_at_10', 0):.4f}  "
          f"R@50={metrics.get('recall_at_50', 0):.4f}")
    if "latency_p50_ms" in metrics:
        print(f"    latency p50={metrics['latency_p50_ms']:.0f}ms  "
              f"p95={metrics['latency_p95_ms']:.0f}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 ablation eval across pipeline presets")
    parser.add_argument("--goldset", type=Path, default=DEFAULT_GOLDSET)
    parser.add_argument("--presets", nargs="+", default=EVAL_PRESETS, metavar="PRESET")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="Cap goldset size for quick runs")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--acronyms", type=Path, default=None)
    args = parser.parse_args()

    if not args.goldset.exists():
        print(f"ERROR: goldset not found: {args.goldset}", file=sys.stderr)
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (DEFAULT_OUTPUT_DIR / ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(SPAR_ROOT / "src"))
    from spar.encoder.registry import get_encoder
    from spar.reranker.registry import get_reranker
    from spar.retrieval.milvus_client import SparMilvusClient
    from spar.router.hybrid_router import HybridRouter

    goldset = _load_goldset(args.goldset, args.limit)
    print(f"Goldset: {len(goldset)} queries | presets: {args.presets} | top_k={args.top_k}")
    if args.limit:
        print(f"(limited to {args.limit})")

    async def _init_services():
        enc = await get_encoder()
        rer = await get_reranker()
        return enc, rer

    encoder, reranker = asyncio.run(_init_services())
    router = HybridRouter()
    milvus = SparMilvusClient()
    milvus.connect()

    try:
        summary_rows: list[dict] = []

        for preset_name in args.presets:
            print(f"\nRunning preset: {preset_name} ...")
            t_start = time.perf_counter()
            result = asyncio.run(
                _run_preset(
                    goldset=goldset,
                    preset_name=preset_name,
                    top_k=args.top_k,
                    router=router,
                    reranker=reranker,
                    encoder=encoder,
                    milvus=milvus,
                    acronyms_path=args.acronyms,
                )
            )
            elapsed = time.perf_counter() - t_start

            _print_preset_summary(preset_name, result["metrics"])
            print(f"  (wall time: {elapsed:.1f}s)")

            preset_path = out_dir / f"{preset_name.lstrip('+').replace(' ', '_')}.json"
            preset_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            print(f"  Saved: {preset_path}")

            summary_rows.append({
                "preset": preset_name,
                "n_queries": result["metrics"]["n_queries"],
                "mrr": result["metrics"]["mrr"],
                "recall_at_5": result["metrics"].get("recall_at_5", 0),
                "recall_at_10": result["metrics"].get("recall_at_10", 0),
                "recall_at_50": result["metrics"].get("recall_at_50", 0),
                "latency_p50_ms": result["metrics"].get("latency_p50_ms"),
                "latency_p95_ms": result["metrics"].get("latency_p95_ms"),
                "output_file": str(preset_path),
            })

        summary_path = out_dir / "ablation_summary.json"
        summary_path.write_text(json.dumps({
            "date": datetime.now().isoformat(),
            "goldset": str(args.goldset),
            "top_k": args.top_k,
            "presets": args.presets,
            "results": summary_rows,
        }, indent=2, ensure_ascii=False))
        print(f"\nSummary: {summary_path}")
        print(f"\nRun gen_phase1_report.py --input-dir {out_dir} to generate the report.")

    finally:
        milvus.close()


if __name__ == "__main__":
    main()
