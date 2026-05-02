"""
E2E 평가 runner: goldset → pipeline(LLM generate 포함) → RAGAS 지표 산출

Usage:
    python -m spar.eval.run_e2e_eval \
        --goldset data/goldsets/retrieval_goldset.jsonl \
        --config e2e \
        --metrics faithfulness,answer_relevancy \
        --top-k 10 \
        --save-dataset data/eval_results/ragas_dataset.jsonl \
        --output data/eval_results/e2e_eval.json

Goldset JSONL 필드:
    query_id, query, gold_answer (선택), section, source_doc, type (선택)

LLM / Encoder:
    LLM_MAIN_URL, LLM_MAIN_MODEL env vars (spar.llm.config 참조)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from spar.eval.ragas_metrics import RagasSample, compute_ragas_metrics
from spar.pipeline.config import PRESET_CONFIGS
from spar.pipeline.graph import build_graph


def _load_goldset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def _run_pipeline(
    goldset: list[dict],
    config_name: str,
    top_k: int,
    llm_client,
    router,
    reranker,
    encoder,
    milvus,
) -> list[tuple[dict, dict]]:
    preset_map = {c.name: c for c in PRESET_CONFIGS}
    if config_name not in preset_map:
        valid = list(preset_map)
        print(f"Error: unknown config '{config_name}'. Available: {valid}", file=sys.stderr)
        sys.exit(1)
    cfg = preset_map[config_name]

    use_llm = getattr(cfg, "use_real_generate", False)
    graph = build_graph(
        router=router,
        reranker=reranker,
        encoder=encoder,
        milvus=milvus,
        config=cfg,
        llm=llm_client if use_llm else None,
    )
    if not use_llm:
        print(f"  [warn] config '{config_name}' has use_real_generate=False → answer will be empty")

    pairs: list[tuple[dict, dict]] = []
    for i, gold in enumerate(goldset, 1):
        qid = gold.get("query_id", f"q{i}")
        preview = gold["query"][:60]
        print(f"  [{i}/{len(goldset)}] {qid} — {preview}", flush=True)
        try:
            state = await graph.ainvoke({
                "query": gold["query"],
                "top_k": top_k,
                "gold_answer": gold.get("gold_answer"),
            })
            pairs.append((gold, state))
        except Exception as exc:
            print(f"    [warn] {qid} failed: {exc}", file=sys.stderr)

    return pairs


def _to_ragas_samples(pairs: list[tuple[dict, dict]]) -> list[RagasSample]:
    samples: list[RagasSample] = []
    for gold, state in pairs:
        reranked = state.get("reranked_chunks")
        retrieved = reranked if reranked is not None else state.get("raw_chunks", [])
        answer = state.get("answer") or ""
        contexts = [c["text"] for c in retrieved[:10] if c.get("text")]

        sample: RagasSample = {
            "query_id": gold.get("query_id", ""),
            "question": gold["query"],
            "answer": answer,
            "contexts": contexts,
        }
        if gold.get("gold_answer"):
            sample["ground_truth"] = gold["gold_answer"]
        samples.append(sample)
    return samples


def _save_ragas_dataset(samples: list[RagasSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n")
    print(f"  RAGAS dataset → {path}")


def _print_summary(metrics: dict) -> None:
    print(f"\n{'─' * 50}")
    print(f"  n_samples : {metrics['n_samples']}")
    for key, val in metrics.items():
        if key in ("n_samples", "per_sample"):
            continue
        print(f"  {key:<22} {val:.4f}")
    print(f"{'─' * 50}\n")


async def _run(args: argparse.Namespace) -> None:
    from spar.encoder.registry import get_encoder
    from spar.llm import LLMRole, get_client
    from spar.reranker.registry import get_reranker
    from spar.retrieval.milvus_client import SparMilvusClient
    from spar.router.hybrid_router import HybridRouter

    metric_names = [m.strip() for m in args.metrics.split(",")]

    goldset = _load_goldset(args.goldset)
    print(f"Goldset: {len(goldset)} queries | config: {args.config} | top_k: {args.top_k}")

    llm = await get_client(LLMRole.MAIN)
    print(f"LLM: {llm.model}")

    encoder = await get_encoder()
    print(f"Encoder: {encoder.model_name}")

    reranker = await get_reranker()
    router = HybridRouter()
    milvus = SparMilvusClient()
    milvus.connect()

    try:
        print(f"\n파이프라인 실행 중 ({args.config})...")
        pairs = await _run_pipeline(
            goldset=goldset,
            config_name=args.config,
            top_k=args.top_k,
            llm_client=llm,
            router=router,
            reranker=reranker,
            encoder=encoder,
            milvus=milvus,
        )
    finally:
        milvus.close()

    if not pairs:
        print("Error: 파이프라인 실행 결과 없음", file=sys.stderr)
        sys.exit(1)

    samples = _to_ragas_samples(pairs)

    if args.save_dataset:
        _save_ragas_dataset(samples, Path(args.save_dataset))

    print(f"\nRAGAS 지표 평가 중: {metric_names}")
    result = await compute_ragas_metrics(
        samples=samples,
        metrics=metric_names,
        llm=llm,
        encoder=encoder if "answer_relevancy" in metric_names else None,
    )

    _print_summary(result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"결과 저장 → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E 평가: goldset → pipeline → RAGAS 지표")
    parser.add_argument("--goldset", required=True, type=Path, help="JSONL goldset 경로")
    parser.add_argument("--config", default="e2e", help="GraphConfig 프리셋 (default: e2e)")
    parser.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy",
        help="평가 지표 쉼표 구분 (default: faithfulness,answer_relevancy)",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--save-dataset", type=Path, default=None, help="중간 RAGAS JSONL 저장 경로")
    parser.add_argument("--output", type=Path, default=None, help="최종 결과 JSON 저장 경로")
    args = parser.parse_args()

    if not args.goldset.exists():
        print(f"Error: goldset not found: {args.goldset}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
