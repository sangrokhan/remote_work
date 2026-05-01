"""
RAGAS 방식 답변 품질 평가 CLI — faithfulness, answer_relevancy

Usage:
    python -m spar.eval.run_ragas_eval \\
        --dataset data/eval_results/ragas_dataset.jsonl \\
        --metrics faithfulness,answer_relevancy \\
        --output data/eval_results/ragas_eval.json

Dataset JSONL format (one JSON object per line):
    {"query_id": "Q001", "question": "...", "answer": "...", "contexts": ["...", "..."]}
    optional: "ground_truth": "..."

LLM / Encoder:
    LLM_MAIN_URL, LLM_MAIN_MODEL env vars (기존 spar.llm.config 사용).
    answer_relevancy 지표는 ENCODER_URL / ENCODER_MODEL env var 참조.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _print_summary(metrics: dict) -> None:
    print(f"\n{'─' * 50}")
    print(f"  n_samples : {metrics['n_samples']}")
    for key, val in metrics.items():
        if key in ("n_samples", "per_sample"):
            continue
        print(f"  {key:<22} {val:.4f}")
    print(f"{'─' * 50}\n")


def _save_output(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"  Saved → {path}")


async def _run(args: argparse.Namespace) -> None:
    from spar.eval.ragas_metrics import compute_ragas_metrics, load_ragas_dataset
    from spar.llm import LLMRole, get_client

    metric_names = [m.strip() for m in args.metrics.split(",")]

    print(f"데이터셋 로드: {args.dataset}")
    samples = load_ragas_dataset(args.dataset)
    print(f"  → {len(samples)}개 샘플")

    llm = await get_client(LLMRole.MAIN)
    print(f"  LLM: {llm.model}")

    encoder = None
    if "answer_relevancy" in metric_names:
        from spar.encoder.registry import get_encoder
        encoder = await get_encoder()
        print(f"  Encoder: {encoder.model_name}")

    print(f"\n지표 평가 중: {metric_names}")
    result = await compute_ragas_metrics(samples, metrics=metric_names, llm=llm, encoder=encoder)

    _print_summary(result)

    if args.output:
        _save_output(Path(args.output), result)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS 방식 답변 품질 평가")
    parser.add_argument("--dataset", required=True, type=Path, help="JSONL 데이터셋 경로")
    parser.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy",
        help="평가 지표 쉼표 구분 (default: faithfulness,answer_relevancy)",
    )
    parser.add_argument("--output", type=Path, default=None, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
