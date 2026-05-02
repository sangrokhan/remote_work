"""RAGAS-based evaluation runner: computes faithfulness, answer relevance, and context precision via LLM judges."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np

from spar.encoder.base import EncoderClient
from spar.llm.client import LLMClient
from spar.prompts import load_prompt

_VALID_METRICS = {"faithfulness", "answer_relevancy"}


class RagasSample(TypedDict):
    query_id: str
    question: str
    answer: str
    contexts: list[str]
    ground_truth: NotRequired[str]


_REQUIRED_FIELDS = {"query_id", "question", "answer", "contexts"}


def load_ragas_dataset(path: Path) -> list[RagasSample]:
    samples: list[RagasSample] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        missing = _REQUIRED_FIELDS - raw.keys()
        if missing:
            raise ValueError(f"Sample missing fields: {missing} — keys: {list(raw.keys())}")
        sample: RagasSample = {
            "query_id": raw["query_id"],
            "question": raw["question"],
            "answer": raw["answer"],
            "contexts": raw["contexts"],
        }
        if "ground_truth" in raw:
            sample["ground_truth"] = raw["ground_truth"]
        samples.append(sample)
    return samples


async def faithfulness_score(
    question: str,
    answer: str,
    contexts: list[str],
    llm: LLMClient,
) -> float:
    claims_prompt = load_prompt("eval_faithfulness_claims.txt").format(answer=answer)
    claims_raw = await llm.chat([{"role": "user", "content": claims_prompt}], max_tokens=512)
    claims = [c.strip() for c in claims_raw.strip().splitlines() if c.strip()]
    if not claims:
        return 1.0

    context_text = "\n---\n".join(contexts)
    verdict_tmpl = load_prompt("eval_faithfulness_verdict.txt")

    async def _check(claim: str) -> bool:
        prompt = verdict_tmpl.format(context=context_text, claim=claim)
        verdict = await llm.chat([{"role": "user", "content": prompt}], max_tokens=10)
        return verdict.strip().lower().startswith("yes")

    verdicts = await asyncio.gather(*[_check(c) for c in claims])
    return sum(verdicts) / len(verdicts)


async def answer_relevancy_score(
    question: str,
    answer: str,
    encoder: EncoderClient,
    llm: LLMClient,
    n_questions: int = 3,
) -> float:
    gen_prompt = load_prompt("eval_answer_relevancy.txt").format(answer=answer, n=n_questions)
    gen_raw = await llm.chat([{"role": "user", "content": gen_prompt}], max_tokens=256)
    generated = [q.strip() for q in gen_raw.strip().splitlines() if q.strip()][:n_questions]
    if not generated:
        return 0.0

    vecs = encoder.encode([question] + generated, normalize=True)
    orig_vec = vecs[0]
    gen_vecs = vecs[1:]
    similarities = gen_vecs @ orig_vec
    return float(np.mean(similarities))


async def compute_ragas_metrics(
    samples: list[RagasSample],
    metrics: list[str],
    llm: LLMClient,
    encoder: EncoderClient | None = None,
) -> dict:
    if not samples:
        raise ValueError("Cannot evaluate empty dataset")
    unknown = set(metrics) - _VALID_METRICS
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Valid: {_VALID_METRICS}")
    if "answer_relevancy" in metrics and encoder is None:
        raise ValueError("encoder required for answer_relevancy metric")

    per_sample: list[dict] = []
    for s in samples:
        row: dict = {"query_id": s["query_id"]}
        if "faithfulness" in metrics:
            row["faithfulness"] = await faithfulness_score(
                s["question"], s["answer"], s["contexts"], llm
            )
        if "answer_relevancy" in metrics:
            row["answer_relevancy"] = await answer_relevancy_score(
                s["question"], s["answer"], encoder, llm  # type: ignore[arg-type]
            )
        per_sample.append(row)

    def _avg(m: str) -> float:
        vals = [r[m] for r in per_sample if r.get(m) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "n_samples": len(samples),
        **{m: _avg(m) for m in metrics},
        "per_sample": per_sample,
    }
