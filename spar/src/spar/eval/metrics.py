from __future__ import annotations

import re
import statistics
from typing import TYPE_CHECKING


def _is_hit(chunk: dict, gold: dict) -> bool:
    """source_doc 일치 + section_num이 gold section으로 시작하면 hit."""
    doc_match = chunk.get("source_doc", "").removesuffix(".md") == gold["source_doc"].removesuffix(".md")
    section_match = chunk.get("section_num", "").startswith(gold["section"])
    return doc_match and section_match


def recall_at_k(retrieved: list[dict], gold: dict, k: int) -> bool:
    return any(_is_hit(c, gold) for c in retrieved[:k])


def reciprocal_rank(retrieved: list[dict], gold: dict) -> float:
    for i, c in enumerate(retrieved, 1):
        if _is_hit(c, gold):
            return 1.0 / i
    return 0.0


def compute_metrics(results: list[dict]) -> dict:
    """
    results: list of {"gold": goldset_item, "retrieved": list[chunk_dict]}
    returns: metrics dict with MRR, Recall@5/10/50, per query-type breakdown
    """
    ks = [5, 10, 50]
    rrs: list[float] = []
    recalls: dict[int, list[float]] = {k: [] for k in ks}
    by_type: dict[str, dict] = {}

    for r in results:
        gold = r["gold"]
        retrieved = r["retrieved"]
        qtype = gold.get("type", "unknown")

        rr = reciprocal_rank(retrieved, gold)
        rrs.append(rr)
        for k in ks:
            recalls[k].append(float(recall_at_k(retrieved, gold, k)))

        if qtype not in by_type:
            by_type[qtype] = {"rrs": [], **{f"r{k}": [] for k in ks}}
        by_type[qtype]["rrs"].append(rr)
        for k in ks:
            by_type[qtype][f"r{k}"].append(float(recall_at_k(retrieved, gold, k)))

    n = len(results)

    def avg(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "n_queries": n,
        "mrr": avg(rrs),
        **{f"recall_at_{k}": avg(recalls[k]) for k in ks},
        "by_type": {
            qt: {
                "n": len(v["rrs"]),
                "mrr": avg(v["rrs"]),
                **{f"recall_at_{k}": avg(v[f"r{k}"]) for k in ks},
            }
            for qt, v in by_type.items()
        },
    }


def hit_rank(retrieved: list[dict], gold: dict) -> int | None:
    """1-based rank of first hit, or None if not found."""
    for i, c in enumerate(retrieved, 1):
        if _is_hit(c, gold):
            return i
    return None


# --- faithfulness and suite aggregation ---

if TYPE_CHECKING:
    from spar.llm.client import LLMClient


async def compute_faithfulness(
    answer: str,
    context_chunks: list[dict],
    gold_answer: str,
    llm_client: "LLMClient",
) -> float:
    from spar.prompts import load_prompt
    prompt = load_prompt("faithfulness_judge.txt")
    context_text = "\n\n".join(c["text"] for c in context_chunks[:5])
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                f"Answer: {answer}\n\n"
                f"Context:\n{context_text}\n\n"
                f"Reference answer: {gold_answer}"
            ),
        },
    ]
    response = await llm_client.chat(messages, max_tokens=16)
    match = re.search(r"([01]?\.\d+|[01])", response.strip())
    return float(match.group(1)) if match else 0.0


def compute_suite_metrics(results: list[dict]) -> list[dict]:
    """
    results: list of {config_name, per_query: [{recall_at_5, recall_at_10, mrr, latency_ms, faithfulness}]}
    Returns one summary row per config.
    """
    rows = []
    for r in results:
        pq = r["per_query"]
        if not pq:
            continue

        def avg(key: str) -> float:
            vals = [x[key] for x in pq if x.get(key) is not None]
            return sum(vals) / len(vals) if vals else 0.0

        latencies = sorted(x["latency_ms"] for x in pq if x.get("latency_ms") is not None)
        p50 = statistics.median(latencies) if latencies else 0.0

        faith_vals = [x["faithfulness"] for x in pq if x.get("faithfulness") is not None]
        faith_avg = sum(faith_vals) / len(faith_vals) if faith_vals else None

        rows.append({
            "config": r["config_name"],
            "recall_at_5": avg("recall_at_5"),
            "recall_at_10": avg("recall_at_10"),
            "mrr": avg("mrr"),
            "p50_ms": p50,
            "faithfulness": faith_avg,
        })
    return rows
