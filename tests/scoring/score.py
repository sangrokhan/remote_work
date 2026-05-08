"""
Precision@k scoring for the agent query API.
Usage: python tests/scoring/score.py --top-k 3
Requires: uvicorn running on port 8001, graph populated with validated triples.
"""
import argparse
import json
import httpx
from pathlib import Path


def precision_at_k(expected: list[str], source_nodes: list[str], k: int) -> float:
    top_k = source_nodes[:k]
    hits = sum(1 for n in top_k if n in expected)
    return hits / k if k > 0 else 0.0


def run_scoring(goldset_path: Path, top_k: int, api_url: str) -> None:
    cases = [json.loads(l) for l in goldset_path.read_text().splitlines() if l.strip()]
    scores = []
    for case in cases:
        resp = httpx.post(f"{api_url}/query", json={
            "question": case["question"],
            "gen_filter": case.get("gen_filter"),
            "top_k": top_k,
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        p = precision_at_k(case["expected_param_ids"], data["source_nodes"], top_k)
        scores.append(p)
        status = "OK" if p > 0 else "MISS"
        print(f"[{status}] [{p:.2f}] {case['question'][:60]}")
        print(f"       expected: {case['expected_param_ids']}")
        print(f"       got:      {data['source_nodes']}")

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\nPrecision@{top_k}: {avg:.2f} ({len(cases)} queries)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--api", default="http://localhost:8001")
    args = parser.parse_args()
    run_scoring(Path("tests/scoring/goldset.jsonl"), args.top_k, args.api)
