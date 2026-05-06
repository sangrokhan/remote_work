"""
Phase 1 Report Generator.

Reads ablation JSON output from run_phase1_ablation.py and produces:
  - Ablation comparison table (MRR, Recall@K, latency)
  - Per query-type breakdown
  - Recall@10 failure cases (50 samples) with failure reason classification
  - Priority matrix → Phase 2 input

Usage:
    python scripts/gen_phase1_report.py --input-dir output/phase1_eval/20260504_120000
    python scripts/gen_phase1_report.py --input-dir output/phase1_eval/20260504_120000 \
        --output output/phase1_eval/20260504_120000/phase1_report.md \
        --preset full_retrieval --n-failures 50
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SPAR_ROOT = Path(__file__).parent.parent

FAILURE_REASONS = [
    "decomposition_needed",
    "vocabulary_mismatch",
    "ambiguous_query",
    "chunking_issue",
    "multi_index_needed",
]


def _classify_failure(item: dict) -> str:
    """Heuristic failure reason classification from goldset + retrieval metadata."""
    if item.get("needs_decomposition"):
        return "decomposition_needed"

    query: str = item.get("query", "")
    words = query.split()

    # very short query — likely ambiguous or underspecified
    if len(words) <= 5:
        return "ambiguous_query"

    # query contains known abbreviation-heavy patterns without retrieved match
    # (uppercase acronym density as proxy for vocabulary mismatch)
    upper_tokens = [w for w in words if w.isupper() and len(w) >= 2]
    if len(upper_tokens) >= 2:
        return "vocabulary_mismatch"

    expected_route = item.get("expected_route", "")
    qtype = item.get("type", "")

    # multi-index: diagnostic or comparative queries that need cross-doc retrieval
    if expected_route in ("diagnostic", "comparative"):
        return "multi_index_needed"

    # procedural queries often fail because of MOP/install chunking
    if expected_route == "procedural":
        return "chunking_issue"

    # terminology/technology queries with no vocab match → vocabulary mismatch
    if qtype in ("terminology", "technology"):
        return "vocabulary_mismatch"

    return "chunking_issue"


def _load_preset_result(input_dir: Path, preset_name: str) -> dict | None:
    fname = preset_name.lstrip("+").replace(" ", "_") + ".json"
    path = input_dir / fname
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_summary(input_dir: Path) -> dict | None:
    path = input_dir / "ablation_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _fmt(val: float | None, fmt: str = ".4f") -> str:
    return f"{val:{fmt}}" if val is not None else "—"


def _ablation_table(summary: dict) -> str:
    rows = summary["results"]
    lines = [
        "## Ablation Results",
        "",
        "| Preset | n | MRR | R@5 | R@10 | R@50 | P50 latency |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lat = f"{r['latency_p50_ms']:.0f}ms" if r.get("latency_p50_ms") else "—"
        lines.append(
            f"| `{r['preset']}` "
            f"| {r['n_queries']} "
            f"| {_fmt(r['mrr'])} "
            f"| {_fmt(r['recall_at_5'])} "
            f"| {_fmt(r['recall_at_10'])} "
            f"| {_fmt(r['recall_at_50'])} "
            f"| {lat} |"
        )
    return "\n".join(lines)


def _delta_table(summary: dict) -> str:
    rows = summary["results"]
    if len(rows) < 2:
        return ""

    baseline = next((r for r in rows if r["preset"] == "baseline"), rows[0])

    lines = [
        "### Delta vs Baseline",
        "",
        "| Preset | ΔMRR | ΔR@5 | ΔR@10 | ΔR@50 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r["preset"] == baseline["preset"]:
            continue

        def d(key: str) -> str:
            delta = r[key] - baseline[key]
            sign = "+" if delta >= 0 else ""
            return f"{sign}{delta:.4f}"

        lines.append(
            f"| `{r['preset']}` | {d('mrr')} | {d('recall_at_5')} | {d('recall_at_10')} | {d('recall_at_50')} |"
        )
    return "\n".join(lines)


def _by_type_table(details: list[dict]) -> str:
    from collections import defaultdict

    by_type: dict[str, list[dict]] = defaultdict(list)
    for item in details:
        by_type[item.get("type", "unknown")].append(item)

    lines = [
        "## Per Query-Type Breakdown (full_retrieval preset)",
        "",
        "| Type | n | MRR | R@5 | R@10 |",
        "|---|---|---|---|---|",
    ]

    for qtype, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        n = len(items)

        def rr(item: dict) -> float:
            rank = item.get("hit_rank")
            return 1.0 / rank if rank else 0.0

        mrr = sum(rr(i) for i in items) / n
        r5 = sum(1 for i in items if i.get("hit_rank") and i["hit_rank"] <= 5) / n
        r10 = sum(1 for i in items if i.get("hit_rank") and i["hit_rank"] <= 10) / n
        lines.append(f"| {qtype} | {n} | {mrr:.4f} | {r5:.4f} | {r10:.4f} |")

    return "\n".join(lines)


def _failure_analysis(details: list[dict], n_samples: int = 50, seed: int = 42) -> str:
    failures = [d for d in details if not d.get("hit_rank") or d["hit_rank"] > 10]

    reason_counts: Counter = Counter()
    for f in failures:
        reason_counts[_classify_failure(f)] += 1

    random.seed(seed)
    sampled = random.sample(failures, min(n_samples, len(failures)))

    lines = [
        f"## Recall@10 Failure Analysis",
        "",
        f"Total Recall@10 failures: **{len(failures)}** / {len(details)} ({len(failures)/max(len(details),1):.1%})",
        "",
        "### Failure Reason Distribution",
        "",
        "| Reason | Count | % |",
        "|---|---|---|",
    ]
    total_fail = sum(reason_counts.values())
    for reason in FAILURE_REASONS:
        cnt = reason_counts.get(reason, 0)
        pct = cnt / total_fail * 100 if total_fail else 0
        lines.append(f"| `{reason}` | {cnt} | {pct:.1f}% |")

    lines += [
        "",
        f"### Sample Failures ({len(sampled)} of {len(failures)})",
        "",
    ]

    by_reason: dict[str, list[dict]] = defaultdict(list)
    for f in sampled:
        by_reason[_classify_failure(f)].append(f)

    for reason in FAILURE_REASONS:
        items = by_reason.get(reason, [])
        if not items:
            continue
        lines.append(f"#### `{reason}` ({len(items)} samples)")
        lines.append("")
        for item in items[:10]:
            rank_str = f"rank={item['hit_rank']}" if item.get("hit_rank") else "not found"
            lines.append(f"- [{item['query_id']}] ({rank_str}) {item['query']}")
        lines.append("")

    return "\n".join(lines)


def _priority_matrix(summary: dict, failure_counts: dict[str, int]) -> str:
    total = sum(failure_counts.values())

    # effort and impact estimates per failure type
    priority_data = [
        ("decomposition_needed", "High", "Low", "Query Decomposition already implemented — tune threshold"),
        ("vocabulary_mismatch", "High", "Medium", "Expand synonym dict, BM25 weight tuning"),
        ("ambiguous_query", "Medium", "Low", "Clarification prompts or fallback broadening"),
        ("multi_index_needed", "High", "High", "Task 2.9 Multi-Index parallel search"),
        ("chunking_issue", "Medium", "High", "Task 1.3 MOP/Feature Description chunking"),
    ]

    lines = [
        "## Priority Matrix → Phase 2 Input",
        "",
        "| Failure Reason | Failures | Impact | Effort | Recommended Action |",
        "|---|---|---|---|---|",
    ]
    for reason, impact, effort, action in priority_data:
        cnt = failure_counts.get(reason, 0)
        pct = cnt / total * 100 if total else 0
        lines.append(f"| `{reason}` | {cnt} ({pct:.0f}%) | {impact} | {effort} | {action} |")

    # derive top recommendations from failure distribution
    top_reason = max(failure_counts, key=failure_counts.get) if failure_counts else None

    lines += [
        "",
        "### Top Recommendations for Phase 2",
        "",
    ]

    # get sorted reasons by count
    sorted_reasons = sorted(failure_counts.items(), key=lambda x: -x[1])
    for i, (reason, cnt) in enumerate(sorted_reasons[:3], 1):
        action = next((a for r, _, _, a in priority_data if r == reason), "Investigate")
        lines.append(f"{i}. **{reason}** ({cnt} failures): {action}")

    return "\n".join(lines)


def _build_report(input_dir: Path, preset_name: str, n_failures: int) -> str:
    summary = _load_summary(input_dir)
    preset_result = _load_preset_result(input_dir, preset_name)

    if summary is None and preset_result is None:
        raise FileNotFoundError(f"No ablation results found in {input_dir}")

    date_str = summary["date"][:10] if summary else "unknown"
    goldset_str = summary.get("goldset", "") if summary else ""
    top_k = summary.get("top_k", 50) if summary else 50

    lines = [
        f"# Phase 1 Evaluation Report",
        f"",
        f"**Date**: {date_str}  ",
        f"**Goldset**: `{Path(goldset_str).name if goldset_str else 'unknown'}`  ",
        f"**Top-K**: {top_k}  ",
        f"**Failure preset**: `{preset_name}`",
        f"",
        "---",
        "",
    ]

    if summary:
        lines += [_ablation_table(summary), "", _delta_table(summary), "", "---", ""]

    if preset_result:
        details = preset_result.get("details", [])
        lines += [_by_type_table(details), "", "---", ""]

        failures_section = _failure_analysis(details, n_failures)
        lines.append(failures_section)
        lines += ["", "---", ""]

        # compute failure counts for priority matrix
        from collections import Counter
        failure_counts = Counter(
            _classify_failure(d)
            for d in details
            if not d.get("hit_rank") or d["hit_rank"] > 10
        )
        lines += [_priority_matrix(summary or {}, dict(failure_counts)), ""]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 1 eval report from ablation results")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preset", default="full_retrieval", help="Preset used for failure analysis")
    parser.add_argument("--n-failures", type=int, default=50)
    args = parser.parse_args()

    if not args.input_dir.exists():
        import sys
        print(f"ERROR: {args.input_dir} not found", file=sys.stderr)
        sys.exit(1)

    report = _build_report(args.input_dir, args.preset, args.n_failures)

    out_path = args.output or (args.input_dir / "phase1_report.md")
    out_path.write_text(report, encoding="utf-8")
    print(f"Report: {out_path}")

    # also print summary to stdout
    lines = report.split("\n")
    for line in lines:
        if line.startswith("#") or "|" in line:
            print(line)


if __name__ == "__main__":
    main()
