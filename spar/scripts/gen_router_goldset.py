#!/usr/bin/env python3
"""QA goldset (retrieval_goldset.jsonl) → router goldset 변환.

사용법:
    python scripts/gen_router_goldset.py
    python scripts/gen_router_goldset.py --input data/goldsets/retrieval_goldset.jsonl --output data/goldsets/router_goldset.jsonl
    python scripts/gen_router_goldset.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SPAR_ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = SPAR_ROOT / "data" / "goldsets" / "retrieval_goldset.jsonl"
DEFAULT_OUTPUT = SPAR_ROOT / "data" / "goldsets" / "router_goldset.jsonl"

TYPE_TO_ROUTE: dict[str, str] = {
    "definition": "definition_explain",
    "procedural": "procedural",
    "diagnostic": "diagnostic",
    "comparative": "comparative",
    "lookup": "structured_lookup",
}


def convert(
    items: list[dict],
    start_id: int = 1,
) -> tuple[list[dict], dict[str, int], int]:
    results: list[dict] = []
    counts: dict[str, int] = {}
    skipped = 0
    for item in items:
        qa_type = item.get("type", "")
        route = TYPE_TO_ROUTE.get(qa_type)
        if route is None:
            print(
                f"  WARN: 미매핑 type={qa_type!r}, query_id={item.get('query_id')} — 스킵",
                file=sys.stderr,
            )
            skipped += 1
            continue
        n = start_id + len(results)
        results.append({
            "query_id": f"RQ{n:04d}",
            "query": item["query"],
            "expected_route": route,
            "source_doc": item.get("source_doc", ""),
            "spec_number": item.get("spec_number", ""),
            "release": item.get("release", ""),
            "qa_query_id": item.get("query_id", ""),
        })
        counts[route] = counts.get(route, 0) + 1
    return results, counts, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="QA goldset → router goldset 변환")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, metavar="FILE")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, metavar="FILE")
    parser.add_argument("--append", action="store_true", help="이어쓰기 (기본: 덮어쓰기)")
    parser.add_argument("--dry-run", action="store_true", help="파일 미생성, 통계만 출력")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: 입력 파일 없음: {args.input}", file=sys.stderr)
        sys.exit(1)

    items = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  WARN: JSON 파싱 실패 — 스킵: {e}", file=sys.stderr)
    print(f"QA goldset {len(items)}개 로드")

    start_id = 1
    if args.append and args.output.exists():
        existing = sum(1 for l in args.output.read_text(encoding="utf-8").splitlines() if l.strip())
        start_id = existing + 1
        print(f"기존 {existing}개 항목에 이어쓰기. RQ{start_id:04d}부터 시작")

    results, counts, skipped = convert(items, start_id)

    route_summary = ", ".join(f"{r}={c}" for r, c in sorted(counts.items()))
    print(f"매핑: {route_summary}")
    if skipped:
        print(f"미매핑(스킵): {skipped}개")

    if args.dry_run:
        print("[DRY RUN] 파일 미생성")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with open(args.output, mode, encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"router_goldset.jsonl → {len(results)}개 저장")


if __name__ == "__main__":
    main()
