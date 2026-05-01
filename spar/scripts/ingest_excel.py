#!/usr/bin/env python3
"""CLI: Excel column 값을 dictionary/acronyms.json keyword 섹션으로 병합."""
from __future__ import annotations

import argparse
from pathlib import Path

from spar.ingest.excel_loader import load_excel_terms, merge_into_acronyms

_DEFAULT_ACRONYMS = Path(__file__).parent.parent / "dictionary" / "acronyms.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Excel column 값을 acronyms.json keywords 섹션에 병합합니다."
    )
    parser.add_argument("--file", required=True, help="Excel 파일 경로 (.xlsx)")
    parser.add_argument("--columns", nargs="+", required=True, help="추출할 column명 (공백 구분)")
    parser.add_argument(
        "--acronyms",
        default=str(_DEFAULT_ACRONYMS),
        help=f"acronyms.json 경로 (기본값: {_DEFAULT_ACRONYMS})",
    )
    args = parser.parse_args()

    terms = load_excel_terms(args.file, args.columns)
    print(f"추출된 term 수: {len(terms)} (파일: {args.file})")

    merge_into_acronyms(terms, args.acronyms)
    print(f"병합 완료 → {args.acronyms}")

    sample = sorted(terms)[:20]
    for term in sample:
        print(f"  + {term}")
    if len(terms) > 20:
        print(f"  ... 외 {len(terms) - 20}개")


if __name__ == "__main__":
    main()
