#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from spar.parsers.docx_config import DocxParseConfig
from spar.parsers.docx_parser import DocxParser


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse .docx to Markdown + CSV/images")
    parser.add_argument("--file", required=True, type=Path, help=".docx 파일 경로")
    parser.add_argument("--output", default=Path("output"), type=Path, help="출력 디렉토리")
    parser.add_argument("--heading-depth", default=2, type=int, help="섹션 경계 heading 레벨 (1~3)")
    args = parser.parse_args()

    cfg = DocxParseConfig(
        heading_depth=args.heading_depth,
        output_dir=args.output,
    )
    result = DocxParser(cfg).parse(args.file)

    md_path = args.output / (args.file.stem + ".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(result.markdown, encoding="utf-8")

    print(f"Parsed: {md_path}")
    print(f"Tables: {len(result.tables)} → {args.output / 'tables'}/")
    print(f"Images: {len(result.images)} → {args.output / 'images'}/")


if __name__ == "__main__":
    main()
