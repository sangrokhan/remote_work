from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import extract_pdf_to_outputs
from .shared import _parse_pages_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--out-md-dir", default="graph_pdf/artifacts/md")
    parser.add_argument("--out-image-dir", default="graph_pdf/artifacts/images")
    parser.add_argument("--stem", default="output")
    parser.add_argument("--pages", help="1-based pages like 1,3,5-8")
    parser.add_argument("--force-table", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-watermark", action="store_true")
    args = parser.parse_args()

    extract_pdf_to_outputs(
        pdf_path=Path(args.pdf_path),
        out_md_dir=Path(args.out_md_dir),
        out_image_dir=Path(args.out_image_dir),
        stem=args.stem,
        pages=_parse_pages_spec(args.pages) if args.pages else None,
        force_table=args.force_table,
        debug=args.debug,
        debug_watermark=args.debug_watermark,
    )


if __name__ == "__main__":
    main()
