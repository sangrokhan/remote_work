from __future__ import annotations

import argparse
from pathlib import Path

from .font_profile import profile_pdf_fonts
from .pipeline import extract_pdf_to_outputs
from .raw import dump_pdf_to_raw_file, materialize_raw_dump
from .shared import _parse_pages_spec


def main() -> None:
    # Keep the CLI intentionally thin so the real behavior lives in pipeline.py.
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", nargs="?")
    parser.add_argument("--out-md-dir", default="graph_pdf/artifacts/md")
    parser.add_argument("--out-image-dir", default="graph_pdf/artifacts/images")
    parser.add_argument("--stem", default="output")
    parser.add_argument("--pages", help="1-based pages like 1,3,5-8")
    parser.add_argument("--force-table", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-watermark", action="store_true")
    parser.add_argument("--profile-fonts", action="store_true")
    parser.add_argument("--add-heading")
    parser.add_argument("--page-write", action="store_true")
    parser.add_argument("--raw")
    parser.add_argument("--from-raw")
    parser.add_argument("--region-log")
    args = parser.parse_args()

    selected_pages = _parse_pages_spec(args.pages) if args.pages else None
    if args.raw and args.from_raw:
        parser.error("--raw and --from-raw cannot be used together")
    if args.raw and not args.pdf_path:
        parser.error("pdf_path is required when using --raw")
    if not args.pdf_path and not args.from_raw:
        parser.error("pdf_path is required unless --from-raw is provided")

    if args.raw:
        dump_pdf_to_raw_file(
            pdf_path=Path(args.pdf_path),
            raw_path=Path(args.raw),
            pages=selected_pages,
        )
        return

    if args.profile_fonts:
        if args.from_raw:
            with materialize_raw_dump(Path(args.from_raw)) as (materialized_pdf_path, _raw_payload):
                profile_pdf_fonts(
                    pdf_path=materialized_pdf_path,
                    out_dir=Path(args.out_md_dir),
                    stem=args.stem,
                    pages=selected_pages,
                )
        else:
            profile_pdf_fonts(
                pdf_path=Path(args.pdf_path),
                out_dir=Path(args.out_md_dir),
                stem=args.stem,
                pages=selected_pages,
            )
        return

    extract_pdf_to_outputs(
        pdf_path=Path(args.pdf_path) if args.pdf_path else None,
        out_md_dir=Path(args.out_md_dir),
        out_image_dir=Path(args.out_image_dir),
        stem=args.stem,
        pages=selected_pages,
        force_table=args.force_table,
        debug=args.debug,
        debug_watermark=args.debug_watermark,
        add_heading=Path(args.add_heading) if args.add_heading else None,
        page_write=args.page_write,
        from_raw=Path(args.from_raw) if args.from_raw else None,
        region_log=Path(args.region_log) if args.region_log else None,
    )


if __name__ == "__main__":
    main()
