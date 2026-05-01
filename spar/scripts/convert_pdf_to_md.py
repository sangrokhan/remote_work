#!/usr/bin/env python3
"""PDF → markdown 변환 CLI (doc-type별 파서 분기).

PRD Task 1.1: 실 구현(파서)은 doc-type별로 별도 PR에서 채움. 본 스켈레톤은
인터페이스 + dispatch만 확정.

Usage:
    python scripts/convert_pdf_to_md.py --input-file foo.pdf \\
        --doc-type parameter_ref --output-dir data/skt-md/parameter_ref/
    python scripts/convert_pdf_to_md.py --input-dir data/skt-pdf/ \\
        --doc-type mop --output-dir data/skt-md/mop/
    python scripts/convert_pdf_to_md.py --input-file ... --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spar.retrieval.milvus_client import DOC_TYPES

# spec은 외부 markdown 코퍼스 — PDF 변환 대상 아님
PDF_DOC_TYPES = [dt for dt in DOC_TYPES if dt != "spec"]


def parse_pdf(pdf_path: Path, doc_type: str) -> str:
    """doc-type별 PDF→md 변환 라우터.

    TODO(Task 1.1): 각 분기에 실제 라이브러리 호출 채우기.
      parameter_ref/counter_ref/alarm_ref → camelot/pdfplumber (표 추출 강화)
      mop/install_guide                  → unstructured (절차 헤더 보존)
      feature_desc                        → pdfplumber (일반 텍스트)
      release_notes                       → unstructured (변경 항목 단위)
    """
    raise NotImplementedError(
        f"PDF parser for doc_type={doc_type!r} not yet implemented (Task 1.1)"
    )


def convert_file(
    pdf: Path, doc_type: str, output_dir: Path, *, dry_run: bool
) -> Path | None:
    out = output_dir / pdf.with_suffix(".md").name
    print(f"Convert: {pdf}  →  {out}  [doc_type={doc_type}]")
    if dry_run:
        print(f"  [DRY RUN] parser={doc_type} (skeleton only — Task 1.1 미구현)")
        return None
    md_text = parse_pdf(pdf, doc_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(md_text, encoding="utf-8")
    return out


def convert_directory(
    input_dir: Path, doc_type: str, output_dir: Path, *, dry_run: bool
) -> None:
    pdfs = sorted(input_dir.rglob("*.pdf"))
    if not pdfs:
        print(f"No .pdf files under {input_dir}")
        return
    for p in pdfs:
        convert_file(p, doc_type, output_dir, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF → markdown 변환")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-file", type=Path)
    src.add_argument("--input-dir", type=Path)
    parser.add_argument(
        "--doc-type", required=True, choices=PDF_DOC_TYPES,
        help=f"PDF 변환 대상 유형 (선택: {', '.join(PDF_DOC_TYPES)}); 3GPP 표준 문서는 이미 md",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.input_file and not args.input_file.exists():
        print(f"ERROR: file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    if args.input_dir and not args.input_dir.is_dir():
        print(f"ERROR: not a directory: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.input_file:
        convert_file(args.input_file, args.doc_type, args.output_dir,
                     dry_run=args.dry_run)
    else:
        convert_directory(args.input_dir, args.doc_type, args.output_dir,
                          dry_run=args.dry_run)


if __name__ == "__main__":
    main()
