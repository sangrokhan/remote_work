from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from extractor.pipeline import extract_pdf_to_outputs
from extractor.raw import load_raw_payload

TABLE_REF_PATTERN = re.compile(r"\[[A-Za-z0-9._-]+_tables\.md - Table \d+\]")
TABLE_HEADER_PATTERN = re.compile(r"^\[[A-Za-z0-9._-]+_tables\.md - Table \d+\]$", re.MULTILINE)
TABLE_HEADER_CAPTURE_RE = re.compile(r"^\[(?P<document_id>[A-Za-z0-9._-]+)_tables\.md - Table (?P<table_no>\d+)\]$")
DEFAULT_ADD_HEADING = Path(__file__).resolve().parent.parent / "fixtures" / "font_heading_profile.sample.json"


def _normalize_stem(raw_stem: str) -> str:
    return raw_stem.removeprefix("raw-")


def _decode_raw_to_pdf(raw_path: Path, output_pdf: Path, force: bool) -> Path:
    if output_pdf.exists() and not force:
        return output_pdf

    payload = load_raw_payload(raw_path)
    encoded = payload.get("document_pdf_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"{raw_path} is missing document_pdf_base64")

    pdf_bytes = base64.b64decode(encoded)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.write_bytes(pdf_bytes)
    return output_pdf


def _page_count(pdf_path: Path) -> int:
    try:
        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        return 0


def _run_parser(
    *,
    pdf_path: Path | None,
    raw_path: Path | None,
    out_root: Path,
    stem: str,
    add_heading: Path | None = None,
) -> dict[str, Any]:
    if raw_path is None and pdf_path is None:
        raise ValueError("pdf_path or raw_path is required")

    md_dir = out_root / "md"
    image_dir = out_root / "images"
    result = extract_pdf_to_outputs(
        pdf_path=pdf_path,
        out_md_dir=md_dir,
        out_image_dir=image_dir,
        stem=stem,
        from_raw=raw_path,
        add_heading=add_heading,
    )
    return {
        "text_chars": len((result["text_file"]).read_text(encoding="utf-8")),
        "table_count": result["summary"]["table_count"],
        "text_file": result["text_file"],
        "table_md_file": result["table_md_file"],
        "markdown": result["markdown"],
        "table_markdown": result["table_markdown"],
    }


def _count_table_references(markdown_text: str) -> int:
    return len(TABLE_REF_PATTERN.findall(markdown_text))


def _count_table_sections(markdown_text: str) -> int:
    return len(TABLE_HEADER_PATTERN.findall(markdown_text))


def _extract_table_references(markdown_text: str) -> list[str]:
    raw = TABLE_REF_PATTERN.findall(markdown_text)
    return [token.replace("[", "").replace("]", "") for token in raw]


def _extract_table_sections(markdown_text: str) -> list[str]:
    sections: list[str] = []
    for line in markdown_text.splitlines():
        match = TABLE_HEADER_CAPTURE_RE.match(line)
        if not match:
            continue
        sections.append(f"{match.group('document_id')}_tables.md - Table {match.group('table_no')}")
    return sections


def _normalize_table_token(token: str) -> str:
    return re.sub(r"\s+", " ", token.strip()).lower()


def _build_suspicions(
    *,
    parse_direct: dict[str, Any],
    parse_from_raw: dict[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []

    direct_refs = _count_table_references(parse_direct["markdown"])
    direct_sections = _count_table_sections(parse_direct["table_markdown"])
    direct_table_count = parse_direct["table_count"]
    direct_ref_tokens = _extract_table_references(parse_direct["markdown"])
    direct_section_tokens = _extract_table_sections(parse_direct["table_markdown"])

    if direct_refs != direct_sections:
        missing_in_markdown = sorted(
            {
                token
                for token in direct_ref_tokens
                if _normalize_table_token(token) not in {
                    _normalize_table_token(section) for section in direct_section_tokens
                }
            }
        )
        missing_in_reference = sorted(
            {
                section
                for section in direct_section_tokens
                if _normalize_table_token(section) not in {
                    _normalize_table_token(token) for token in direct_ref_tokens
                }
            }
        )
        if missing_in_markdown:
            reasons.append(
                f"본문 참조만 존재: {', '.join(missing_in_markdown[:10])}"
            )
        if missing_in_reference:
            reasons.append(
                f"table markdown에서만 존재: {', '.join(sorted(missing_in_reference)[:10])}"
            )
        reasons.append(
            f"body에는 table reference 수치 {direct_refs}개, table markdown 섹션은 {direct_sections}개로 불일치"
        )
    if direct_sections != direct_table_count:
        reasons.append(
            f"summary.table_count={direct_table_count}와 table markdown 섹션 수({direct_sections})가 불일치"
        )

    if parse_from_raw is not None:
        if parse_direct["table_count"] != parse_from_raw["table_count"]:
            reasons.append(
                "pdf 경로 파싱과 --from-raw 파싱의 table_count가 다름"
            )
        if parse_direct["table_markdown"] != parse_from_raw["table_markdown"]:
            reasons.append(
                "pdf 경로 파싱과 --from-raw 파싱의 table markdown 내용이 다름"
            )
        if parse_direct["text_chars"] != parse_from_raw["text_chars"]:
            reasons.append("pdf 경로 파싱과 --from-raw 파싱의 본문 길이가 다름")

    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct visual PDFs from raw dumps and inspect parser output.")
    parser.add_argument("--samples-dir", default="samples", help="raw dump 폴더 (기본: samples)")
    parser.add_argument(
        "--pdf-dir",
        default="artifacts/sample_visuals/pdfs",
        help="덤프를 PDF로 복원해 저장할 폴더",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/sample_visuals/parse",
        help="파싱 산출물을 저장할 루트 폴더",
    )
    parser.add_argument(
        "--pattern",
        default="*.dump",
        help="샘플 선택 패턴 (예: raw-*.dump)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 PDF/파싱 산출물을 덮어쓰기",
    )
    parser.add_argument(
        "--compare-from-raw",
        action="store_true",
        help="동일 샘플을 --from-raw로 다시 파싱해 PDF 파싱 결과와 비교",
    )
    parser.add_argument(
        "--heading-profile",
        default=str(DEFAULT_ADD_HEADING),
        help="헤딩 프로파일 json 경로(기본값 fixtures/font_heading_profile.sample.json).",
    )
    parser.add_argument(
        "--remove-heading",
        action="store_true",
        help="기본/지정 heading 적용을 비활성화합니다.",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir).resolve()
    pdf_dir = Path(args.pdf_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    add_heading = None if args.remove_heading else Path(args.heading_profile)
    if add_heading is not None and not add_heading.exists():
        print(f"warning: heading file not found: {add_heading}")
        add_heading = None

    raw_paths = sorted(samples_dir.glob(args.pattern))
    if not raw_paths:
        print(f"no sample found in {samples_dir} with pattern '{args.pattern}'")
        return

    report: list[dict[str, Any]] = []

    for raw_path in raw_paths:
        stem = raw_path.stem
        stem = _normalize_stem(stem)
        output_pdf = pdf_dir / f"{stem}.pdf"
        sample_root = out_root / stem

        print(f"\n[{stem}] raw: {raw_path}")

        pdf_path = _decode_raw_to_pdf(raw_path, output_pdf, force=args.force)
        page_count = _page_count(pdf_path)

        parse_direct = _run_parser(
            pdf_path=pdf_path,
            raw_path=None,
            out_root=sample_root / "pdf_path",
            stem=stem,
            add_heading=add_heading,
        )
        print(f"  visual pdf: {pdf_path} ({page_count} pages)")
        print(f"  parsed (pdf): {parse_direct['text_chars']} chars, tables={parse_direct['table_count']}")

        entry: dict[str, Any] = {
            "sample": str(raw_path),
            "pdf": str(pdf_path),
            "pages": page_count,
            "text_chars_pdf": parse_direct["text_chars"],
            "table_count_pdf": parse_direct["table_count"],
            "text_file": str(parse_direct["text_file"]),
            "table_md_file": str(parse_direct["table_md_file"]),
        }

        if args.compare_from_raw:
            parse_from_raw = _run_parser(
                pdf_path=None,
                raw_path=raw_path,
                out_root=sample_root / "from_raw",
                stem=f"{stem}_from_raw",
                add_heading=add_heading,
            )
            markdown_eq = parse_direct["markdown"] == parse_from_raw["markdown"]
            table_markdown_eq = parse_direct["table_markdown"] == parse_from_raw["table_markdown"]

            entry.update(
                {
                    "text_chars_from_raw": parse_from_raw["text_chars"],
                    "table_count_from_raw": parse_from_raw["table_count"],
                    "markdown_matches": markdown_eq,
                    "table_markdown_matches": table_markdown_eq,
                }
            )
            print(
                "  compare --from-raw: "
                f"markdown_same={markdown_eq}, table_markdown_same={table_markdown_eq}"
            )
            suspicions = _build_suspicions(parse_direct=parse_direct, parse_from_raw=parse_from_raw)
        else:
            suspicions = _build_suspicions(parse_direct=parse_direct, parse_from_raw=None)

        print(f"  suspicious cases: {'none' if not suspicions else '; '.join(suspicions)}")

        entry.update({"suspicious_cases": suspicions})

        report.append(entry)

    summary_path = out_root / "sample_raw_visualization_report.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport: {summary_path}")


if __name__ == "__main__":
    main()
