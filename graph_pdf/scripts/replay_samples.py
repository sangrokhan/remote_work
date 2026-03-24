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

TABLE_REF_PATTERN = re.compile(r"\[Table reference:\s*Page\s+\d+\s+table\s+\d+\]")
TABLE_HEADER_PATTERN = re.compile(r"^### Page \d+ table \d+$", re.MULTILINE)
NOTE_MAX_ROW_COUNT = 12
NOTE_MAX_VERTICAL_GAP = 40.0
NOTE_MIN_X_OVERLAP = 20.0


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
    debug: bool = False,
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
        debug=debug,
    )
    return {
        "text_chars": len((result["text_file"]).read_text(encoding="utf-8")),
        "table_count": result["summary"]["table_count"],
        "text_file": result["text_file"],
        "table_md_file": result["table_md_file"],
        "markdown": result["markdown"],
        "table_markdown": result["table_markdown"],
        "debug_file": result["debug_file"],
    }


def _bbox_overlap_x(a: list[float], b: list[float]) -> float:
    a0, a1 = a[0], a[2]
    b0, b1 = b[0], b[2]
    return max(0.0, min(a1, b1) - max(a0, b0))


def _is_single_column_note(table_meta: dict[str, Any]) -> bool:
    col_count = int(table_meta.get("col_count", 0) or 0)
    row_count = int(table_meta.get("row_count", 0) or 0)
    bbox = table_meta.get("bbox") or []
    if len(bbox) != 4:
        return False
    width = float(bbox[2]) - float(bbox[0])
    return col_count == 1 and 1 <= row_count <= NOTE_MAX_ROW_COUNT and width > NOTE_MIN_X_OVERLAP


def _find_overlap_candidates(
    debug_file: Path | None,
    max_vertical_gap: float = NOTE_MAX_VERTICAL_GAP,
    min_x_overlap: float = NOTE_MIN_X_OVERLAP,
) -> dict[str, Any]:
    if debug_file is None:
        return {
            "note_overlap_candidates": [],
            "table_overlap_candidates": [],
            "overlap_pages": [],
            "note_pages": [],
            "table_overlap_pages": [],
        }

    path = Path(debug_file)
    if not path.exists():
        return {
            "note_overlap_candidates": [],
            "table_overlap_candidates": [],
            "overlap_pages": [],
            "note_pages": [],
            "table_overlap_pages": [],
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    note_overlap_candidates: list[dict[str, Any]] = []
    table_overlap_candidates: list[dict[str, Any]] = []
    overlap_pages: set[int] = set()
    note_pages: set[int] = set()
    table_overlap_pages: set[int] = set()

    for page_payload in payload.get("pages", []):
        page_no = int(page_payload.get("page", 0))
        tables: list[dict[str, Any]] = [
            table for table in page_payload.get("tables", [])
            if len(table.get("bbox", [])) == 4
        ]
        if len(tables) < 2:
            continue

        sorted_tables = sorted(
            tables,
            key=lambda table: float(table.get("bbox", [0, 0, 0, 0])[1]),
        )

        for idx, upper in enumerate(sorted_tables):
            upper_bbox = list(map(float, upper.get("bbox", [0, 0, 0, 0])))
            upper_bottom = upper_bbox[3]
            for lower in sorted_tables[idx + 1 :]:
                lower_bbox = list(map(float, lower.get("bbox", [0, 0, 0, 0])))
                lower_top = lower_bbox[1]
                gap = lower_top - upper_bottom
                if gap < 0:
                    continue
                if gap > max_vertical_gap:
                    break

                x_overlap = _bbox_overlap_x(upper_bbox, lower_bbox)
                if x_overlap < min_x_overlap:
                    continue

                candidate = {
                    "page": page_no,
                    "upper": {
                        "col_count": int(upper.get("col_count", 0) or 0),
                        "row_count": int(upper.get("row_count", 0) or 0),
                        "bbox": [
                            round(float(upper_bbox[0]), 2),
                            round(float(upper_bbox[1]), 2),
                            round(float(upper_bbox[2]), 2),
                            round(float(upper_bbox[3]), 2),
                        ],
                    },
                    "lower": {
                        "col_count": int(lower.get("col_count", 0) or 0),
                        "row_count": int(lower.get("row_count", 0) or 0),
                        "bbox": [
                            round(float(lower_bbox[0]), 2),
                            round(float(lower_bbox[1]), 2),
                            round(float(lower_bbox[2]), 2),
                            round(float(lower_bbox[3]), 2),
                        ],
                    },
                    "vertical_gap": round(gap, 2),
                    "x_overlap": round(x_overlap, 2),
                }

                upper_is_note = _is_single_column_note(upper)
                lower_is_note = _is_single_column_note(lower)
                if upper_is_note or lower_is_note:
                    note_overlap_candidates.append(candidate)
                    note_pages.add(page_no)
                else:
                    table_overlap_candidates.append(candidate)
                    table_overlap_pages.add(page_no)
                overlap_pages.add(page_no)

    return {
        "note_overlap_candidates": note_overlap_candidates,
        "table_overlap_candidates": table_overlap_candidates,
        "overlap_pages": sorted(overlap_pages),
        "note_pages": sorted(note_pages),
        "table_overlap_pages": sorted(table_overlap_pages),
    }


def _count_table_references(markdown_text: str) -> int:
    return len(TABLE_REF_PATTERN.findall(markdown_text))


def _count_table_sections(markdown_text: str) -> int:
    return len(TABLE_HEADER_PATTERN.findall(markdown_text))


def _extract_table_references(markdown_text: str) -> list[str]:
    raw = TABLE_REF_PATTERN.findall(markdown_text)
    return [token.replace("[", "").replace("]", "") for token in raw]


def _extract_table_sections(markdown_text: str) -> list[str]:
    return [line[4:] for line in markdown_text.splitlines() if TABLE_HEADER_PATTERN.match(line)]


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
                    _normalize_table_token(f"Table reference: {section}") for section in direct_section_tokens
                }
            }
        )
        missing_in_reference = sorted(
            {
                f"Table reference: {section}"
                for section in direct_section_tokens
                if _normalize_table_token(f"Table reference: {section}") not in {
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
            f"body에는 Table reference 수치 {direct_refs}개, table markdown 섹션은 {direct_sections}개로 불일치"
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
        "--analyze-note-overlap",
        action="store_true",
        help="debug 기반으로 겹침 후보를 분석해 single-column(note) 후보와 table 겹침을 분리 출력",
    )
    parser.add_argument(
        "--max-overlap-gap",
        type=float,
        default=NOTE_MAX_VERTICAL_GAP,
        help="겹침 후보 판별에서 상단/하단 블록 간 최대 간격(기본값: 40.0)",
    )
    parser.add_argument(
        "--min-x-overlap",
        type=float,
        default=NOTE_MIN_X_OVERLAP,
        help="겹침 후보 판별에서 최소 x 축 교차 길이(기본값: 20.0)",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir).resolve()
    pdf_dir = Path(args.pdf_dir).resolve()
    out_root = Path(args.out_dir).resolve()

    raw_paths = sorted(samples_dir.glob(args.pattern))
    if not raw_paths:
        print(f"no sample found in {samples_dir} with pattern '{args.pattern}'")
        return

    report: list[dict[str, Any]] = []

    for raw_path in raw_paths:
        stem = raw_path.stem
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
            debug=args.compare_from_raw or args.analyze_note_overlap,
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

        if args.analyze_note_overlap:
            overlap = _find_overlap_candidates(
                debug_file=parse_direct.get("debug_file"),
                max_vertical_gap=args.max_overlap_gap,
                min_x_overlap=args.min_x_overlap,
            )
            print(
                f"  overlap candidates: "
                f"note={len(overlap['note_overlap_candidates'])}, "
                f"table={len(overlap['table_overlap_candidates'])}"
            )
            sample_root.mkdir(parents=True, exist_ok=True)
            overlap_cases_path = sample_root / f"{stem}_overlap_cases.json"
            overlap_cases_path.write_text(
                json.dumps(
                    {
                        "sample": raw_path.name,
                        "pdf": str(pdf_path),
                        "note_overlap_candidates": overlap["note_overlap_candidates"],
                        "table_overlap_candidates": overlap["table_overlap_candidates"],
                        "overlap_pages": overlap["overlap_pages"],
                        "note_pages": overlap["note_pages"],
                        "table_overlap_pages": overlap["table_overlap_pages"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            note_only_path = sample_root / f"{stem}_note_only.json"
            note_only_path.write_text(
                json.dumps(
                    {
                        "sample": raw_path.name,
                        "pdf": str(pdf_path),
                        "note_candidates": overlap["note_overlap_candidates"],
                        "note_pages": overlap["note_pages"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            entry.update(
                {
                    "overlap_candidates": overlap,
                    "note_overlap_candidates": overlap["note_overlap_candidates"],
                    "table_overlap_candidates": overlap["table_overlap_candidates"],
                    "overlap_pages": overlap["overlap_pages"],
                    "note_overlap_pages": overlap["note_pages"],
                    "table_overlap_pages": overlap["table_overlap_pages"],
                }
            )

        if args.compare_from_raw:
            parse_from_raw = _run_parser(
                pdf_path=None,
                raw_path=raw_path,
                out_root=sample_root / "from_raw",
                stem=f"{stem}_from_raw",
                debug=args.analyze_note_overlap,
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
