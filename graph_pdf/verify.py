from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from extractor import extract_pdf_to_outputs
from sample_generator import (
    WATERMARK_TEXT,
    create_demo_pdf,
    get_demo_tables,
    get_demo_text_lines,
)


def _normalize(value: str) -> str:
    v = str(value or "")
    v = v.replace("<br>", " <br> ").replace("\\n", " <br> ")
    v = re.sub(r"\s+", " ", v)
    return v.strip().lower()


def _parse_table_row(line: str) -> List[str]:
    text = line.strip()
    if not text.startswith("|") or not text.endswith("|"):
        return []

    is_separator = bool(re.fullmatch(r"\|\s*:?-{2,}:?(\s*\|\s*:?-{2,}:?\s*)+\|", text))
    if is_separator:
        return []

    body = text.strip("|")
    return [cell.strip() for cell in body.split("|")]


def _extract_markdown_tables(markdown_text: str) -> Dict[Tuple[str, ...], List[List[str]]]:
    lines = markdown_text.splitlines()
    tables: Dict[Tuple[str, ...], List[List[str]]] = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("### Page "):
            i += 1
            continue

        i += 1
        rows: List[List[str]] = []
        while i < len(lines) and lines[i].startswith("|"):
            parsed = _parse_table_row(lines[i])
            if parsed:
                rows.append(parsed)
            i += 1

        if len(rows) < 2:
            continue

        header = tuple(_normalize(cell) for cell in rows[0])
        tables[header] = rows[1:]

    return tables


def _row_matches(expected_row: Sequence[str], actual_row: Sequence[str]) -> bool:
    if len(expected_row) != len(actual_row):
        return False

    for e, a in zip(expected_row, actual_row):
        e_n = _normalize(e)
        a_n = _normalize(a)
        if not e_n:
            if a_n:
                return False
            continue
        if e_n != a_n:
            return False

    return True


def _contains_all_rows(
    expected_rows: Sequence[Sequence[str]],
    actual_rows: Sequence[Sequence[str]],
) -> None:
    normalized_actual = [tuple(_normalize(cell) for cell in row) for row in actual_rows]

    for expected in expected_rows:
        normalized_expected = tuple(_normalize(cell) for cell in expected)
        matched = any(_row_matches(normalized_expected, actual) for actual in normalized_actual)
        if not matched:
            raise AssertionError(f"missing expected table row: {expected}")


def run_checks() -> int:
    base = Path(__file__).resolve().parent
    root = base / "artifacts" / "verify"
    pdf_path = root / "sample_verify.pdf"
    md_dir = root / "md"
    image_dir = root / "images"

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    create_demo_pdf(pdf_path)

    result = extract_pdf_to_outputs(
        pdf_path=pdf_path,
        out_md_dir=md_dir,
        out_image_dir=image_dir,
        stem="verify",
    )

    markdown_text = result["markdown"]
    txt_text = result["text_file"].read_text(encoding="utf-8")
    lower_text = txt_text.lower()

    if _normalize(WATERMARK_TEXT) in _normalize(txt_text):
        raise AssertionError("watermark string remained in extracted text")

    header_footer_markers = (
        "graph pdf demo header",
        "prepared for table + text extraction tests",
        "graph pdf demo footer",
        "footer details: keep header/footer clean",
    )
    for marker in header_footer_markers:
        if marker in lower_text:
            raise AssertionError(f"layout marker was not removed: {marker}")

    for token in get_demo_text_lines()[:3]:
        if _normalize(token) not in lower_text:
            raise AssertionError(f"missing expected body text: {token}")

    extracted_tables = _extract_markdown_tables(markdown_text)
    demo_tables = get_demo_tables()

    for table_name, (header, rows) in demo_tables.items():
        normalized_header = tuple(_normalize(cell) for cell in header)
        if normalized_header not in extracted_tables:
            raise AssertionError(f"missing expected table: {table_name}")
        _contains_all_rows(rows, extracted_tables[normalized_header])

    if result["summary"]["table_count"] < len(demo_tables):
        raise AssertionError("expected table count is lower than fixture table definitions")

    all_rows = [row for rows in extracted_tables.values() for row in rows]
    if not any(row and _normalize(row[0]) == "" and any(_normalize(cell) for cell in row[1:]) for row in all_rows):
        raise AssertionError("merged/continuation-style rows were not present in table output")

    if len(result["image_files"]) < 1:
        raise AssertionError("expected at least one page image")

    print("[verify] PASS")
    print("text_file:", result["text_file"])
    print("md_file:", result["md_file"])
    print("image count:", len(result["image_files"]))
    print("table count:", result["summary"]["table_count"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run_checks())
    except Exception as e:
        print("[verify] FAIL:", e)
        sys.exit(1)
