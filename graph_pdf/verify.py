from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Sequence

from extractor import extract_pdf_to_outputs
from sample_generator import (
    WATERMARK_TEXT,
    create_demo_pdf,
    get_demo_tables,
    get_demo_text_lines,
)


def _normalize(value: str) -> str:
    v = str(value or "")
    v = v.replace("<br>", " ")
    v = v.replace("\n", " ").replace("\\n", " ")
    v = re.sub(r"\s+", " ", v)
    return v.strip().lower()


def _extract_markdown_tables(markdown_text: str) -> List[List[List[str]]]:
    lines = markdown_text.splitlines()
    tables: List[List[List[str]]] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("### Page ") or " table " not in line:
            i += 1
            continue

        i += 1
        table_lines: List[str] = []
        while i < len(lines) and not lines[i].startswith("### Page "):
            table_lines.append(lines[i])
            i += 1

        rows: List[List[str]] = []
        current_row: List[str] = []
        current_cell_idx = -1
        for raw in table_lines:
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("- Row "):
                if current_row:
                    rows.append(current_row)
                current_row = []
                current_cell_idx = -1
                continue

            field_match = re.match(r"^\s{2}[^:]+:\s*(.*)$", raw)
            if field_match:
                current_row.append(field_match.group(1).strip())
                current_cell_idx = len(current_row) - 1
                continue

            bullet_match = re.match(r"^\s{2}(- .+)$", raw)
            if bullet_match and current_row and current_cell_idx >= 0:
                current_row[current_cell_idx] = (
                    current_row[current_cell_idx] + "\n" + bullet_match.group(1).strip()
                ).strip()

        if current_row:
            rows.append(current_row)

        if rows:
            tables.append(rows)

    return tables


def _row_matches(expected_row: Sequence[str], actual_row: Sequence[str]) -> bool:
    if len(expected_row) != len(actual_row):
        return False

    def _contains(expected: str, actual: str) -> bool:
        e = _normalize(expected)
        a = _normalize(actual)
        if not e:
            return not a

        e_tokens = e.split()
        if len(e_tokens) == 1:
            return e in a or any(token == e for token in a.split())

        return e in a

    for e, a in zip(expected_row, actual_row):
        if not _contains(e, a):
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


def _row_values_present_in_text(expected_row: Sequence[str], text: str) -> bool:
    for cell in expected_row:
        normalized = _normalize(cell)
        if not normalized:
            continue
        if normalized not in text:
            return False
    return True


def run_checks() -> int:
    base = Path(__file__).resolve().parent
    root = base / "artifacts" / "verify"
    pdf_path = root / "sample_verify.pdf"
    md_dir = root / "md"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    create_demo_pdf(pdf_path)

    result = extract_pdf_to_outputs(
        pdf_path=pdf_path,
        out_md_dir=md_dir,
        out_image_dir=root / "images",
        stem="verify",
    )

    markdown_text = result["markdown"]
    txt_text = result["text_file"].read_text(encoding="utf-8")
    lower_text = txt_text.lower()
    normalized_text = _normalize(txt_text)

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
        if _normalize(token) not in normalized_text:
            raise AssertionError(f"missing expected body text: {token}")

    extracted_tables = _extract_markdown_tables(markdown_text)
    demo_tables = get_demo_tables()
    flattened_rows: List[Sequence[str]] = [row for rows in extracted_tables for row in rows]
    expected_rows = [row for _, rows in demo_tables.values() for row in rows]

    for expected_row in expected_rows:
        normalized_expected = tuple(_normalize(cell) for cell in expected_row)
        table_match = any(
            _row_matches(normalized_expected, [_normalize(cell) for cell in actual])
            for actual in flattened_rows
        )
        if table_match:
            continue

        if not _row_values_present_in_text(expected_row, normalized_text):
            raise AssertionError(f"missing expected table row: {expected_row}")

    if result["summary"]["table_count"] < len(demo_tables):
        raise AssertionError("expected table count is lower than fixture table definitions")

    if not any(not str(row[0]).strip() and any(str(cell).strip() for cell in row[1:]) for row in expected_rows):
        raise AssertionError("fixture does not include merged/continuation-style table rows")

    print("[verify] PASS")
    print("text_file:", result["text_file"])
    print("md_file:", result["md_file"])
    print("embedded images:", len(result["image_files"]))
    print("table count:", result["summary"]["table_count"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run_checks())
    except Exception as e:
        print("[verify] FAIL:", e)
        sys.exit(1)
