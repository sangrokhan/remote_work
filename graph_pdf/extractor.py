from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Sequence

import pdfplumber


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _is_watermark_line(text: str) -> bool:
    return "CONFIDENTIAL" in (text or "").upper()


def _repair_watermark_bleed(text: str) -> str:
    # Some rotated/conflicting watermark text can leak as a trailing single letter.
    text = re.sub(r"\s+[A-Za-z]$", "", text)
    return text.strip()


def _extract_body_text(page: pdfplumber.page.PageObject, header_margin: float, footer_margin: float) -> str:
    body_bbox = (0, footer_margin, page.width, page.height - header_margin)
    body_page = page.crop(body_bbox)
    raw = body_page.extract_text(x_tolerance=1.5, y_tolerance=2) or ""
    lines = []
    for line in raw.splitlines():
        fixed = _repair_watermark_bleed(line.strip())
        if fixed and not re.fullmatch(r"^[A-Za-z]$", fixed):
            lines.append(fixed)

    # Remove line-level watermark fragments that appear as many consecutive single letters.
    filtered = []
    i = 0
    while i < len(lines):
        if re.fullmatch(r"^[A-Za-z]$", lines[i]):
            run_start = i
            while i < len(lines) and re.fullmatch(r"^[A-Za-z]$", lines[i]):
                i += 1
            run_len = i - run_start
            if run_len < 6:
                filtered.extend(lines[run_start:i])
        else:
            filtered.append(lines[i])
            i += 1

    return "\n".join(ln for ln in filtered if not _is_watermark_line(ln))


def _merge_cells(table: Sequence[Sequence[str]]) -> List[List[str]]:
    out = []
    for row in table:
        out.append([str(cell or "").replace("\n", " ").strip() for cell in row])
    return out


def _looks_like_table(table: Sequence[Sequence[str]]) -> bool:
    if len(table) < 2:
        return False

    row_count = len(table)
    if row_count > 20:
        return False

    max_cols = max(len(r) for r in table)
    if max_cols < 2:
        return False

    if not any(cell.strip() for cell in table[0]):
        return False

    non_empty_cells = sum(1 for row in table for cell in row if str(cell).strip())
    if non_empty_cells < (max_cols * 2):
        return False

    return True


def _table_regions(page: pdfplumber.page.PageObject, x_tolerance: float = 5.0) -> List[tuple]:
    candidates = []
    for edge in page.horizontal_edges:
        if edge["top"] < 80 or edge["top"] > page.height - 80:
            continue
        if edge["x1"] - edge["x0"] < 120:
            continue

        placed = False
        for region in candidates:
            if abs(region["x0"] - edge["x0"]) < x_tolerance and abs(region["x1"] - edge["x1"]) < x_tolerance:
                region["lines"].append(edge)
                placed = True
                break

        if not placed:
            candidates.append({"x0": edge["x0"], "x1": edge["x1"], "lines": [edge]})

    # Keep table-shaped groups.
    return [
        (group["x0"], group["x1"], group["lines"])
        for group in candidates
        if len(group["lines"]) >= 4
    ]


def _extract_tables_from_crop(
    page: pdfplumber.page.PageObject,
    crop_bbox: tuple[float, float, float, float],
) -> List[List[List[str]]]:
    x0, y0, x1, y1 = crop_bbox
    crop = page.crop(crop_bbox)

    # Keep left/right outer lines to avoid merged adjacent-table results.
    y_min, y_max = y0, y1
    v_lines = []
    for edge in page.vertical_edges:
        if edge["x0"] < x0 or edge["x0"] > x1:
            continue
        if edge["top"] > y_max or edge["bottom"] < y_min:
            continue
        v_lines.append(edge["x0"])

    explicit_v = sorted({x0, x1, *v_lines})

    candidates = [
        {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "explicit_vertical_lines": explicit_v,
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "intersection_tolerance": 2,
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
        },
        {
            "vertical_strategy": "text",
            "horizontal_strategy": "lines",
            "explicit_vertical_lines": explicit_v,
            "snap_tolerance": 4,
            "join_tolerance": 4,
            "intersection_tolerance": 2,
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
        },
    ]

    for settings in candidates:
        tables = crop.extract_tables(table_settings=settings) or []
        cleaned = [
            _merge_cells(table)
            for table in tables
            if _looks_like_table(table)
        ]
        if cleaned:
            return cleaned
    return []


def _extract_tables(page: pdfplumber.page.PageObject) -> List[List[List[str]]]:
    table_regions = _table_regions(page)
    extracted: list[tuple] = []
    merged = []

    # Targeted extraction from table-like regions with missing outer vertical borders.
    for x0, x1, lines in table_regions:
        y0 = min(edge["top"] for edge in lines) - 2
        y1 = max(edge["top"] for edge in lines) + 2
        crop_bbox = (max(0.0, x0), max(0.0, y0), min(page.width, x1), min(page.height, y1))
        for table in _extract_tables_from_crop(page, crop_bbox):
            key = tuple(tuple(row) for row in table)
            if key not in extracted:
                extracted.append(key)
                merged.append(table)

    # Fallback: page-wide extraction for any remaining structure.
    if merged:
        return merged

    fallback_settings = [
        {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "intersection_tolerance": 2,
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
        },
        {
            "vertical_strategy": "text",
            "horizontal_strategy": "lines",
            "snap_tolerance": 4,
            "join_tolerance": 4,
            "intersection_tolerance": 2,
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
        },
        {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 4,
            "join_tolerance": 4,
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
        },
    ]

    for settings in fallback_settings:
        tables = page.extract_tables(table_settings=settings) or []
        cleaned = [_merge_cells(table) for table in tables if _looks_like_table(table)]
        if cleaned:
            return cleaned

    return []


def _md_table_from_rows(rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    align = ["---" for _ in header]
    body = rows[1:]

    def _row_to_md(cols: Sequence[str]) -> str:
        return "| " + " | ".join(_normalize_text(c) for c in cols) + " |"

    lines = [_row_to_md(header), _row_to_md(align)]
    lines.extend(_row_to_md(row) for row in body)
    return "\n".join(lines)


def extract_pdf_to_outputs(
    pdf_path: Path,
    out_md_dir: Path,
    out_image_dir: Path,
    stem: str,
    header_margin: float = 90,
    footer_margin: float = 40,
) -> dict:
    out_md_dir.mkdir(parents=True, exist_ok=True)
    out_image_dir.mkdir(parents=True, exist_ok=True)

    output_text = []
    output_tables = []
    image_files: List[Path] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            page_text = _extract_body_text(page, header_margin=header_margin, footer_margin=footer_margin)
            if page_text.strip():
                output_text.append(f"### Page {page_idx}\n{page_text}")

            tables = _extract_tables(page)
            if tables:
                for table_idx, table in enumerate(tables, start=1):
                    markdown_table = _md_table_from_rows(table)
                    if markdown_table:
                        output_tables.append(
                            f"### Page {page_idx} table {table_idx}\n{markdown_table}"
                        )

            # Save a full-page raster image. This is useful for multimodal indexing pipelines.
            image = page.to_image(resolution=170)
            image_file = out_image_dir / f"{stem}_page_{page_idx:02d}.png"
            image.save(str(image_file), format="png")
            image_files.append(image_file)

    markdown = "\n\n".join(output_text)
    if output_tables:
        markdown = markdown + "\n\n" + "\n\n".join(output_tables)

    text_file = out_md_dir / f"{stem}.txt"
    md_file = out_md_dir / f"{stem}.md"
    text_file.write_text(markdown, encoding="utf-8")

    pure_text = "\n\n".join(output_text)
    md_file.write_text(pure_text, encoding="utf-8")

    summary = {
        "pdf": str(pdf_path),
        "text_file": str(text_file),
        "md_file": str(md_file),
        "images": [str(p) for p in image_files],
        "table_count": sum(1 for _ in output_tables),
    }
    summary_file = out_md_dir / f"{stem}_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "markdown": markdown,
        "text_file": text_file,
        "md_file": md_file,
        "image_files": image_files,
        "summary": summary,
    }


if __name__ == "__main__":  # basic manual run
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--out-md-dir", default="graph_pdf/artifacts/md")
    parser.add_argument("--out-image-dir", default="graph_pdf/artifacts/images")
    parser.add_argument("--stem", default="output")
    args = parser.parse_args()

    extract_pdf_to_outputs(
        pdf_path=Path(args.pdf_path),
        out_md_dir=Path(args.out_md_dir),
        out_image_dir=Path(args.out-image_dir),
        stem=args.stem,
    )
