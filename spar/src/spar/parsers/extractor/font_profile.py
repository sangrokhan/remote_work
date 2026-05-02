from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pdfplumber

from .shared import _normalize_debug_color
from .tables import _extract_tables
from .text import _extract_body_word_lines


def _font_color_key(color: object) -> str:
    normalized = _normalize_debug_color(color)
    if isinstance(normalized, list):
        return ",".join(f"{float(value):.3f}" for value in normalized)
    if isinstance(normalized, (int, float)):
        return f"{float(normalized):.3f}"
    return str(normalized or "")


def _style_key(line: dict) -> Tuple[float, str]:
    font_size = round(float(line.get("dominant_font_size", line.get("size", 0.0)) or 0.0), 2)
    font_color = _font_color_key(line.get("color"))
    return font_size, font_color


def profile_pdf_fonts(
    pdf_path: Path,
    out_dir: Path,
    stem: str,
    header_margin: float = 90.0,
    footer_margin: float = 40.0,
    pages: Optional[Sequence[int]] = None,
    sample_limit: int = 3,
) -> dict:
    # This mode scans normalized body lines so large documents can be profiled without running table extraction.
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_pages = set(int(page_no) for page_no in (pages or []))
    style_map: Dict[Tuple[float, str], dict] = {}
    scanned_pages = 0

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            if selected_pages and page_no not in selected_pages:
                continue

            scanned_pages += 1
            excluded_bboxes = [bbox for _rows, bbox in _extract_tables(page, force_table=False)]
            line_payloads = _extract_body_word_lines(
                page=page,
                header_margin=header_margin,
                footer_margin=footer_margin,
                excluded_bboxes=excluded_bboxes,
            )
            for line in line_payloads:
                if bool(line.get("is_shape_text")):
                    continue
                font_size, font_color = _style_key(line)
                if font_size <= 0.0:
                    continue

                style = style_map.setdefault(
                    (font_size, font_color),
                    {
                        "font_size": font_size,
                        "font_color": font_color,
                        "line_count": 0,
                        "sample_page": 0,
                        "sample_texts": [],
                        "seen_pages": set(),
                    },
                )
                style["line_count"] += 1
                style["seen_pages"].add(page_no)
                if not style["sample_page"]:
                    style["sample_page"] = page_no
                text = str(line.get("text") or "").strip()
                if text and text not in style["sample_texts"] and len(style["sample_texts"]) < sample_limit:
                    style["sample_texts"].append(text)

    styles: List[dict] = []
    for entry in sorted(
        style_map.values(),
        key=lambda item: (-int(item["line_count"]), -float(item["font_size"]), str(item["font_color"])),
    ):
        styles.append(
            {
                "font_size": float(entry["font_size"]),
                "font_color": str(entry["font_color"]),
                "line_count": int(entry["line_count"]),
                "page_count": len(entry["seen_pages"]),
                "sample_page": int(entry["sample_page"]),
                "sample_texts": list(entry["sample_texts"]),
            }
        )

    payload = {
        "pdf": str(pdf_path),
        "page_count": scanned_pages,
        "style_count": len(styles),
        "styles": styles,
    }

    json_file = out_dir / f"{stem}_font_profile.json"
    csv_file = out_dir / f"{stem}_font_profile.csv"
    json_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["font_size", "font_color", "line_count", "page_count", "sample_page", "sample_texts"],
        )
        writer.writeheader()
        for entry in styles:
            writer.writerow(
                {
                    "font_size": f"{float(entry['font_size']):.2f}",
                    "font_color": str(entry["font_color"]),
                    "line_count": int(entry["line_count"]),
                    "page_count": int(entry["page_count"]),
                    "sample_page": int(entry["sample_page"]),
                    "sample_texts": " || ".join(entry["sample_texts"]),
                }
            )

    return {
        "profile": payload,
        "json_file": json_file,
        "csv_file": csv_file,
    }
