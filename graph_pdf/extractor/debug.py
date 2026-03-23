from __future__ import annotations

from collections import Counter
from typing import List

from .shared import (
    _build_segment_groups,
    _char_rotation_degrees,
    _merge_horizontal_band_segments,
    _merge_numeric_positions,
    _merge_vertical_band_segments,
    _normalize_debug_color,
    _round_graphic_object,
    _round_segment,
)
from .tables import _table_regions
from .text import _detect_body_bounds, _extract_body_text_lines, _extract_body_word_lines


def _build_text_profile(line_payloads: List[dict]) -> dict:
    # Text-size histograms are meant for debugging future document-structure heuristics.
    font_size_counter: Counter[float] = Counter()
    fontname_counter: Counter[str] = Counter()
    font_color_counter: Counter[str] = Counter()
    for line in line_payloads:
        for size in line.get("font_size_candidates", []):
            font_size_counter[round(float(size), 2)] += 1
        for fontname in line.get("fontnames", []):
            if str(fontname):
                fontname_counter[str(fontname)] += 1
        color_key = str(line.get("dominant_font_color") or "")
        if color_key:
            font_color_counter[color_key] += 1

    dominant_font_size = max(font_size_counter, key=font_size_counter.get) if font_size_counter else 0.0
    dominant_fontname = max(fontname_counter, key=fontname_counter.get) if fontname_counter else ""
    return {
        "line_count": len(line_payloads),
        "font_size_histogram": {
            f"{size:.2f}": count for size, count in sorted(font_size_counter.items())
        },
        "fontname_histogram": dict(sorted(fontname_counter.items())),
        "font_color_histogram": dict(sorted(font_color_counter.items())),
        "font_size_candidates": sorted(font_size_counter),
        "dominant_font_size": dominant_font_size,
        "dominant_fontname": dominant_fontname,
    }


def _color_key(color: object) -> str:
    normalized = _normalize_debug_color(color)
    if isinstance(normalized, list):
        return ",".join(f"{float(value):.3f}" for value in normalized)
    if isinstance(normalized, (int, float)):
        return f"{float(normalized):.3f}"
    return str(normalized or "")


def _collect_rotated_text_debug(page: "pdfplumber.page.Page", page_no: int) -> List[dict]:
    # Debug output keeps only rotated chars because non-rotated body text is already visible elsewhere.
    entries: List[dict] = []
    for char in getattr(page, "chars", []):
        rotation = _char_rotation_degrees(char)
        if abs(rotation) <= 0.1:
            continue
        entries.append(
            {
                "page": page_no,
                "text": str(char.get("text", "")),
                "rotation": rotation,
                "size": float(char.get("size", 0.0)),
                "x0": float(char.get("x0", 0.0)),
                "x1": float(char.get("x1", 0.0)),
                "top": float(char.get("top", 0.0)),
                "bottom": float(char.get("bottom", 0.0)),
                "matrix": list(char.get("matrix", ())),
                "non_stroking_color": char.get("non_stroking_color"),
                "stroking_color": char.get("stroking_color"),
                "fontname": char.get("fontname"),
            }
        )
    return entries


def _collect_table_drawing_debug(
    page: "pdfplumber.page.Page",
    page_no: int,
    header_margin: float = 90.0,
    footer_margin: float = 40.0,
) -> dict:
    # This payload explains how the extractor turned raw edges into table regions.
    body_top, body_bottom = _detect_body_bounds(page, header_margin=header_margin, footer_margin=footer_margin)
    groups = _table_regions(page)
    tables: List[dict] = []
    for index, (x0, x1, lines) in enumerate(groups, start=1):
        top = min(float(edge["top"]) for edge in lines)
        bottom = max(float(edge["top"]) for edge in lines)
        horizontal_edges = [
            edge
            for edge in lines
            if float(edge["top"]) >= body_top and float(edge["bottom"]) <= body_bottom
        ]
        vertical_edges = [
            edge
            for edge in page.vertical_edges
            if float(edge["x0"]) >= x0 - 2.0
            and float(edge["x0"]) <= x1 + 2.0
            and float(edge["bottom"]) >= top
            and float(edge["top"]) <= bottom
        ]
        horizontal_positions = _merge_numeric_positions([float(edge["top"]) for edge in horizontal_edges])
        vertical_positions = _merge_numeric_positions([x0, x1, *(float(edge["x0"]) for edge in vertical_edges)])
        horizontal_groups = _build_segment_groups(
            horizontal_edges,
            axis_key="top",
            merge_fn=_merge_horizontal_band_segments,
            tolerance=1.0,
            body_top=body_top,
            body_bottom=body_bottom,
        )
        vertical_groups = _build_segment_groups(
            vertical_edges,
            axis_key="x0",
            merge_fn=_merge_vertical_band_segments,
            tolerance=1.0,
            body_top=body_top,
            body_bottom=body_bottom,
        )
        tables.append(
            {
                "index": index,
                "bbox": [round(x0, 2), round(top, 2), round(x1, 2), round(bottom, 2)],
                "row_count": max(0, len(horizontal_positions) - 1),
                "col_count": max(0, len(vertical_positions) - 1),
                "horizontal_lines": [round(value, 2) for value in horizontal_positions],
                "vertical_lines": [round(value, 2) for value in vertical_positions],
                "horizontal_segments": [_round_segment(edge, body_top=body_top, body_bottom=body_bottom) for edge in horizontal_edges],
                "vertical_segments": [_round_segment(edge, body_top=body_top, body_bottom=body_bottom) for edge in vertical_edges],
                "horizontal_groups": horizontal_groups,
                "vertical_groups": vertical_groups,
                "horizontal_count": len(horizontal_positions),
                "vertical_count": len(vertical_positions),
            }
        )

    line_payloads = _extract_body_word_lines(page=page, header_margin=header_margin, footer_margin=footer_margin)
    for line in line_payloads:
        line["dominant_font_color"] = _color_key(line.get("color"))
    raw_text_lines, normalized_text_lines = _extract_body_text_lines(page, header_margin=header_margin, footer_margin=footer_margin)
    text_profile = _build_text_profile(line_payloads)

    return {
        "page": page_no,
        "body_bounds": [round(body_top, 2), round(body_bottom, 2)],
        "table_count": len(tables),
        "tables": tables,
        "source_drawings": {
            "lines": [
                _round_graphic_object(line, body_top=body_top, body_bottom=body_bottom)
                for line in getattr(page, "lines", [])
            ],
            "rects": [
                _round_graphic_object(rect, body_top=body_top, body_bottom=body_bottom)
                for rect in getattr(page, "rects", [])
            ],
            "curves": [
                _round_graphic_object(curve, body_top=body_top, body_bottom=body_bottom)
                for curve in getattr(page, "curves", [])
            ],
        },
        "text_debug": {
            "profile": text_profile,
            "raw_lines": raw_text_lines,
            "raw_line_boxes": [
                {
                    "text": str(line.get("text") or ""),
                    "x0": round(float(line.get("x0", 0.0)), 2),
                    "x1": round(float(line.get("x1", 0.0)), 2),
                    "top": round(float(line.get("top", 0.0)), 2),
                    "bottom": round(float(line.get("bottom", 0.0)), 2),
                    "size": round(float(line.get("size", 0.0)), 2),
                    "fontname": str(line.get("fontname") or ""),
                    "fontnames": list(line.get("fontnames", [])),
                    "dominant_font_size": round(float(line.get("dominant_font_size", 0.0)), 2),
                    "dominant_font_color": str(line.get("dominant_font_color") or ""),
                    "font_size_candidates": [
                        round(float(size), 2) for size in line.get("font_size_candidates", [])
                    ],
                    "text_start_x": round(float(line.get("text_start_x", line.get("x0", 0.0))), 2),
                    "marker_candidate": bool(line.get("marker_candidate")),
                }
                for line in line_payloads
            ],
            "normalized_lines": normalized_text_lines,
        },
    }


def _collect_page_edge_debug(
    page: "pdfplumber.page.Page",
    page_no: int,
    header_margin: float = 90.0,
    footer_margin: float = 40.0,
) -> dict:
    # Edge debug is lower-level than table debug and shows both all edges and the subset used for tables.
    body_top, body_bottom = _detect_body_bounds(page, header_margin=header_margin, footer_margin=footer_margin)
    groups = _table_regions(page)
    selected_horizontal_edges = []
    selected_vertical_edges = []
    for x0, x1, lines in groups:
        top = min(float(edge["top"]) for edge in lines)
        bottom = max(float(edge["top"]) for edge in lines)
        selected_horizontal_edges.extend(lines)
        selected_vertical_edges.extend(
            edge
            for edge in page.vertical_edges
            if float(edge["x0"]) >= x0 - 2.0
            and float(edge["x0"]) <= x1 + 2.0
            and float(edge["bottom"]) >= top
            and float(edge["top"]) <= bottom
        )

    return {
        "page": page_no,
        "body_bounds": [round(body_top, 2), round(body_bottom, 2)],
        "all_horizontal_edges": [_round_segment(edge, body_top=body_top, body_bottom=body_bottom) for edge in page.horizontal_edges],
        "selected_horizontal_edges": [_round_segment(edge, body_top=body_top, body_bottom=body_bottom) for edge in selected_horizontal_edges],
        "all_vertical_edges": [_round_segment(edge, body_top=body_top, body_bottom=body_bottom) for edge in page.vertical_edges],
        "selected_vertical_edges": [_round_segment(edge, body_top=body_top, body_bottom=body_bottom) for edge in selected_vertical_edges],
    }
