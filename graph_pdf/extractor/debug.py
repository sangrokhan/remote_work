from __future__ import annotations

from typing import List

from .shared import (
    _build_segment_groups,
    _char_rotation_degrees,
    _merge_horizontal_band_segments,
    _merge_numeric_positions,
    _merge_vertical_band_segments,
    _round_segment,
)
from .tables import _table_regions
from .text import _detect_body_bounds, _extract_body_text_lines, _extract_body_word_lines


def _collect_rotated_text_debug(page: "pdfplumber.page.Page", page_no: int) -> List[dict]:
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
    raw_text_lines, normalized_text_lines = _extract_body_text_lines(page, header_margin=header_margin, footer_margin=footer_margin)

    return {
        "page": page_no,
        "body_bounds": [round(body_top, 2), round(body_bottom, 2)],
        "table_count": len(tables),
        "tables": tables,
        "text_debug": {
            "raw_lines": raw_text_lines,
            "raw_line_boxes": [
                {
                    "text": str(line.get("text") or ""),
                    "x0": round(float(line.get("x0", 0.0)), 2),
                    "x1": round(float(line.get("x1", 0.0)), 2),
                    "top": round(float(line.get("top", 0.0)), 2),
                    "bottom": round(float(line.get("bottom", 0.0)), 2),
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
