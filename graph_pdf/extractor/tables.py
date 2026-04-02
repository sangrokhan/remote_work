from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Tuple

import pdfplumber

from .shared import (
    TableChunk,
    TableRows,
    _bboxes_intersect,
    _build_segment_groups,
    _merge_horizontal_band_segments,
    _merge_numeric_positions,
    _merge_vertical_band_segments,
    _normalize_text,
)
from .text import (
    _clean_cell_line,
    _detect_body_bounds,
    _filter_page_for_extraction,
    _extract_body_word_lines,
    _is_layout_artifact,
    _normalize_cell_lines,
    _repair_watermark_bleed,
)

_THIN_FILL_RECT_MAX_HEIGHT = 2.0
_SEPARATOR_RECT_MAX_THICKNESS = 0.5


def _merge_cells(table: Sequence[Sequence[str]]) -> List[List[str]]:
    # pdfplumber can yield `None` cells, so normalize early to simple stripped strings.
    return [[str(cell or "").strip() for cell in row] for row in table]


def _normalize_extracted_table(table: Sequence[Sequence[str]]) -> List[List[str]]:
    # Table normalization is deliberately cell-local so geometric table structure stays untouched.
    normalized: List[List[str]] = []
    for row in table:
        normalized_row = []
        for cell in row:
            cell_text = str(cell or "")
            explicit_lines = [_clean_cell_line(part) for part in cell_text.splitlines()]
            explicit_lines = [line for line in explicit_lines if line]
            if len(explicit_lines) > 1:
                normalized_row.append("\n".join(explicit_lines))
            else:
                normalized_row.append("\n".join(_normalize_cell_lines(cell_text)))
        normalized.append(normalized_row)
    return normalized


def _table_rejection_reason(table: Sequence[Sequence[str]]) -> str | None:
    # Rejection stays intentionally minimal to avoid throwing away sparse but valid tables.
    if not table:
        return "empty table"
    normalized_rows = [[str(cell or "").strip() for cell in row] for row in table]
    if not any(cell for cell in normalized_rows[0]):
        return "empty first row"
    return None


def _log_rejected_table(
    table: Sequence[Sequence[str]],
    crop_bbox: Tuple[float, float, float, float],
    reason: str,
) -> None:
    # Rejection logging is only for manual debugging; tests assert on accepted output instead.
    row_count = len(table)
    col_count = max((len(row) for row in table), default=0)
    bbox_text = ", ".join(f"{value:.2f}" for value in crop_bbox)
    print(f"[table-reject] bbox=({bbox_text}) rows={row_count} cols={col_count} reason={reason}")


def _looks_like_header_row(row: Sequence[str]) -> bool:
    # Header detection stays structure-based: multiple compact cells, no semantic keyword checks.
    if not row:
        return False
    tokens = [_normalize_text(c) for c in row if _normalize_text(c)]
    if len(tokens) < 2:
        return False
    if any("\n" in token for token in tokens):
        return False
    total_length = sum(len(token) for token in tokens)
    max_length = max(len(token) for token in tokens)
    average_length = total_length / len(tokens)
    return max_length <= 32 and average_length <= 18 and total_length <= max(64, len(tokens) * 18)


def _effective_non_empty_column_indices(
    rows: Sequence[Sequence[str]],
) -> list[int]:
    # Ignore empty cells introduced by renderer artifacts when deciding if a region is really one-column.
    column_indexes: set[int] = set()
    for row in rows:
        if not row:
            continue
        for idx, value in enumerate(row):
            if _normalize_text(value):
                column_indexes.add(idx)
    return sorted(column_indexes)


def _collapse_empty_columns(rows: Sequence[Sequence[str]]) -> List[List[str]]:
    kept_indices = _effective_non_empty_column_indices(rows)
    if not kept_indices:
        return [[] for _ in rows]
    return [
        [row[idx] if idx < len(row) else "" for idx in kept_indices]
        for row in rows
    ]


def _extract_region_words(
    page: pdfplumber.page.PageObject,
    bbox: Tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    # Region words are used to infer text flow independent of table cell extraction.
    filtered_page = _filter_page_for_extraction(page)
    x0, top, x1, bottom = bbox
    words = (
        filtered_page
        .crop((x0, top, x1, bottom))
        .extract_words(
            x_tolerance=1.5,
            y_tolerance=2.0,
            keep_blank_chars=False,
            extra_attrs=["size"],
        )
        or []
    )

    normalized: list[dict[str, Any]] = []
    for word in words:
        text = _repair_watermark_bleed(_normalize_text(str(word.get("text") or "")))
        if not text or _is_layout_artifact(text):
            continue
        normalized.append({
            "text": text,
            "x0": float(word.get("x0", 0.0)),
            "x1": float(word.get("x1", 0.0)),
            "top": float(word.get("top", 0.0)),
            "bottom": float(word.get("bottom", 0.0)),
        })
    return normalized


def _extract_region_lines(words: Sequence[dict[str, Any]], y_tolerance: float = 2.5) -> list[list[dict[str, Any]]]:
    lines: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: (float(item["top"]), float(item["x0"]))):
        if not lines or abs(float(word["top"]) - float(lines[-1][0]["top"])) > y_tolerance:
            lines.append([word])
            continue
        lines[-1].append(word)
    return lines


def _is_black_color(value: object, threshold: float = 0.18) -> bool:
    if value is None:
        return False

    if isinstance(value, (int, float)):
        return float(value) <= threshold

    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return all(float(component) <= threshold for component in value[:3])

    return False


def _is_black_line_segment(segment: dict[str, Any], color_threshold: float = 0.18) -> bool:
    color = segment.get("stroking_color")
    if color is None:
        color = segment.get("non_stroking_color")
    return _is_black_color(color, threshold=color_threshold)


def _is_black_fill_rect(rect: dict[str, Any], color_threshold: float = 0.18) -> bool:
    if not bool(rect.get("fill", False)):
        return False
    color = rect.get("non_stroking_color")
    if color is None:
        color = rect.get("stroking_color")
    return _is_black_color(color, threshold=color_threshold)


def _is_vertical_separator_rect(rect: dict[str, Any], thickness: float = _SEPARATOR_RECT_MAX_THICKNESS) -> bool:
    if not _is_black_fill_rect(rect):
        return False
    width = abs(float(rect.get("width", float(rect.get("x1", 0.0)) - float(rect.get("x0", 0.0))) or 0.0))
    height = abs(float(rect.get("height", float(rect.get("bottom", 0.0)) - float(rect.get("top", 0.0))) or 0.0))
    return width <= thickness and height > width


def _is_horizontal_separator_rect(rect: dict[str, Any], thickness: float = _SEPARATOR_RECT_MAX_THICKNESS) -> bool:
    if not _is_black_fill_rect(rect):
        return False
    width = abs(float(rect.get("width", float(rect.get("x1", 0.0)) - float(rect.get("x0", 0.0))) or 0.0))
    height = abs(float(rect.get("height", float(rect.get("bottom", 0.0)) - float(rect.get("top", 0.0))) or 0.0))
    return height <= thickness and width > height


def _extract_region_word_payloads(
    page: pdfplumber.page.PageObject,
    bbox: Tuple[float, float, float, float],
) -> list[dict[str, float | str]]:
    filtered_page = _filter_page_for_extraction(page)
    x0, top, x1, bottom = bbox
    words = (
        filtered_page
        .crop((x0, top, x1, bottom))
        .extract_words(
            x_tolerance=1.5,
            y_tolerance=2.0,
            keep_blank_chars=False,
            extra_attrs=["size"],
        )
        or []
    )

    payloads: list[dict[str, float | str]] = []
    for line_index, line_words in enumerate(_extract_region_lines(words)):
        ordered_words = sorted(line_words, key=lambda item: float(item.get("x0", 0.0)))
        for word in ordered_words:
            text = _normalize_text(str(word.get("text") or ""))
            if not text or _is_layout_artifact(text):
                continue
            payloads.append(
                {
                    "text": text,
                    "x0": float(word.get("x0", 0.0)),
                    "x1": float(word.get("x1", 0.0)),
                    "top": float(word.get("top", 0.0)),
                    "bottom": float(word.get("bottom", 0.0)),
                    "size": float(word.get("size", 0.0) or 0.0),
                    "line_index": line_index,
                }
            )
    return payloads


def _extract_black_lines_for_table(
    page: pdfplumber.page.PageObject,
    crop_bbox: Tuple[float, float, float, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    x0, y0, x1, y1 = crop_bbox
    horizontal_edges: list[dict[str, Any]] = []
    vertical_edges: list[dict[str, Any]] = []

    for line in getattr(page, "lines", []):
        if not _is_black_line_segment(line):
            continue
        line_x0 = float(line.get("x0", 0.0))
        line_x1 = float(line.get("x1", line_x0))
        line_top = float(line.get("top", 0.0))
        line_bottom = float(line.get("bottom", line_top))
        if line_bottom < y0 - 1.0 or line_top > y1 + 1.0:
            continue
        overlap = min(max(line_x0, line_x1), x1) - max(min(line_x0, line_x1), x0)
        if overlap < -1.0:
            continue
        orientation = line.get("orientation")
        if orientation == "h" or abs(line_bottom - line_top) <= abs(line_x1 - line_x0):
            horizontal_edges.append(dict(line))
        if orientation == "v" or abs(line_x1 - line_x0) < abs(line_bottom - line_top):
            vertical_edges.append(dict(line))

    for rect in getattr(page, "rects", []) or []:
        rect_x0 = float(rect.get("x0", 0.0))
        rect_x1 = float(rect.get("x1", rect_x0))
        rect_top = float(rect.get("top", 0.0))
        rect_bottom = float(rect.get("bottom", rect_top))
        if rect_bottom < y0 - 1.0 or rect_top > y1 + 1.0:
            continue
        overlap = min(max(rect_x0, rect_x1), x1) - max(min(rect_x0, rect_x1), x0)
        if overlap < -1.0:
            continue
        if _is_horizontal_separator_rect(rect):
            horizontal_edges.append(dict(rect))
        if _is_vertical_separator_rect(rect):
            vertical_edges.append(dict(rect))

    if not horizontal_edges:
        for edge in getattr(page, "horizontal_edges", []):
            if not _is_black_line_segment(edge):
                continue
            edge_top = float(edge.get("top", 0.0))
            edge_bottom = float(edge.get("bottom", edge_top))
            if edge_bottom < y0 - 1.0 or edge_top > y1 + 1.0:
                continue
            edge_x0 = float(edge.get("x0", 0.0))
            edge_x1 = float(edge.get("x1", edge_x0))
            overlap = min(max(edge_x0, edge_x1), x1) - max(min(edge_x0, edge_x1), x0)
            if overlap < -1.0:
                continue
            horizontal_edges.append(edge)

    if not vertical_edges:
        for edge in getattr(page, "vertical_edges", []):
            if not _is_black_line_segment(edge):
                continue
            edge_x0 = float(edge.get("x0", 0.0))
            edge_x1 = float(edge.get("x1", edge_x0))
            overlap = min(max(edge_x0, edge_x1), x1) - max(min(edge_x0, edge_x1), x0)
            if overlap < -1.0:
                continue
            edge_top = float(edge.get("top", 0.0))
            edge_bottom = float(edge.get("bottom", edge_top))
            if edge_bottom < y0 - 1.0 or edge_top > y1 + 1.0:
                continue
            vertical_edges.append(edge)

    merged_horizontal = []
    for group in _build_segment_groups(
        horizontal_edges,
        axis_key="top",
        merge_fn=_merge_horizontal_band_segments,
        tolerance=1.0,
    ):
        merged_horizontal.extend(group["merged_segments"])

    merged_vertical = []
    for group in _build_segment_groups(
        vertical_edges,
        axis_key="x0",
        merge_fn=_merge_vertical_band_segments,
        tolerance=1.0,
    ):
        merged_vertical.extend(group["merged_segments"])

    return merged_horizontal, merged_vertical


def _build_row_bands(
    crop_bbox: Tuple[float, float, float, float],
    horizontal_segments: list[dict[str, Any]],
) -> tuple[list[float], list[Tuple[float, float, float, float]], str | None]:
    x0, _y0, x1, _y1 = crop_bbox
    crop_width = max(0.0, x1 - x0)
    min_line_span = max(8.0, crop_width * 0.12)
    line_positions = sorted(
        (float(segment["top"]) + float(segment["bottom"])) / 2.0
        for segment in horizontal_segments
        if (float(segment["x1"]) - float(segment["x0"])) >= min_line_span
    )
    line_positions = _merge_numeric_positions(line_positions, tolerance=1.0)
    if len(line_positions) < 2:
        return [], [], "insufficient_row_lines"

    row_bands: list[tuple[float, float, float, float]] = []
    min_band_height = 2.0
    for top, bottom in zip(line_positions, line_positions[1:]):
        band_top = float(top)
        band_bottom = float(bottom)
        if band_bottom - band_top >= min_band_height:
            row_bands.append((x0, band_top, x1, band_bottom))

    if not row_bands:
        return line_positions, [], "invalid_row_bands"
    return line_positions, row_bands, None


def _build_column_bands(
    crop_bbox: Tuple[float, float, float, float],
    row_bands: list[Tuple[float, float, float, float]],
    vertical_segments: list[dict[str, Any]],
) -> tuple[list[float], list[Tuple[float, float, float, float]], str | None]:
    x0, y0, x1, y1 = crop_bbox
    if not row_bands:
        return [], [(x0, y0, x1, y1)], "no_internal_columns"

    edge_tolerance = 8.0
    candidate_segments: list[tuple[float, float, float]] = []
    for segment in vertical_segments:
        segment_x = float(segment["x0"]) if float(segment["x0"]) == float(segment["x1"]) else (float(segment["x0"]) + float(segment["x1"])) / 2.0
        if segment_x <= x0 + edge_tolerance or segment_x >= x1 - edge_tolerance:
            continue
        segment_top = float(segment["top"])
        segment_bottom = float(segment["bottom"])
        candidate_segments.append((segment_x, segment_top, segment_bottom))

    x_positions = [segment_x for segment_x, _segment_top, _segment_bottom in candidate_segments]
    x_positions = _merge_numeric_positions(sorted(x_positions), tolerance=8.0)
    if not x_positions:
        return [], [(x0, y0, x1, y1)], "no_internal_vertical_lines"

    if any(x0 < pos < x1 for pos in x_positions):
        x_boundaries = [x0, *x_positions, x1]
    else:
        x_boundaries = [x0, x1]

    x_boundaries = sorted(dict.fromkeys([float(v) for v in x_boundaries]))
    column_bands: list[tuple[float, float, float, float]] = []
    for left, right in zip(x_boundaries, x_boundaries[1:]):
        if right - left > 1.0:
            column_bands.append((left, y0, right, y1))

    if not column_bands:
        return x_positions, [(x0, y0, x1, y1)], "invalid_column_bands"

    return x_positions, column_bands, None


def _build_payload_grid(
    page: pdfplumber.page.PageObject,
    crop_bbox: Tuple[float, float, float, float],
    row_bands: list[tuple[float, float, float, float]],
    column_bands: list[tuple[float, float, float, float]],
) -> tuple[
    list[list[list[dict[str, Any]]]],
    int,
    int,
    int,
    int,
]:
    payloads = _extract_region_word_payloads(page, crop_bbox)
    if not payloads:
        return (
            [[[] for _ in column_bands] for _ in row_bands],
            0,
            0,
            0,
            0,
        )

    if not row_bands or not column_bands:
        return (
            [[[] for _ in column_bands] for _ in row_bands],
            0,
            0,
            0,
            len(payloads),
        )

    cell_payloads: list[list[list[dict[str, Any]]]] = [
        [[] for _ in column_bands] for _ in row_bands
    ]
    assigned = 0
    ambiguous = 0
    unassigned = 0

    for payload in payloads:
        payload_text = str(payload.get("text") or "")
        if not payload_text:
            continue

        payload_x0 = float(payload["x0"])
        payload_x1 = float(payload["x1"])
        payload_top = float(payload["top"])
        payload_bottom = float(payload["bottom"])

        row_index, row_score, row_ambiguous_count = _pick_band_for_payload(
            payload_top,
            payload_bottom,
            row_bands,
            axis="y",
        )
        column_index, col_score, col_ambiguous_count = _pick_band_for_payload(
            payload_x0,
            payload_x1,
            column_bands,
            axis="x",
        )

        if (
            row_index is None
            or column_index is None
            or row_score <= 0.0
            or col_score <= 0.0
        ):
            unassigned += 1
            continue

        if row_ambiguous_count > 1 or col_ambiguous_count > 1:
            ambiguous += 1

        cell_payloads[row_index][column_index].append(
            {
                **payload,
                "line_index": payload.get("line_index"),
                "normalized_x0": payload_x0,
            }
        )
        assigned += 1

    return cell_payloads, assigned, ambiguous, unassigned, len(payloads)


def _rows_from_payload_grid(
    cell_payloads: list[list[list[dict[str, Any]]]],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in cell_payloads:
        row_values: list[str] = []
        for cell_payloads in row:
            if not cell_payloads:
                row_values.append("")
                continue

            sorted_cell_payloads = sorted(
                cell_payloads,
                key=lambda payload: (
                    int(payload.get("line_index", -1)) if payload.get("line_index") is not None else -1,
                    float(payload.get("top", 0.0)),
                    float(payload.get("x0", 0.0)),
                )
            )

            grouped_lines: list[dict[str, Any]] = []
            previous_payload: dict[str, Any] | None = None
            for payload in sorted_cell_payloads:
                payload_text = str(payload.get("text", "")).strip()
                if not payload_text:
                    continue

                current_line_index = payload.get("line_index")
                previous_line_index = previous_payload.get("line_index") if previous_payload else None

                same_line = False
                if previous_payload is not None:
                    if current_line_index is not None and previous_line_index is not None:
                        same_line = current_line_index == previous_line_index
                    else:
                        same_line = abs(float(payload.get("top", 0.0)) - float(previous_payload.get("top", 0.0))) <= 2.5

                if not same_line or not grouped_lines:
                    grouped_lines.append(
                        {
                            "words": [payload_text],
                            "top": float(payload.get("top", 0.0)),
                            "bottom": float(payload.get("bottom", payload.get("top", 0.0))),
                            "font_size": float(payload.get("size", 0.0) or 0.0),
                        }
                    )
                else:
                    grouped_lines[-1]["words"].append(payload_text)
                    grouped_lines[-1]["bottom"] = max(
                        float(grouped_lines[-1]["bottom"]),
                        float(payload.get("bottom", payload.get("top", 0.0))),
                    )
                    grouped_lines[-1]["font_size"] = max(
                        float(grouped_lines[-1]["font_size"]),
                        float(payload.get("size", 0.0) or 0.0),
                    )

                previous_payload = payload

            logical_lines: list[str] = []
            current_line: str | None = None
            current_bottom = 0.0
            current_font_size = 0.0
            for grouped_line in grouped_lines:
                line_text = " ".join(grouped_line["words"]).strip()
                if not line_text:
                    continue
                line_top = float(grouped_line["top"])
                line_bottom = float(grouped_line["bottom"])
                line_font_size = float(grouped_line["font_size"] or 0.0)
                if current_line is None:
                    current_line = line_text
                    current_bottom = line_bottom
                    current_font_size = line_font_size
                    continue

                gap = max(0.0, line_top - current_bottom)
                reference_font_size = line_font_size or current_font_size
                gap_ratio = (gap / reference_font_size) if reference_font_size > 0.0 else float("inf")
                if current_line.endswith("-"):
                    current_line = f"{current_line}{line_text}".strip()
                elif gap_ratio < 0.2:
                    current_line = f"{current_line} {line_text}".strip()
                else:
                    logical_lines.append(current_line)
                    current_line = line_text
                current_bottom = line_bottom
                current_font_size = line_font_size or current_font_size

            if current_line:
                logical_lines.append(current_line)

            row_values.append("\n".join(logical_lines))

        rows.append(row_values)

    return rows


def _interval_overlap(start: float, end: float, lower: float, upper: float) -> float:
    return max(0.0, min(end, upper) - max(start, lower))


def _pick_band_for_payload(
    start: float,
    end: float,
    bands: list[Tuple[float, float, float, float]],
    *,
    axis: str,
) -> tuple[int | None, float, int]:
    center = (start + end) / 2.0
    best_score = -1.0
    best_index: int | None = None
    for index, band in enumerate(bands):
        if axis == "y":
            band_start, band_end = band[1], band[3]
        else:
            band_start, band_end = band[0], band[2]
        score = _interval_overlap(start, end, band_start, band_end)
        if score > best_score + 1e-9:
            best_score = score
            best_index = index
            continue
        if abs(score - best_score) <= 1e-9:
            continue

    if best_index is None:
        best_distance = float("inf")
        for index, band in enumerate(bands):
            if axis == "y":
                band_start = band[1]
                band_end = band[3]
            else:
                band_start = band[0]
                band_end = band[2]
            band_center = (band_start + band_end) / 2.0
            distance = abs(center - band_center)
            if distance < best_distance:
                best_distance = distance
                best_index = index

    if axis == "y":
        ambiguity_count = 0
        for index, band in enumerate(bands):
            band_start, band_end = band[1], band[3]
            if _interval_overlap(start, end, band_start, band_end) == best_score and best_score > 0.0:
                ambiguity_count += 1
        return best_index, best_score, max(1, ambiguity_count)

    ambiguity_count = 0
    for index, band in enumerate(bands):
        band_start, band_end = band[0], band[2]
        if _interval_overlap(start, end, band_start, band_end) == best_score and best_score > 0.0:
            ambiguity_count += 1
    return best_index, best_score, max(1, ambiguity_count)


def _build_grid_rows_from_black_lines(
    page: pdfplumber.page.PageObject,
    crop_bbox: Tuple[float, float, float, float],
) -> tuple[
    list[list[str]],
    list[float],
    list[float],
    list[tuple[float, float, float, float]],
    list[tuple[float, float, float, float]],
    dict[str, Any],
]:
    horizontal_segments, vertical_segments = _extract_black_lines_for_table(page, crop_bbox)
    row_line_positions, row_bands, row_error = _build_row_bands(crop_bbox, horizontal_segments)
    if row_error is not None:
        return [], row_line_positions, [], [], [], {
            "reason": row_error,
            "stage": "row_bands",
            "raw_row_lines": len(horizontal_segments),
        }

    column_line_positions, column_bands, column_error = _build_column_bands(crop_bbox, row_bands, vertical_segments)
    if column_error is not None and column_error != "no_internal_vertical_lines":
        return [], row_line_positions, column_line_positions, row_bands, column_bands, {
            "reason": column_error,
            "stage": "column_bands",
            "raw_vertical_lines": len(vertical_segments),
        }

    cell_payloads, assigned_payload_count, ambiguous_payload_count, unassigned_payload_count, raw_payload_count = _build_payload_grid(
        page,
        crop_bbox,
        row_bands,
        column_bands,
    )
    if not raw_payload_count:
        return [], row_line_positions, column_line_positions, row_bands, column_bands, {
            "reason": "no_text_units",
            "stage": "line_assignment",
            "raw_row_lines": len(horizontal_segments),
            "raw_vertical_lines": len(vertical_segments),
            "raw_payload_count": 0,
        }

    if not any(any(cell) for cell in cell_payloads):
        return [], row_line_positions, column_line_positions, row_bands, column_bands, {
            "reason": "no_text_assigned_to_cells",
            "stage": "line_assignment",
            "raw_row_lines": len(horizontal_segments),
            "raw_vertical_lines": len(vertical_segments),
            "raw_payload_count": raw_payload_count,
            "assigned_payload_count": assigned_payload_count,
            "ambiguous_payload_count": ambiguous_payload_count,
            "unassigned_payload_count": unassigned_payload_count,
        }

    rows: list[list[str]] = _rows_from_payload_grid(cell_payloads)

    return (
        rows,
        row_line_positions,
        column_line_positions,
        row_bands,
        column_bands,
        {
            "stage": "line_assignment",
            "raw_row_lines": len(horizontal_segments),
            "raw_vertical_lines": len(vertical_segments),
            "raw_payload_count": raw_payload_count,
            "assigned_payload_count": assigned_payload_count,
            "ambiguous_payload_count": ambiguous_payload_count,
            "unassigned_payload_count": unassigned_payload_count,
            "row_count": len(rows),
            "column_count": len(column_bands),
            "column_error": column_error,
            "candidate_rows": 0,
            "merged_rows": 0,
            "ignored_rows": 0,
        },
    )


def _extract_region_line_rows(
    page: pdfplumber.page.PageObject,
    bbox: Tuple[float, float, float, float],
) -> list[list[str]]:
    lines = _extract_region_line_payloads(page, bbox)
    rows: list[list[str]] = []
    for line in lines:
        text = _normalize_text(str(line.get("text") or ""))
        if text:
            rows.append([text])
    return rows


def _compact_fallback_rows(rows: list[list[str]]) -> list[list[str]]:
    compacted: list[list[str]] = []
    for row in rows:
        normalized = " ".join(_normalize_text(cell) for cell in row if _normalize_text(cell)).strip()
        if not normalized:
            continue
        if compacted and compacted[-1] == [normalized]:
            continue
        compacted.append([normalized])
    return compacted


def _line_text_from_words(words_in_line: Sequence[dict[str, Any]]) -> str:
    return " ".join(str(word.get("text") or "").strip() for word in sorted(words_in_line, key=lambda item: float(item["x0"]))).strip()


def _extract_region_line_payloads(
    page: pdfplumber.page.PageObject,
    bbox: Tuple[float, float, float, float],
) -> list[dict[str, float | str]]:
    lines = _extract_region_lines(_extract_region_words(page, bbox))
    payloads: list[dict[str, float | str]] = []
    for words_in_line in lines:
        text = _line_text_from_words(words_in_line)
        if not text:
            continue
        payloads.append(
            {
                "text": text,
                "x0": min(float(word["x0"]) for word in words_in_line),
                "x1": max(float(word["x1"]) for word in words_in_line),
                "top": min(float(word["top"]) for word in words_in_line),
                "bottom": max(float(word["bottom"]) for word in words_in_line),
            }
        )
    return payloads


def _first_non_empty_cell_value(row: Sequence[str]) -> str:
    for value in row:
        normalized = _normalize_text(value)
        if normalized:
            return normalized
    return ""


def _rect_fill_color_key(rect: dict[str, Any]) -> tuple[Any, ...] | int | float | None:
    color = rect.get("non_stroking_color")
    if color is None:
        color = rect.get("stroking_color")
    if color is None:
        return None
    if isinstance(color, (int, float)):
        return round(float(color), 3)
    if isinstance(color, (list, tuple)):
        return tuple(round(float(value), 3) for value in color[:3])
    return None


def _normalize_color_match(left: object, right: object, tolerance: float = 0.02) -> bool:
    if left is None and right is None:
        return True
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= tolerance
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            return False
        return all(abs(float(l) - float(r)) <= tolerance for l, r in zip(left, right))
    return False


def _internal_grid_counts(
    page: pdfplumber.page.PageObject,
    bbox: Tuple[float, float, float, float],
) -> tuple[int, int]:
    # Internal edges are a table-structure signal compared with prose boxes.
    x0, y0, x1, y1 = bbox
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    if width <= 0.0 or height <= 0.0:
        return (0, 0)

    internal_vertical = 0
    for edge in getattr(page, "vertical_edges", []):
        edge_x0 = float(edge.get("x0", 0.0))
        edge_top = float(edge.get("top", 0.0))
        edge_bottom = float(edge.get("bottom", 0.0))
        if edge_top > y1 or edge_bottom < y0:
            continue
        if edge_x0 <= x0 + 2.0 or edge_x0 >= x1 - 2.0:
            continue
        if edge_bottom - edge_top >= height * 0.35:
            internal_vertical += 1

    internal_horizontal = 0
    for edge in getattr(page, "horizontal_edges", []):
        edge_top = float(edge.get("top", 0.0))
        edge_x0 = float(edge.get("x0", 0.0))
        edge_x1 = float(edge.get("x1", 0.0))
        if edge_top <= y0 + 2.0 or edge_top >= y1 - 2.0:
            continue
        edge_length = edge_x1 - edge_x0
        if edge_length >= width * 0.35 and edge_x0 < x1 and edge_x1 > x0:
            internal_horizontal += 1

    return internal_vertical, internal_horizontal


def _line_color_key(line: dict[str, Any]) -> tuple[Any, ...] | int | float | None:
    # Line colors from PDF objects are used to detect note envelopes.
    color = line.get("stroking_color")
    if color is None:
        color = line.get("non_stroking_color")
    if color is None:
        return None
    if isinstance(color, (int, float)):
        return round(float(color), 3)
    if isinstance(color, (list, tuple)):
        return tuple(round(float(value), 3) for value in color[:3])
    return None


def _note_border_signature(
    page: pdfplumber.page.PageObject,
    bbox: Tuple[float, float, float, float],
) -> tuple[bool, dict[str, Any]]:
    # Note-like candidates are often bounded by top/bottom horizontal lines
    # that share color and mostly overlap each other in X coordinates.
    x0, y0, x1, y1 = bbox
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    if width <= 0.0 or height <= 0.0:
        return False, {"reason": "invalid_bbox"}

    overlap_threshold = max(1, width * 0.55)
    top_band = y0 + height * 0.22
    bottom_band = y1 - height * 0.22
    min_edge_length = max(8.0, width * 0.45)
    start_tolerance = max(2.0, width * 0.02)
    inner_fill_tol = max(1.0, width * 0.02)

    def is_blue_color(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return False
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            r = float(value[0])
            g = float(value[1])
            b = float(value[2])
            return b > 0.45 and r < 0.35 and g < 0.35
        return False

    top_candidates = []
    bottom_candidates = []
    interior_lines = 0
    for edge in getattr(page, "horizontal_edges", []):
        edge_top = float(edge.get("top", 0.0))
        edge_bottom = float(edge.get("bottom", edge_top))
        if edge_bottom < y0 - 2.0 or edge_top > y1 + 2.0:
            continue

        edge_x0 = float(edge.get("x0", 0.0))
        edge_x1 = float(edge.get("x1", edge_x0))
        overlap = min(edge_x1, x1) - max(edge_x0, x0)
        if overlap < overlap_threshold:
            continue
        if edge_x1 - edge_x0 < min_edge_length:
            continue
        if edge_top < top_band:
            top_candidates.append(edge)
        elif edge_top > bottom_band:
            bottom_candidates.append(edge)
        else:
            interior_lines += 1

    # Some PDFs emit top/bottom separators as very thin filled rects rather than line edges.
    rects = getattr(page, "rects", []) or []
    for rect in rects:
        if not bool(rect.get("fill")):
            continue
        if bool(rect.get("stroke")):
            continue
        rect_top = float(rect.get("top", 0.0))
        rect_bottom = float(rect.get("bottom", rect_top))
        rect_height = rect_bottom - rect_top
        if abs(rect_height) > _THIN_FILL_RECT_MAX_HEIGHT:
            continue

        edge_top = float(rect_top)
        edge_bottom = float(rect_bottom)
        if edge_bottom < y0 - 2.0 or edge_top > y1 + 2.0:
            continue

        edge_x0 = float(rect.get("x0", 0.0))
        edge_x1 = float(rect.get("x1", edge_x0))
        overlap = min(edge_x1, x1) - max(edge_x0, x0)
        if overlap < overlap_threshold:
            continue
        if edge_x1 - edge_x0 < min_edge_length:
            continue
        # Keep blue-only border segments as note candidates.
        color = rect.get("non_stroking_color")
        if color is None:
            color = rect.get("stroking_color")
        if not is_blue_color(color):
            continue
        if edge_top < top_band:
            top_candidates.append(rect)
        elif edge_top > bottom_band:
            bottom_candidates.append(rect)
        else:
            interior_lines += 1

    top_meta = [{"color": _line_color_key(edge)} for edge in top_candidates]
    bottom_meta = [{"color": _line_color_key(edge)} for edge in bottom_candidates]

    if not top_candidates or not bottom_candidates:
        return False, {
            "reason": "missing_border_lines",
            "top_candidate_count": len(top_candidates),
            "bottom_candidate_count": len(bottom_candidates),
            "interior_line_count": interior_lines,
            "top_colors": top_meta,
            "bottom_colors": bottom_meta,
        }

    for top_edge in top_candidates:
        top_color = _line_color_key(top_edge)
        top_x0 = float(top_edge.get("x0", 0.0))
        top_x1 = float(top_edge.get("x1", top_x0))
        top_y0 = float(top_edge.get("top", 0.0))
        top_y1 = float(top_edge.get("bottom", top_y0))
        top_w = abs(top_x1 - top_x0)
        for bottom_edge in bottom_candidates:
            bottom_color = _line_color_key(bottom_edge)
            if not _normalize_color_match(top_color, bottom_color):
                continue
            bottom_x0 = float(bottom_edge.get("x0", 0.0))
            bottom_x1 = float(bottom_edge.get("x1", bottom_x0))
            bottom_y0 = float(bottom_edge.get("top", 0.0))
            bottom_y1 = float(bottom_edge.get("bottom", bottom_y0))
            if abs(top_x0 - bottom_x0) > start_tolerance:
                continue
            if abs(top_x1 - bottom_x1) > start_tolerance:
                continue

            # Additional note signal:
            # There is a filled rect spanning the same width between the two horizontal blue lines.
            inner_full_width_fill = False
            gap_top = min(top_y1, top_y0, bottom_y0, bottom_y1)
            gap_bottom = max(top_y1, top_y0, bottom_y0, bottom_y1)
            for rect in rects:
                if not bool(rect.get("fill")):
                    continue
                if bool(rect.get("stroke")):
                    continue
                interior_left = float(rect.get("x0", 0.0))
                interior_right = float(rect.get("x1", interior_left))
                interior_top = float(rect.get("top", 0.0))
                interior_bottom = float(rect.get("bottom", interior_top))
                interior_w = interior_right - interior_left
                if interior_bottom <= gap_top or interior_top >= gap_bottom:
                    continue
                if interior_top <= gap_top + 0.05 or interior_bottom >= gap_bottom - 0.05:
                    continue
                if abs(interior_w - top_w) > inner_fill_tol:
                    continue
                x_overlap = min(interior_right, max(top_x1, bottom_x1)) - max(interior_left, min(top_x0, bottom_x0))
                if x_overlap < 0.0:
                    continue
                if x_overlap < width * 0.88:
                    continue
                if abs((interior_right - interior_left) - (top_x1 - top_x0)) <= inner_fill_tol:
                    inner_full_width_fill = True
                    break

            return True, {
                "reason": "matched_border_lines",
                "color": top_color,
                "top_line_y": float(top_edge.get("top", 0.0)),
                "bottom_line_y": float(bottom_edge.get("top", 0.0)),
                "top_line_x0": float(top_edge.get("x0", 0.0)),
                "top_line_x1": float(top_edge.get("x1", 0.0)),
                "bottom_line_x0": float(bottom_edge.get("x0", 0.0)),
                "bottom_line_x1": float(bottom_edge.get("x1", 0.0)),
                "inner_full_width_fill": inner_full_width_fill,
                "inner_full_width_fill_count": int(inner_full_width_fill),
            }

    return False, {
        "reason": "unmatched_border_colors",
        "top_candidate_count": len(top_candidates),
        "bottom_candidate_count": len(bottom_candidates),
        "interior_line_count": interior_lines,
        "top_colors": top_meta,
        "bottom_colors": bottom_meta,
    }


def _content_width_ratio(
    region_words: Sequence[dict[str, Any]],
    fallback_width: float | None = None,
    bbox: Tuple[float, float, float, float] | None = None,
) -> float:
    if not region_words:
        if fallback_width is not None and fallback_width > 0.0:
            return 1.0
        return 0.0

    min_x0 = min(float(word.get("x0", 0.0)) for word in region_words)
    max_x1 = max(float(word.get("x1", 0.0)) for word in region_words)
    text_width = max(0.0, max_x1 - min_x0)
    if text_width <= 0.0:
        return 0.0

    if bbox is None:
        if fallback_width is None or fallback_width <= 0.0:
            return 1.0
        return text_width / fallback_width

    region_width = max(0.0, float(bbox[2]) - float(bbox[0]))
    if region_width <= 0.0:
        return 1.0
    return text_width / region_width


def _normalized_row_text(row: Sequence[str]) -> str:
    return "".join(_normalize_text(cell) for cell in row if _normalize_text(cell))


def _rows_match(a: Sequence[str], b: Sequence[str]) -> bool:
    # Header rows are compared after whitespace normalization to avoid duplicate header output.
    if len(a) == len(b) and all(_normalize_text(x) == _normalize_text(y) for x, y in zip(a, b)):
        return True
    return bool(_normalized_row_text(a)) and _normalized_row_text(a) == _normalized_row_text(b)


def _header_row_count(rows: Sequence[Sequence[str]], max_header_rows: int = 2) -> int:
    # Treat consecutive compact top rows as a multi-row header block without semantic keyword checks.
    count = 0
    for row in rows[:max_header_rows]:
        if not _looks_like_header_row(row):
            break
        if count >= 1 and len(rows) > count + 1:
            normalized_cells = [_normalize_text(cell) for cell in row if _normalize_text(cell)]
            next_cells = [_normalize_text(cell) for cell in rows[count + 1] if _normalize_text(cell)]
            current_total = sum(len(cell) for cell in normalized_cells)
            next_total = sum(len(cell) for cell in next_cells)
            if next_total > 0 and current_total >= next_total * 0.9:
                break
        count += 1
    return count


def _is_single_cell_fragment_row(row: Sequence[str]) -> tuple[bool, int, str]:
    # True only when the row is effectively one cell and that cell is header-like.
    non_empty = [(idx, _normalize_text(cell)) for idx, cell in enumerate(row) if _normalize_text(cell)]
    if len(non_empty) != 1:
        return False, -1, ""
    idx, text = non_empty[0]
    if not _looks_like_header_row([text]):
        return False, -1, ""
    return True, idx, text


def _can_merge_header_split_rows(
    previous_row: Sequence[str],
    current_row: Sequence[str],
    merged_row_count: int,
) -> tuple[bool, int, str]:
    # Merge only short header fragments in the leading rows of a table.
    if merged_row_count > 4:
        return False, -1, ""

    prev_is_single, prev_idx, prev_text = _is_single_cell_fragment_row(previous_row)
    curr_is_single, curr_idx, curr_text = _is_single_cell_fragment_row(current_row)
    if not (prev_is_single and curr_is_single):
        return False, -1, ""
    if prev_idx != curr_idx:
        return False, -1, ""

    joiner = " "
    # Preserve broken words like "Displa" + "y Name" without inserting extra space.
    if prev_text[-1].isalnum() and curr_text[0].islower():
        joiner = ""
    return True, prev_idx, joiner


def _collapse_header_rows(header_rows: Sequence[Sequence[str]]) -> List[str]:
    # Markdown can only express one header row directly, so fold multi-row headers into one logical row.
    if not header_rows:
        return []
    col_count = max((len(row) for row in header_rows), default=0)
    collapsed: List[str] = []
    for col_idx in range(col_count):
        parts = [
            str(row[col_idx]).strip()
            for row in header_rows
            if col_idx < len(row) and str(row[col_idx]).strip()
        ]
        collapsed.append("\n".join(parts))
    return collapsed


def _split_repeated_header(prev_rows: TableRows, curr_rows: TableRows) -> TableRows:
    # When a page repeats the same header row, only keep the first occurrence in the merged output.
    prev_header_count = _header_row_count(prev_rows)
    curr_header_count = _header_row_count(curr_rows)
    if not prev_header_count or prev_header_count != curr_header_count:
        comparable_count = min(prev_header_count, len(curr_rows))
        if comparable_count <= 0:
            return curr_rows

        trimmed_prefix: TableRows = []
        matched_rows = 0
        for idx in range(comparable_count):
            previous_row = list(prev_rows[idx])
            current_row = list(curr_rows[idx])
            if _rows_match(previous_row, current_row):
                matched_rows += 1
                continue
            width = max(len(previous_row), len(current_row))
            trimmed_row = [""] * width
            row_has_tail = False
            row_matches = True

            for col_idx in range(width):
                previous_cell = str(previous_row[col_idx]).strip() if col_idx < len(previous_row) else ""
                current_cell = str(current_row[col_idx]).strip() if col_idx < len(current_row) else ""
                previous_text = _normalize_text(previous_cell)
                current_text = _normalize_text(current_cell)

                if not previous_text and not current_text:
                    continue
                if previous_text and current_text == previous_text:
                    continue
                if previous_cell and current_cell.startswith(f"{previous_cell}\n"):
                    tail = "\n".join(
                        part.strip()
                        for part in current_cell.split("\n")[1:]
                        if part.strip()
                    )
                    if tail:
                        trimmed_row[col_idx] = tail
                        row_has_tail = True
                        continue
                row_matches = False
                break

            if not row_matches:
                break

            matched_rows += 1
            if row_has_tail:
                trimmed_prefix.append(trimmed_row)

        if not matched_rows:
            return curr_rows

        return trimmed_prefix + [list(row) for row in curr_rows[matched_rows:]]
    if all(
        _rows_match(prev_rows[idx], curr_rows[idx])
        for idx in range(curr_header_count)
    ):
        return curr_rows[curr_header_count:]
    return curr_rows


def _is_continuation_chunk(prev_rows: TableRows, curr_rows: TableRows) -> bool:
    # Continuation chunks usually keep the schema but leave the first column blank while the row carries on.
    if not prev_rows or not curr_rows:
        return False
    if len(prev_rows[0]) != len(curr_rows[0]):
        return False
    first = curr_rows[0]
    if not first or _looks_like_header_row(first) or _normalize_text(first[0]):
        return False
    return any(_normalize_text(cell) for cell in first[1:])


def _should_try_table_continuation_merge(
    pending_page: int | None,
    current_page: int,
) -> bool:
    # Cross-page merging is limited to immediately adjacent pages.
    return pending_page is not None and current_page == pending_page + 1


def _body_text_boxes(
    page: pdfplumber.page.PageObject,
    header_margin: float,
    footer_margin: float,
    excluded_bboxes: Sequence[Tuple[float, float, float, float]] = (),
) -> List[Tuple[float, float, float, float]]:
    # These boxes are used only to detect prose sitting between two candidate table fragments.
    filtered_page = _filter_page_for_extraction(page)
    body_top, body_bottom = _detect_body_bounds(page, header_margin=header_margin, footer_margin=footer_margin)
    text_boxes: List[Tuple[float, float, float, float]] = []
    for word in filtered_page.extract_words() or []:
        text = _normalize_text(str(word.get("text") or ""))
        if not text or _is_layout_artifact(text):
            continue
        bbox = (
            float(word.get("x0", 0.0)),
            float(word.get("top", 0.0)),
            float(word.get("x1", 0.0)),
            float(word.get("bottom", 0.0)),
        )
        if bbox[3] <= body_top or bbox[1] >= body_bottom:
            continue
        if any(_bboxes_intersect(bbox, excluded_bbox) for excluded_bbox in excluded_bboxes):
            continue
        text_boxes.append(bbox)
    return text_boxes


def _is_overlap_in_x(
    subject: Tuple[float, float, float, float],
    reference: Tuple[float, float, float, float],
    *,
    min_overlap_ratio: float = 0.0,
    min_overlap_width: float = 0.0,
) -> bool:
    if min_overlap_ratio <= 0.0 and min_overlap_width <= 0.0:
        return True

    overlap = min(float(subject[2]), float(reference[2])) - max(float(subject[0]), float(reference[0]))
    if overlap <= 0.0:
        return False
    if overlap >= min_overlap_width and min_overlap_width > 0.0:
        return True
    if min_overlap_ratio <= 0.0:
        return overlap > 0.0

    subject_width = max(1.0, float(subject[2]) - float(subject[0]))
    reference_width = max(1.0, float(reference[2]) - float(reference[0]))
    overlap_ratio = overlap / min(subject_width, reference_width)
    return overlap_ratio >= min_overlap_ratio


def _gap_text_boxes_after_bbox(
    body_text_boxes: Sequence[Tuple[float, float, float, float]],
    bbox: Tuple[float, float, float, float],
    max_gap: float | None = None,
    overlap_bbox: Tuple[float, float, float, float] | None = None,
    min_x_overlap_ratio: float = 0.0,
    min_x_overlap_width: float = 0.0,
) -> List[Tuple[float, float, float, float]]:
    # Text after a table candidate can block continuation into the next page.
    bottom = float(bbox[3])
    if max_gap is None:
        return [
            text_bbox
            for text_bbox in body_text_boxes
            if float(text_bbox[1]) > bottom + 1.0
            and _is_overlap_in_x(
                subject=text_bbox,
                reference=overlap_bbox if overlap_bbox is not None else bbox,
                min_overlap_ratio=min_x_overlap_ratio,
                min_overlap_width=min_x_overlap_width,
            )
        ]
    max_top = bottom + max(0.0, float(max_gap))
    return [
        text_bbox
        for text_bbox in body_text_boxes
        if bottom + 1.0 < float(text_bbox[1]) <= max_top
        and _is_overlap_in_x(
            subject=text_bbox,
            reference=overlap_bbox if overlap_bbox is not None else bbox,
            min_overlap_ratio=min_x_overlap_ratio,
            min_x_overlap_width=min_x_overlap_width,
        )
    ]


def _gap_text_boxes_before_bbox(
    body_text_boxes: Sequence[Tuple[float, float, float, float]],
    bbox: Tuple[float, float, float, float],
    max_gap: float | None = None,
    overlap_bbox: Tuple[float, float, float, float] | None = None,
    min_x_overlap_ratio: float = 0.0,
    min_x_overlap_width: float = 0.0,
) -> List[Tuple[float, float, float, float]]:
    # Text before a table candidate can mean the new page starts with prose rather than a continuation table.
    top = float(bbox[1])
    if max_gap is None:
        return [
            text_bbox
            for text_bbox in body_text_boxes
            if float(text_bbox[3]) < top - 1.0
            and _is_overlap_in_x(
                subject=text_bbox,
                reference=overlap_bbox if overlap_bbox is not None else bbox,
                min_overlap_ratio=min_x_overlap_ratio,
                min_x_overlap_width=min_x_overlap_width,
            )
        ]
    min_bottom = top - max(0.0, float(max_gap))
    return [
        text_bbox
        for text_bbox in body_text_boxes
        if min_bottom <= float(text_bbox[3]) < top - 1.0
        and _is_overlap_in_x(
            subject=text_bbox,
            reference=overlap_bbox if overlap_bbox is not None else bbox,
            min_overlap_ratio=min_x_overlap_ratio,
            min_x_overlap_width=min_x_overlap_width,
        )
    ]


def _has_gap_text_after_bbox(
    body_text_boxes: Sequence[Tuple[float, float, float, float]],
    bbox: Tuple[float, float, float, float],
    max_gap: float | None = None,
    overlap_bbox: Tuple[float, float, float, float] | None = None,
    min_x_overlap_ratio: float = 0.0,
    min_x_overlap_width: float = 0.0,
) -> bool:
    # Fast predicate version used by cross-page merge checks.
    threshold = float(bbox[3])
    overlap_reference = overlap_bbox if overlap_bbox is not None else bbox
    if max_gap is None:
        for text_bbox in body_text_boxes:
            if float(text_bbox[1]) > threshold + 1.0 and _is_overlap_in_x(
                subject=text_bbox,
                reference=overlap_reference,
                min_overlap_ratio=min_x_overlap_ratio,
                min_overlap_width=min_x_overlap_width,
            ):
                return True
    else:
        max_top = threshold + max(0.0, float(max_gap))
        for text_bbox in body_text_boxes:
            top = float(text_bbox[1])
            if threshold + 1.0 < top <= max_top and _is_overlap_in_x(
                subject=text_bbox,
                reference=overlap_reference,
                min_overlap_ratio=min_x_overlap_ratio,
                min_overlap_width=min_x_overlap_width,
            ):
                return True
    return False


def _has_gap_text_before_bbox(
    body_text_boxes: Sequence[Tuple[float, float, float, float]],
    bbox: Tuple[float, float, float, float],
    max_gap: float | None = None,
    overlap_bbox: Tuple[float, float, float, float] | None = None,
    min_x_overlap_ratio: float = 0.0,
    min_x_overlap_width: float = 0.0,
) -> bool:
    # Fast predicate version used by cross-page merge checks.
    threshold = float(bbox[1])
    overlap_reference = overlap_bbox if overlap_bbox is not None else bbox
    if max_gap is None:
        for text_bbox in body_text_boxes:
            if float(text_bbox[3]) < threshold - 1.0 and _is_overlap_in_x(
                subject=text_bbox,
                reference=overlap_reference,
                min_overlap_ratio=min_x_overlap_ratio,
                min_overlap_width=min_x_overlap_width,
            ):
                return True
    else:
        min_top = threshold - max(0.0, float(max_gap))
        for text_bbox in body_text_boxes:
            bottom = float(text_bbox[3])
            if min_top <= bottom < threshold - 1.0 and _is_overlap_in_x(
                subject=text_bbox,
                reference=overlap_reference,
                min_overlap_ratio=min_x_overlap_ratio,
                min_overlap_width=min_x_overlap_width,
            ):
                return True
    return False


def _vertical_axes_for_bbox(
    page: pdfplumber.page.PageObject,
    bbox: Tuple[float, float, float, float],
) -> List[float]:
    # Shared vertical axes are one of the strongest geometric signals that two fragments belong to the same table.
    x0, y0, x1, y1 = bbox
    axes = [
        float(edge["x0"])
        for edge in page.vertical_edges
        if float(edge["x0"]) >= x0 - 2.0
        and float(edge["x0"]) <= x1 + 2.0
        and float(edge["bottom"]) >= y0
        and float(edge["top"]) <= y1
    ]
    return _merge_numeric_positions(axes, tolerance=1.0)


def _continuation_regions_should_merge(
    prev_bbox: Tuple[float, float, float, float],
    curr_bbox: Tuple[float, float, float, float],
    prev_axes: Sequence[float],
    curr_axes: Sequence[float],
    body_top: float,
    body_bottom: float,
    has_gap_text: bool | Sequence[Tuple[float, float, float, float]],
    edge_tolerance: float = 24.0,
    axis_tolerance: float = 1.0,
    prev_page_height: float | None = None,
) -> bool:
    # Merge only when geometry matches and no body text sits between the two fragments.
    _prev_x0, _prev_top, _prev_x1, prev_bottom = prev_bbox
    _curr_x0, curr_top, _curr_x1, _curr_bottom = curr_bbox

    shared_axes = [axis for axis in prev_axes if any(abs(axis - other) <= axis_tolerance for other in curr_axes)]
    prev_near_footer = abs(body_bottom - prev_bottom) <= edge_tolerance
    curr_near_header = abs(curr_top - body_top) <= edge_tolerance
    if bool(has_gap_text) and not (prev_near_footer or curr_near_header):
        return False

    # Near-footer / near-header placement is the common continuation pattern.
    # Allow that path even when shared vertical axes are not fully stable.
    if prev_near_footer or curr_near_header:
        return True

    if not shared_axes:
        return False

    if prev_page_height is None:
        return True

    # Cross-page continuation usually occurs near the top of the next page and near the bottom
    # of the previous page. A large geometric jump is likely a new table block, not a split.
    gap_across_pages = (float(prev_page_height) - prev_bottom) + (curr_top - body_top)
    return gap_across_pages <= 220.0


def _format_markdown_cell(value: str) -> str:
    # Markdown cells preserve logical line breaks with `<br>` while escaping literal pipe characters.
    raw_lines = str(value or "").splitlines() or [str(value or "")]
    if len(raw_lines) > 1:
        lines: list[str] = []
        for raw_line in raw_lines:
            normalized_line = _normalize_cell_lines(raw_line)
            if normalized_line:
                lines.append(" ".join(normalized_line))
            elif raw_line.strip():
                lines.append(raw_line.strip())
    else:
        lines = _normalize_cell_lines(value)
    if not lines:
        return ""
    return "<br>".join(line.replace("|", "\\|") for line in lines)


def _format_header_markdown_cell(value: str) -> str:
    # Header rows are already semantically separated, so preserve their row boundaries directly.
    parts = [str(part or "").strip().replace("|", "\\|") for part in str(value or "").splitlines()]
    parts = [part for part in parts if part]
    return "<br>".join(parts)


def _is_colored_line(edge: dict[str, Any]) -> bool:
    # Explicit color lines are treated as strong table boundary signals.
    color = edge.get("stroking_color")
    if color is None:
        color = edge.get("non_stroking_color")
    if color is None:
        return False

    # Single-channel colors are considered colored only when not white-like and not near-black.
    if isinstance(color, (int, float)):
        value = float(color)
        if value <= 0.01 or value >= 0.99:
            return False
        return True

    # RGB-like values: accept non-gray or non-binary black/white channels.
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        values = [float(v) for v in color[:3]]
        if all(0.95 <= value <= 1.02 for value in values):
            return False
        if max(values) - min(values) <= 0.02 and max(values) <= 0.20:
            return False
        return True

    return False


def _table_regions(
    page: pdfplumber.page.PageObject,
    y_tolerance: float = 65.0,
    min_lines: int = 3,
    excluded_bboxes: Sequence[Tuple[float, float, float, float]] | None = None,
) -> List[tuple]:
    # Table discovery is driven by connected edge geometry rather than text layout alone.
    del y_tolerance
    excluded_bboxes = list(excluded_bboxes or [])

    def _horizontal_edge_owned_by_excluded_bbox(
        edge: dict[str, Any],
        bbox: Tuple[float, float, float, float],
        *,
        x_tolerance: float = 3.0,
        y_tolerance: float = 6.0,
    ) -> bool:
        edge_x0 = float(edge["x0"])
        edge_x1 = float(edge["x1"])
        edge_center_y = (float(edge["top"]) + float(edge["bottom"])) / 2.0
        return (
            abs(edge_x0 - float(bbox[0])) <= x_tolerance
            and abs(edge_x1 - float(bbox[2])) <= x_tolerance
            and edge_center_y >= float(bbox[1]) - y_tolerance
            and edge_center_y <= float(bbox[3]) + y_tolerance
        )

    def _vertical_edge_owned_by_excluded_bbox(
        edge: dict[str, Any],
        bbox: Tuple[float, float, float, float],
        *,
        x_tolerance: float = 3.0,
        min_overlap: float = 8.0,
    ) -> bool:
        edge_center_x = (float(edge["x0"]) + float(edge["x1"])) / 2.0
        if not (
            abs(edge_center_x - float(bbox[0])) <= x_tolerance
            or abs(edge_center_x - float(bbox[2])) <= x_tolerance
        ):
            return False
        overlap = min(float(edge["bottom"]), float(bbox[3])) - max(float(edge["top"]), float(bbox[1]))
        return overlap >= min_overlap

    body_top, body_bottom = _detect_body_bounds(page, header_margin=90.0, footer_margin=40.0)
    horizontal_edges: list[dict[str, Any]] = []
    vertical_edges: list[dict[str, Any]] = []

    for line in getattr(page, "lines", []):
        if not _is_black_line_segment(line):
            continue
        if float(line["bottom"]) <= body_top or float(line["top"]) >= body_bottom:
            continue
        orientation = line.get("orientation")
        if (orientation == "h" or abs(float(line["bottom"]) - float(line["top"])) <= abs(float(line["x1"]) - float(line["x0"]))) and not any(
            _horizontal_edge_owned_by_excluded_bbox(line, bbox) for bbox in excluded_bboxes
        ):
            horizontal_edges.append(dict(line))
        if (orientation == "v" or abs(float(line["x1"]) - float(line["x0"])) < abs(float(line["bottom"]) - float(line["top"]))) and not any(
            _vertical_edge_owned_by_excluded_bbox(line, bbox) for bbox in excluded_bboxes
        ):
            vertical_edges.append(dict(line))

    for rect in getattr(page, "rects", []) or []:
        rect_top = float(rect.get("top", 0.0))
        rect_bottom = float(rect.get("bottom", rect_top))
        if rect_bottom <= body_top or rect_top >= body_bottom:
            continue
        if _is_horizontal_separator_rect(rect) and not any(
            _horizontal_edge_owned_by_excluded_bbox(rect, bbox) for bbox in excluded_bboxes
        ):
            horizontal_edges.append(dict(rect))
        if _is_vertical_separator_rect(rect) and not any(
            _vertical_edge_owned_by_excluded_bbox(rect, bbox) for bbox in excluded_bboxes
        ):
            vertical_edges.append(dict(rect))

    if not horizontal_edges:
        horizontal_edges = [
            edge
            for edge in page.horizontal_edges
            if float(edge["bottom"]) > body_top
            and float(edge["top"]) < body_bottom
            and not any(_horizontal_edge_owned_by_excluded_bbox(edge, bbox) for bbox in excluded_bboxes)
        ]
    if not vertical_edges:
        vertical_edges = [
            edge
            for edge in page.vertical_edges
            if float(edge["bottom"]) > body_top
            and float(edge["top"]) < body_bottom
            and not any(_vertical_edge_owned_by_excluded_bbox(edge, bbox) for bbox in excluded_bboxes)
        ]

    merged_h = []
    for group in _build_segment_groups(horizontal_edges, axis_key="top", merge_fn=_merge_horizontal_band_segments, tolerance=1.0):
        merged_h.extend(group["merged_segments"])

    merged_v = []
    for group in _build_segment_groups(vertical_edges, axis_key="x0", merge_fn=_merge_vertical_band_segments, tolerance=1.0):
        merged_v.extend(group["merged_segments"])

    if not merged_h:
        return []

    graph: List[set[int]] = [set() for _ in range(len(merged_h))]
    component_verticals: List[set[int]] = [set() for _ in range(len(merged_h))]
    tolerance = 1.0

    for h_idx, h_edge in enumerate(merged_h):
        for v_idx, v_edge in enumerate(merged_v):
            # A horizontal line joins a component only when a vertical edge crosses it within tolerance.
            intersects = (
                float(v_edge["x0"]) >= float(h_edge["x0"]) - tolerance
                and float(v_edge["x0"]) <= float(h_edge["x1"]) + tolerance
                and float(h_edge["top"]) >= float(v_edge["top"]) - tolerance
                and float(h_edge["top"]) <= float(v_edge["bottom"]) + tolerance
            )
            if intersects:
                component_verticals[h_idx].add(v_idx)

    for i in range(len(merged_h)):
        for j in range(i + 1, len(merged_h)):
            if component_verticals[i] and component_verticals[j]:
                shared_vertical = component_verticals[i].intersection(component_verticals[j])
                if shared_vertical:
                    graph[i].add(j)
                    graph[j].add(i)

    visited = set()
    groups: List[tuple] = []
    for start in range(len(merged_h)):
        if start in visited:
            continue
        stack = [start]
        component = []
        shared_verticals = set()
        while stack:
            idx = stack.pop()
            if idx in visited:
                continue
            visited.add(idx)
            component.append(idx)
            shared_verticals.update(component_verticals[idx])
            stack.extend(graph[idx] - visited)

        component_lines = [merged_h[idx] for idx in component]
        # Table regions are admitted by visible grid geometry only:
        # at least two horizontal rules with at least one shared vertical connector.
        if not shared_verticals:
            continue
        if len(component_lines) < 2:
            continue

        x0 = min(float(edge["x0"]) for edge in component_lines)
        x1 = max(float(edge["x1"]) for edge in component_lines)
        x0 = min(x0, *(float(merged_v[idx]["x0"]) for idx in shared_verticals))
        x1 = max(x1, *(float(merged_v[idx]["x1"]) for idx in shared_verticals))
        groups.append((x0, x1, component_lines))

    return sorted(groups, key=lambda item: min(float(edge["top"]) for edge in item[2]))


def _merge_touching_fill_rects(
    rects: Sequence[dict],
    tolerance: float = 1.0,
) -> List[Tuple[float, float, float, float]]:
    # Adjacent fill-only rects often represent one visual box split into multiple PDF drawing objects.
    merged: List[Tuple[float, float, float, float]] = []
    ordered = sorted(
        rects,
        key=lambda rect: (
            round(float(rect.get("top", 0.0)), 1),
            round(float(rect.get("bottom", 0.0)), 1),
            float(rect.get("x0", 0.0)),
        ),
    )
    for rect in ordered:
        candidate = (
            float(rect.get("x0", 0.0)),
            float(rect.get("top", 0.0)),
            float(rect.get("x1", 0.0)),
            float(rect.get("bottom", 0.0)),
        )
        if not merged:
            merged.append(candidate)
            continue

        prev_x0, prev_top, prev_x1, prev_bottom = merged[-1]
        cur_x0, cur_top, cur_x1, cur_bottom = candidate
        same_band = abs(prev_top - cur_top) <= tolerance and abs(prev_bottom - cur_bottom) <= tolerance
        touching = cur_x0 <= prev_x1 + tolerance
        if same_band and touching:
            merged[-1] = (
                min(prev_x0, cur_x0),
                min(prev_top, cur_top),
                max(prev_x1, cur_x1),
                max(prev_bottom, cur_bottom),
            )
            continue
        merged.append(candidate)
    return merged


def _merge_touching_fill_rects_by_color(
    rects: Sequence[dict[str, Any]],
    tolerance: float = 1.0,
    color_tolerance: float = 0.02,
) -> List[Tuple[float, float, float, float]]:
    color_groups: dict[tuple[Any, ...] | int | float | None, list[dict[str, Any]]] = {}
    for rect in rects:
        key = _rect_fill_color_key(rect)
        color_groups.setdefault(key, []).append(rect)

    merged: List[Tuple[float, float, float, float]] = []
    for key, group_rects in color_groups.items():
        merged.extend(_merge_touching_fill_rects(group_rects, tolerance=tolerance))
    return merged


def _strip_coverage_ratio(
    bbox: Tuple[float, float, float, float],
    strip_rects: Sequence[dict[str, Any]],
    line_y: float,
    tolerance: float = 2.0,
) -> float:
    x0, _top, x1, _bottom = bbox
    if x1 <= x0:
        return 0.0

    intervals: List[tuple[float, float]] = []
    for rect in strip_rects:
        rect_top = float(rect.get("top", 0.0))
        rect_bottom = float(rect.get("bottom", 0.0))
        if rect_bottom < line_y - tolerance or rect_top > line_y + tolerance:
            continue

        rx0 = float(rect.get("x0", 0.0))
        rx1 = float(rect.get("x1", 0.0))
        interval_x0 = max(x0, rx0)
        interval_x1 = min(x1, rx1)
        if interval_x1 - interval_x0 > 0.0:
            intervals.append((interval_x0, interval_x1))

    if not intervals:
        return 0.0

    intervals.sort(key=lambda item: item[0])
    merged: list[tuple[float, float]] = [intervals[0]]
    for left, right in intervals[1:]:
        last_left, last_right = merged[-1]
        if left <= last_right + 1.5:
            merged[-1] = (last_left, max(last_right, right))
        else:
            merged.append((left, right))

    covered = sum(right - left for left, right in merged)
    return covered / (x1 - x0)


def _contains_bbox(outer: Tuple[float, float, float, float], inner: Tuple[float, float, float, float], *, overlap_ratio: float = 0.98) -> bool:
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    if ix0 < ox0 or ix1 > ox1 or iy0 < oy0 or iy1 > oy1:
        return False

    intersection = (min(ox1, ix1) - max(ox0, ix0)) * (min(oy1, iy1) - max(oy0, iy0))
    if intersection <= 0.0:
        return False

    inner_area = (ix1 - ix0) * (iy1 - iy0)
    if inner_area <= 0.0:
        return False
    return intersection / inner_area >= overlap_ratio


def _to_rect_entry(rect: dict[str, Any] | tuple[float, float, float, float]) -> dict[str, Any]:
    if isinstance(rect, dict):
        return {
            "x0": float(rect.get("x0", 0.0)),
            "top": float(rect.get("top", rect.get("y0", 0.0))),
            "x1": float(rect.get("x1", 0.0)),
            "bottom": float(rect.get("bottom", rect.get("y1", 0.0))),
            "fill": bool(rect.get("fill", True)),
            "stroke": bool(rect.get("stroke", False)),
            "non_stroking_color": rect.get("non_stroking_color"),
            "stroking_color": rect.get("stroking_color"),
        }

    if len(rect) == 4:
        x0, top, x1, bottom = rect
        return {
            "x0": float(x0),
            "top": float(top),
            "x1": float(x1),
            "bottom": float(bottom),
            "fill": True,
            "stroke": False,
            "non_stroking_color": None,
            "stroking_color": None,
        }

    raise TypeError(f"unsupported rect type: {type(rect)!r}")


def _dedupe_redundant_rectangles(
    rects: Sequence[dict[str, Any] | tuple[float, float, float, float]],
) -> List[dict[str, Any]]:
    if len(rects) <= 1:
        return [_to_rect_entry(rect) for rect in rects]

    converted = [_to_rect_entry(rect) for rect in rects]
    ordered = sorted(
        converted,
        key=lambda rect: -(rect["x1"] - rect["x0"]) * (rect["bottom"] - rect["top"]),
    )

    kept: List[dict[str, Any]] = []
    for rect in ordered:
        if not bool(rect.get("fill")):
            continue
        candidate = (
            float(rect.get("x0", 0.0)),
            float(rect.get("top", 0.0)),
            float(rect.get("x1", 0.0)),
            float(rect.get("bottom", 0.0)),
        )
        keep = True
        for existing in kept:
            existing_bbox = (
                float(existing.get("x0", 0.0)),
                float(existing.get("top", 0.0)),
                float(existing.get("x1", 0.0)),
                float(existing.get("bottom", 0.0)),
            )
            if not _is_nearly_white_color(rect.get("non_stroking_color", rect.get("stroking_color"))):
                continue
            if _contains_bbox(existing_bbox, candidate):
                keep = False
                break
        if keep:
            kept.append(rect)
    return kept


def _bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_x_overlap_ratio(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax0, _, ax1, _ = a
    bx0, _, bx1, _ = b
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    min_width = max(1.0, min(ax1 - ax0, bx1 - bx0))
    return intersection / min_width


def _is_nearly_white_color(value: object) -> bool:
    if value is None:
        return False

    if isinstance(value, (int, float)):
        return float(value) >= 0.98

    if isinstance(value, (list, tuple)) and len(value) >= 3:
        components = [float(component) for component in value[:3]]
        return all(0.95 <= component <= 1.02 for component in components)

    return False


def _candidate_image_regions_for_notes(
    page: pdfplumber.page.PageObject,
    min_width: float = 12.0,
    min_height: float = 10.0,
) -> List[Tuple[float, float, float, float]]:
    # Image anchors are used as optional boundaries for prose-like one-column regions.
    body_top, body_bottom = _detect_body_bounds(page, header_margin=90.0, footer_margin=40.0)
    regions: List[Tuple[float, float, float, float]] = []
    for image in getattr(page, "images", []) or []:
        x0 = float(image.get("x0", 0.0))
        top = float(image.get("top", 0.0))
        x1 = float(image.get("x1", x0))
        bottom = float(image.get("bottom", top))
        if x1 <= x0 or bottom <= top:
            continue
        if (x1 - x0) < min_width or (bottom - top) < min_height:
            continue
        if bottom <= body_top or top >= body_bottom:
            continue
        regions.append((x0, top, x1, bottom))
    regions.sort(key=lambda bbox: (bbox[1], bbox[0]))
    return regions


def _horizontal_separator_lines(
    page: pdfplumber.page.PageObject,
    min_length: float = 90.0,
) -> List[Tuple[float, float, float, float]]:
    # Use horizontal separators as hard stop points for single-column note merging.
    body_top, body_bottom = _detect_body_bounds(page, header_margin=90.0, footer_margin=40.0)
    lines: List[Tuple[float, float, float, float]] = []
    for edge in getattr(page, "horizontal_edges", []):
        x0 = float(edge.get("x0", 0.0))
        x1 = float(edge.get("x1", x0))
        y = float(edge.get("top", edge.get("y0", 0.0)))
        if y <= body_top or y >= body_bottom:
            continue
        if x1 - x0 >= min_length:
            lines.append((x0, y, x1, y))
    lines.sort(key=lambda line: (line[1], line[0]))
    return lines


def _blue_note_horizontal_segments(
    page: pdfplumber.page.PageObject,
    min_length: float = 90.0,
) -> List[dict[str, Any]]:
    body_top, body_bottom = _detect_body_bounds(page, header_margin=90.0, footer_margin=40.0)

    def is_blue_color(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return False
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            r = float(value[0])
            g = float(value[1])
            b = float(value[2])
            return b > 0.45 and r < 0.35 and g < 0.35
        return False

    segments: list[dict[str, Any]] = []
    for edge in getattr(page, "horizontal_edges", []):
        x0 = float(edge.get("x0", 0.0))
        x1 = float(edge.get("x1", x0))
        top = float(edge.get("top", edge.get("y0", 0.0)))
        bottom = float(edge.get("bottom", top))
        if top <= body_top or bottom >= body_bottom:
            continue
        if x1 - x0 < min_length:
            continue
        color = _line_color_key(edge)
        if not is_blue_color(color):
            continue
        segments.append({"x0": x0, "x1": x1, "top": top, "bottom": bottom, "color": color})

    for rect in getattr(page, "rects", []) or []:
        if not bool(rect.get("fill")) or bool(rect.get("stroke")):
            continue
        rect_top = float(rect.get("top", 0.0))
        rect_bottom = float(rect.get("bottom", rect_top))
        if rect_top <= body_top or rect_bottom >= body_bottom:
            continue
        if abs(rect_bottom - rect_top) > _THIN_FILL_RECT_MAX_HEIGHT:
            continue
        x0 = float(rect.get("x0", 0.0))
        x1 = float(rect.get("x1", x0))
        if x1 - x0 < min_length:
            continue
        color = rect.get("non_stroking_color")
        if color is None:
            color = rect.get("stroking_color")
        if not is_blue_color(color):
            continue
        segments.append({"x0": x0, "x1": x1, "top": rect_top, "bottom": rect_bottom, "color": _line_color_key({"stroking_color": color})})

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float, float, tuple[Any, ...] | int | float | None]] = set()
    for segment in sorted(segments, key=lambda item: (float(item["top"]), float(item["x0"]))):
        key = (
            round(float(segment["x0"]), 2),
            round(float(segment["top"]), 2),
            round(float(segment["x1"]), 2),
            round(float(segment["bottom"]), 2),
            segment["color"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(segment)
    return deduped


def _note_group_region_candidates(
    page: pdfplumber.page.PageObject,
    image_regions: Sequence[Tuple[float, float, float, float]] | None = None,
) -> List[Tuple[float, float, float, float]]:
    image_regions = list(image_regions or _candidate_image_regions_for_notes(page))
    if not image_regions:
        return []

    segments = _blue_note_horizontal_segments(page)
    if not segments:
        return []

    candidates: list[Tuple[float, float, float, float]] = []
    start_tolerance = 12.0
    end_tolerance = 12.0

    for anchor in image_regions:
        anchor_area = (float(anchor[0]), float(anchor[1]), float(anchor[2]), float(anchor[3]))
        top_segment: dict[str, Any] | None = None
        bottom_segment: dict[str, Any] | None = None

        for segment in segments:
            segment_area = (
                float(segment["x0"]),
                float(segment["top"]),
                float(segment["x1"]),
                float(segment["bottom"]),
            )
            if _bbox_x_overlap_ratio(anchor_area, segment_area) < 0.10:
                continue
            if float(segment["bottom"]) <= float(anchor[1]):
                top_segment = segment
            elif float(segment["top"]) >= float(anchor[3]):
                if bottom_segment is None:
                    bottom_segment = segment
                break

        if top_segment is None or bottom_segment is None:
            continue
        if not _normalize_color_match(top_segment["color"], bottom_segment["color"]):
            continue
        if abs(float(top_segment["x0"]) - float(bottom_segment["x0"])) > start_tolerance:
            continue
        if abs(float(top_segment["x1"]) - float(bottom_segment["x1"])) > end_tolerance:
            continue

        candidates.append(
            (
                min(float(top_segment["x0"]), float(bottom_segment["x0"])),
                min(float(top_segment["top"]), float(top_segment["bottom"])),
                max(float(top_segment["x1"]), float(bottom_segment["x1"])),
                max(float(bottom_segment["top"]), float(bottom_segment["bottom"])),
            )
        )

    if not candidates:
        return []

    deduped: list[Tuple[float, float, float, float]] = []
    seen: set[Tuple[float, float, float, float]] = set()
    for bbox in sorted(candidates, key=lambda item: (item[1], item[0])):
        key = tuple(round(float(value), 2) for value in bbox)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(bbox)
    return deduped


def _select_note_anchor_for_bbox(
    page: pdfplumber.page.PageObject,
    bbox: Tuple[float, float, float, float],
    image_regions: Sequence[Tuple[float, float, float, float]] | None = None,
) -> Tuple[float, float, float, float] | None:
    image_regions = list(image_regions or _candidate_image_regions_for_notes(page))
    if not image_regions:
        return None

    note_y0, note_y1 = bbox[1], bbox[3]
    note_center_y = (note_y0 + note_y1) / 2.0
    best_anchor: Tuple[float, float, float, float] | None = None
    best_score = -1.0

    for region in image_regions:
        _image_x0, image_top, _image_x1, image_bottom = region
        x_overlap = _bbox_x_overlap_ratio(region, bbox)
        if x_overlap < 0.10:
            continue
        if image_bottom < note_y0 or image_top > note_y1:
            continue

        image_center_y = (image_top + image_bottom) / 2.0
        score = x_overlap * 100.0 - abs(note_center_y - image_center_y)
        if score > best_score:
            best_score = score
            best_anchor = region

    return best_anchor


def _note_anchors_for_bbox(
    page: pdfplumber.page.PageObject,
    bbox: Tuple[float, float, float, float],
    image_regions: Sequence[Tuple[float, float, float, float]] | None = None,
) -> List[Tuple[float, float, float, float]]:
    image_regions = list(image_regions or _candidate_image_regions_for_notes(page))
    if not image_regions:
        return []

    note_y0, note_y1 = bbox[1], bbox[3]
    matches: list[Tuple[float, float, float, float]] = []
    for region in image_regions:
        _image_x0, image_top, _image_x1, image_bottom = region
        x_overlap = _bbox_x_overlap_ratio(region, bbox)
        if x_overlap < 0.10:
            continue
        if image_bottom < note_y0 or image_top > note_y1:
            continue
        matches.append(region)

    deduped: list[Tuple[float, float, float, float]] = []
    seen: set[Tuple[float, float, float, float]] = set()
    for region in sorted(matches, key=lambda item: (item[1], item[0])):
        key = tuple(round(float(value), 2) for value in region)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(region)
    return deduped


def _split_note_rows_by_anchors(
    page: pdfplumber.page.PageObject,
    bbox: Tuple[float, float, float, float],
    *,
    image_regions: Sequence[Tuple[float, float, float, float]] | None = None,
) -> List[dict[str, Any]]:
    anchors = _note_anchors_for_bbox(page, bbox, image_regions=image_regions)
    if len(anchors) <= 1:
        return []

    line_payloads = _extract_region_line_payloads(page, bbox)
    if not line_payloads:
        return []

    rows_by_anchor: list[list[list[str]]] = [[] for _ in anchors]
    tops_by_anchor: list[list[float]] = [[] for _ in anchors]
    bottoms_by_anchor: list[list[float]] = [[] for _ in anchors]
    for line in line_payloads:
        line_text = _normalize_text(str(line.get("text") or ""))
        if not line_text:
            continue
        anchor_index = 0
        line_top = float(line["top"])
        for idx, anchor in enumerate(anchors):
            next_anchor_top = float(anchors[idx + 1][1]) if idx + 1 < len(anchors) else float("inf")
            if line_top < next_anchor_top:
                anchor_index = idx
                break
        rows_by_anchor[anchor_index].append([line_text])
        tops_by_anchor[anchor_index].append(float(line["top"]))
        bottoms_by_anchor[anchor_index].append(float(line["bottom"]))

    split_candidates: list[dict[str, Any]] = []
    for idx, anchor in enumerate(anchors):
        rows = _compact_fallback_rows(rows_by_anchor[idx])
        if not rows:
            continue
        split_bbox = (
            bbox[0],
            min(anchor[1], min(tops_by_anchor[idx], default=anchor[1])),
            bbox[2],
            max(anchor[3], max(bottoms_by_anchor[idx], default=anchor[3])),
        )
        split_candidates.append(
            {
                "bbox": split_bbox,
                "rows": rows,
                "note_anchor": tuple(round(value, 2) for value in anchor),
            }
        )
    return split_candidates


def _extract_tables_from_crop(
    page: pdfplumber.page.PageObject,
    crop_bbox: Tuple[float, float, float, float],
    *,
    fallback_to_text_rows: bool = False,
    strategy_debug: list[dict] | None = None,
    strategy_source: str = "crop",
    strategy_source_name: str | None = None,
) -> List[TableChunk]:
    # Stage 1: rebuild rows/columns from black line geometry and line-payload text assignment.
    line_rows, row_line_positions, column_line_positions, row_bands, column_bands, line_payload_debug = (
        _build_grid_rows_from_black_lines(page, crop_bbox)
    )

    if line_rows:
        if strategy_debug is not None:
            strategy_debug.append(
                {
                    "source": strategy_source,
                    "source_name": strategy_source_name,
                    "crop_bbox": [round(float(value), 2) for value in crop_bbox],
                    "mode": "line_grid",
                    "strategy_index": 0,
                    "status": "success",
                    "row_line_count": len(row_line_positions),
                    "column_line_count": len(column_line_positions),
                    "row_band_count": len(row_bands),
                    "column_band_count": len(column_bands),
                    "raw_row_lines": int(line_payload_debug.get("raw_row_lines", 0)),
                    "raw_vertical_lines": int(line_payload_debug.get("raw_vertical_lines", 0)),
                    "raw_payload_count": int(line_payload_debug.get("raw_payload_count", 0)),
                    "assigned_payload_count": int(line_payload_debug.get("assigned_payload_count", 0)),
                    "ambiguous_payload_count": int(line_payload_debug.get("ambiguous_payload_count", 0)),
                    "unassigned_payload_count": int(line_payload_debug.get("unassigned_payload_count", 0)),
                    "used_fallback_to_text_rows": False,
                }
            )
        return [(line_rows, crop_bbox)]

    if strategy_debug is not None:
        strategy_debug.append(
            {
                "source": strategy_source,
                "source_name": strategy_source_name,
                "crop_bbox": [round(float(value), 2) for value in crop_bbox],
                "mode": "line_grid",
                "strategy_index": 0,
                "status": "failed",
                "failure_reason": line_payload_debug.get("reason", "line_grid_failed"),
                "failure_stage": line_payload_debug.get("stage"),
                "row_line_count": len(row_line_positions),
                "column_line_count": len(column_line_positions),
                "row_band_count": len(row_bands),
                "column_band_count": len(column_bands),
                "raw_row_lines": int(line_payload_debug.get("raw_row_lines", 0)),
                "raw_vertical_lines": int(line_payload_debug.get("raw_vertical_lines", 0)),
                "raw_payload_count": int(line_payload_debug.get("raw_payload_count", 0)),
                "assigned_payload_count": int(line_payload_debug.get("assigned_payload_count", 0)),
                "ambiguous_payload_count": int(line_payload_debug.get("ambiguous_payload_count", 0)),
                "unassigned_payload_count": int(line_payload_debug.get("unassigned_payload_count", 0)),
                "used_fallback_to_text_rows": fallback_to_text_rows,
            }
        )

    if fallback_to_text_rows:
        line_rows = _compact_fallback_rows(_extract_region_line_rows(page, crop_bbox))
        if strategy_debug is not None:
            strategy_debug.append(
                {
                    "source": strategy_source,
                    "source_name": strategy_source_name,
                    "crop_bbox": [round(float(value), 2) for value in crop_bbox],
                    "strategy_index": 1,
                    "mode": "line_fallback",
                    "raw_table_count": 1 if line_rows else 0,
                    "raw_row_count": len(line_rows),
                    "kept_table_count": 1 if line_rows else 0,
                    "kept_row_count": len(line_rows),
                    "rejected_count": 0,
                    "rejections": [],
                    "used_fallback_to_text_rows": True,
                }
            )
        if line_rows:
            return [(line_rows, crop_bbox)]

    # legacy_path = page.extract_tables(
    #     table_settings={
    #         "vertical_strategy": "lines",
    #         "horizontal_strategy": "lines",
    #         "min_words_vertical": 2,
    #         "min_words_horizontal": 2,
    #         "snap_tolerance": 2,
    #         "join_tolerance": 1,
    #     }
    # )
    # if legacy_path:
    #     legacy_rows = _compact_fallback_rows(legacy_path[0]) if legacy_path else []
    #     if legacy_rows:
    #         return [(legacy_rows, crop_bbox)]

    return []


def _extract_tables(
    page: pdfplumber.page.PageObject,
    force_table: bool = False,
    strategy_debug: list[dict] | None = None,
    excluded_bboxes: Sequence[Tuple[float, float, float, float]] | None = None,
) -> List[TableChunk]:
    page = _filter_page_for_extraction(page)
    seen_keys = set()
    merged: List[TableChunk] = []
    table_regions = _table_regions(page, excluded_bboxes=excluded_bboxes)

    def _table_key(
        rows: Sequence[Sequence[str]],
        bbox: Tuple[float, float, float, float],
    ) -> tuple[tuple[tuple[str, ...], ...], tuple[float, float, float, float]]:
        normalized_rows = tuple(
            tuple(_normalize_text(str(cell)) for cell in row) for row in rows
        )
        normalized_bbox = tuple(round(float(v), 2) for v in bbox)
        return normalized_rows, normalized_bbox

    for region_index, (x0, x1, lines) in enumerate(table_regions):
        y0 = min(edge["top"] for edge in lines) - 2
        y1 = max(edge["top"] for edge in lines) + 2
        crop_bbox = (max(0.0, x0), max(0.0, y0), min(page.width, x1), min(page.height, y1))
        for table, crop_box in _extract_tables_from_crop(
            page,
            crop_bbox,
            fallback_to_text_rows=False,
            strategy_debug=strategy_debug,
            strategy_source="table_region",
            strategy_source_name=f"table_region#{region_index}",
        ):
            table = _normalize_extracted_table(table)
            key = _table_key(table, crop_box)
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append((table, crop_box))

    if merged or not force_table:
        return merged
    return merged


def _table_text_from_rows(rows: Sequence[Sequence[str]]) -> str:
    # Convert normalized row data into markdown table text used by the final artifacts.
    if not rows:
        return ""

    header_row_count = _header_row_count(rows)
    header_rows = rows[:header_row_count] if header_row_count else rows[:1]
    header = _collapse_header_rows(header_rows)
    if not any(_normalize_text(cell) for cell in header) and rows:
        first_row = list(rows[0])
        header = [str(cell or "").strip() for cell in first_row]
    body = rows[header_row_count:] if header_row_count else rows[1:]
    if not body:
        if len(rows) > 1:
            header = [str(cell or "").strip() for cell in rows[0]]
            body = [list(row) for row in rows[1:]]
        else:
            body = []

    formatted_header = [
        _format_header_markdown_cell(cell or f"Column {idx + 1}")
        for idx, cell in enumerate(header)
    ]
    formatted_body = []
    for row in body:
        padded_row = list(row) + [""] * max(0, len(header) - len(row))
        formatted_body.append([_format_markdown_cell(str(value or "")) for value in padded_row])

    header_line = "| " + " | ".join(formatted_header) + " |"
    divider_line = "| " + " | ".join("---" for _ in header) + " |"
    body_lines = []
    for row in formatted_body:
        body_lines.append("| " + " | ".join(row) + " |")
    return "\n".join([header_line, divider_line, *body_lines])


def _format_page_comment(page_no: int) -> str:
    return f"[//]: # (Page {page_no})"


def _merge_split_rows(rows: TableRows) -> TableRows:
    # Post-process rows that were extracted as separate fragments even though they belong to the same logical row.
    if not rows:
        return rows

    merged: TableRows = [list(rows[0])]
    for row in rows[1:]:
        if not any(_normalize_text(cell) for cell in row):
            continue
        can_merge_header, header_idx, joiner = _can_merge_header_split_rows(
            merged[-1],
            row,
            len(merged),
        )
        if can_merge_header:
            merged[-1][header_idx] = f"{merged[-1][header_idx]}{joiner}{_normalize_text(row[header_idx])}"
            continue

        non_empty = [idx for idx, cell in enumerate(row) if _normalize_text(cell)]
        if len(merged) > 1 and row and not _normalize_text(row[0]):
            previous = merged[-1]
            previous_second = _normalize_text(previous[1]) if len(previous) > 1 else ""
            current_second = _normalize_text(row[1]) if len(row) > 1 else ""
            if len(non_empty) == 1 and non_empty[0] > 0:
                idx = non_empty[0]
                joiner = "\n" if previous[idx].strip() else ""
                previous[idx] = f"{previous[idx]}{joiner}{row[idx]}".strip()
                continue
            if len(non_empty) >= 2 and 1 in non_empty and 2 in non_empty and previous_second and current_second == previous_second:
                joiner = "\n" if previous[2].strip() else ""
                previous[2] = f"{previous[2]}{joiner}{row[2]}".strip()
                continue
        merged.append(list(row))
    return merged


def _append_output_table(
    output_tables: List[str],
    document_id: str,
    table_no: int,
    table_rows: TableRows,
    *,
    page_no: int | None = None,
) -> None:
    # Table numbering is derived at append time so merged cross-page tables keep one output block.
    merged_rows = _collapse_empty_columns(_merge_split_rows(table_rows))
    table_text = _table_text_from_rows(merged_rows)
    if table_text:
        block = f"[{document_id}_tables.md - Table {table_no}]\n{table_text}"
        if page_no is not None:
            output_tables.append(f"{_format_page_comment(page_no)}\n{block}")
        else:
            output_tables.append(block)
