from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

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
    _detect_body_bounds,
    _filter_page_for_extraction,
    _is_layout_artifact,
    _normalize_cell_lines,
)


def _merge_cells(table: Sequence[Sequence[str]]) -> List[List[str]]:
    return [[str(cell or "").strip() for cell in row] for row in table]


def _collapse_structural_triplet_columns(table: Sequence[Sequence[str]]) -> List[List[str]]:
    rows = [list(row) for row in table]
    if not rows:
        return []

    col_count = max((len(row) for row in rows), default=0)
    if col_count == 0 or col_count % 3 != 0:
        return [list(row) for row in rows]

    padded_rows = [list(row) + [""] * (col_count - len(row)) for row in rows]
    collapsed_indices: List[int] = []
    for start in range(0, col_count, 3):
        left_values = [_normalize_text(row[start]) for row in padded_rows]
        right_values = [_normalize_text(row[start + 2]) for row in padded_rows]
        if any(left_values) or any(right_values):
            collapsed_indices.extend([start, start + 1, start + 2])
            continue
        collapsed_indices.append(start + 1)

    return [[row[idx] for idx in collapsed_indices] for row in padded_rows]


def _normalize_extracted_table(table: Sequence[Sequence[str]]) -> List[List[str]]:
    normalized: List[List[str]] = []
    for row in table:
        normalized_row = []
        for cell in row:
            normalized_row.append("\n".join(_normalize_cell_lines(str(cell or ""))))
        normalized.append(normalized_row)
    return _collapse_structural_triplet_columns(normalized)


def _table_rejection_reason(table: Sequence[Sequence[str]]) -> str | None:
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
    row_count = len(table)
    col_count = max((len(row) for row in table), default=0)
    bbox_text = ", ".join(f"{value:.2f}" for value in crop_bbox)
    print(f"[table-reject] bbox=({bbox_text}) rows={row_count} cols={col_count} reason={reason}")


def _looks_like_header_row(row: Sequence[str]) -> bool:
    if not row:
        return False
    normalized = [_normalize_text(c) for c in row]
    tokens = [cell for cell in normalized if cell]
    if not tokens:
        return False
    alpha_like = sum(1 for token in tokens if re.fullmatch(r"[A-Za-z][A-Za-z0-9\s/&._:-]*", token))
    short = sum(1 for token in tokens if len(token) <= 24)
    return alpha_like >= len(tokens) * 0.8 and short >= len(tokens) * 0.8


def _rows_match(a: Sequence[str], b: Sequence[str]) -> bool:
    if len(a) != len(b):
        return False
    return all(_normalize_text(x) == _normalize_text(y) for x, y in zip(a, b))


def _split_repeated_header(prev_rows: TableRows, curr_rows: TableRows) -> TableRows:
    if prev_rows and curr_rows and _rows_match(prev_rows[0], curr_rows[0]):
        return curr_rows[1:]
    return curr_rows


def _is_continuation_chunk(prev_rows: TableRows, curr_rows: TableRows) -> bool:
    if not prev_rows or not curr_rows:
        return False
    if len(prev_rows[0]) != len(curr_rows[0]):
        return False
    first = curr_rows[0]
    if not first or _looks_like_header_row(first) or _normalize_text(first[0]):
        return False
    return any(_normalize_text(cell) for cell in first[1:])


def _extract_continuation_lines(
    page_text: str,
    repeated_header: str,
    next_row_label: str,
) -> List[str]:
    lines = [line.strip() for line in str(page_text or "").splitlines() if line.strip()]
    if not lines:
        return []
    try:
        start_idx = next(i for i, line in enumerate(lines) if _normalize_text(line) == _normalize_text(repeated_header))
    except StopIteration:
        return []

    tail = lines[start_idx + 1 :]
    if not tail:
        return []

    stop_idx = len(tail)
    normalized_label = _normalize_text(next_row_label)
    for i, line in enumerate(tail):
        if _normalize_text(line).startswith(normalized_label):
            stop_idx = i
            break
    return tail[:stop_idx]


def _normalize_two_col_continuation_rows(rows: TableRows) -> TableRows:
    return [["", str(row[0] or "").strip(), str(row[1] or "").strip()] for row in rows if row]


def _should_try_table_continuation_merge(
    pending_page: int | None,
    current_page: int,
) -> bool:
    return pending_page is not None and current_page == pending_page + 1


def _body_text_boxes(
    page: pdfplumber.page.PageObject,
    header_margin: float,
    footer_margin: float,
    excluded_bboxes: Sequence[Tuple[float, float, float, float]] = (),
) -> List[Tuple[float, float, float, float]]:
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


def _gap_text_boxes_after_bbox(
    page: pdfplumber.page.PageObject,
    bbox: Tuple[float, float, float, float],
    table_bboxes: Sequence[Tuple[float, float, float, float]],
    header_margin: float,
    footer_margin: float,
) -> List[Tuple[float, float, float, float]]:
    return [
        text_bbox
        for text_bbox in _body_text_boxes(
            page,
            header_margin=header_margin,
            footer_margin=footer_margin,
            excluded_bboxes=table_bboxes,
        )
        if float(text_bbox[1]) >= float(bbox[3])
    ]


def _gap_text_boxes_before_bbox(
    page: pdfplumber.page.PageObject,
    bbox: Tuple[float, float, float, float],
    table_bboxes: Sequence[Tuple[float, float, float, float]],
    header_margin: float,
    footer_margin: float,
) -> List[Tuple[float, float, float, float]]:
    return [
        text_bbox
        for text_bbox in _body_text_boxes(
            page,
            header_margin=header_margin,
            footer_margin=footer_margin,
            excluded_bboxes=table_bboxes,
        )
        if float(text_bbox[3]) <= float(bbox[1])
    ]


def _vertical_axes_for_bbox(
    page: pdfplumber.page.PageObject,
    bbox: Tuple[float, float, float, float],
) -> List[float]:
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
    gap_text_boxes: Sequence[Tuple[float, float, float, float]],
    edge_tolerance: float = 24.0,
    axis_tolerance: float = 1.0,
) -> bool:
    _prev_x0, _prev_top, _prev_x1, prev_bottom = prev_bbox
    _curr_x0, curr_top, _curr_x1, _curr_bottom = curr_bbox

    shared_axes = [axis for axis in prev_axes if any(abs(axis - other) <= axis_tolerance for other in curr_axes)]
    if not shared_axes or gap_text_boxes:
        return False

    prev_near_footer = abs(body_bottom - prev_bottom) <= edge_tolerance
    curr_near_header = abs(curr_top - body_top) <= edge_tolerance
    if prev_near_footer or curr_near_header:
        return True
    return True


def _maybe_merge_missing_first_column_chunk(
    pending_table: TableRows | None,
    current_rows: TableRows,
    page_text: str,
) -> TableRows | None:
    if not pending_table or not current_rows:
        return None
    if len(pending_table[0]) != 3:
        return None
    if any(len(row) != 2 for row in current_rows):
        return None

    last_row = pending_table[-1]
    if len(last_row) < 3 or not _normalize_text(last_row[1]):
        return None

    repeated_header = " ".join(str(cell or "").strip() for cell in pending_table[0]).strip()
    continuation_lines = _extract_continuation_lines(page_text, repeated_header, str(current_rows[0][0] or ""))
    if continuation_lines:
        joiner = "\n" if str(last_row[2] or "").strip() else ""
        last_row[2] = f"{last_row[2]}{joiner}" + "\n".join(continuation_lines)

    normalized_rows = _normalize_two_col_continuation_rows(current_rows)
    if not normalized_rows:
        return None

    pending_table.extend(normalized_rows)
    return pending_table


def _format_markdown_cell(value: str) -> str:
    lines = _normalize_cell_lines(value)
    if not lines:
        return ""
    return "<br>".join(line.replace("|", "\\|") for line in lines)


def _table_regions(
    page: pdfplumber.page.PageObject,
    y_tolerance: float = 65.0,
    min_lines: int = 3,
) -> List[tuple]:
    del y_tolerance

    body_top, body_bottom = _detect_body_bounds(page, header_margin=90.0, footer_margin=40.0)
    horizontal_edges = [
        edge
        for edge in page.horizontal_edges
        if float(edge["x1"]) - float(edge["x0"]) >= 50.0
        and float(edge["bottom"]) > body_top
        and float(edge["top"]) < body_bottom
    ]
    vertical_edges = [
        edge
        for edge in page.vertical_edges
        if float(edge["bottom"]) > body_top
        and float(edge["top"]) < body_bottom
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
        if len(component_lines) < min_lines or not shared_verticals:
            continue

        x0 = min(float(edge["x0"]) for edge in component_lines)
        x1 = max(float(edge["x1"]) for edge in component_lines)
        x0 = min(x0, *(float(merged_v[idx]["x0"]) for idx in shared_verticals))
        x1 = max(x1, *(float(merged_v[idx]["x1"]) for idx in shared_verticals))
        groups.append((x0, x1, component_lines))

    return sorted(groups, key=lambda item: min(float(edge["top"]) for edge in item[2]))


def _extract_tables_from_crop(
    page: pdfplumber.page.PageObject,
    crop_bbox: Tuple[float, float, float, float],
) -> List[TableChunk]:
    x0, y0, x1, y1 = crop_bbox
    crop = page.crop(crop_bbox)

    v_lines = []
    for edge in page.vertical_edges:
        if edge["x0"] < x0 or edge["x0"] > x1:
            continue
        if edge["top"] > y1 or edge["bottom"] < y0:
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
        cleaned = []
        for table in tables:
            reason = _table_rejection_reason(table)
            if reason is not None:
                _log_rejected_table(table, crop_bbox, reason)
                continue
            cleaned.append(_merge_cells(table))
        if cleaned:
            return [(table, crop_bbox) for table in cleaned]
    return []


def _extract_tables(
    page: pdfplumber.page.PageObject,
    force_table: bool = False,
) -> List[TableChunk]:
    page = _filter_page_for_extraction(page)
    seen_keys = set()
    merged: List[TableChunk] = []

    for x0, x1, lines in _table_regions(page):
        y0 = min(edge["top"] for edge in lines) - 2
        y1 = max(edge["top"] for edge in lines) + 2
        crop_bbox = (max(0.0, x0), max(0.0, y0), min(page.width, x1), min(page.height, y1))
        for table, crop_box in _extract_tables_from_crop(page, crop_bbox):
            table = _normalize_extracted_table(table)
            rows_key = tuple(tuple(row) for row in table)
            bbox_key = tuple(round(v, 2) for v in crop_box)
            key = (rows_key, bbox_key)
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append((table, crop_box))

    if merged or not force_table:
        return merged

    full_bbox = (0.0, 0.0, float(page.width), float(page.height))
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
            "intersection_tolerance": 2,
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
        },
    ]

    for settings in fallback_settings:
        tables = page.extract_tables(table_settings=settings) or []
        cleaned = []
        for table in tables:
            reason = _table_rejection_reason(table)
            if reason is not None:
                _log_rejected_table(table, full_bbox, reason)
                continue
            cleaned.append(_merge_cells(table))
        if cleaned:
            for table in cleaned:
                table = _normalize_extracted_table(table)
                rows_key = tuple(tuple(row) for row in table)
                bbox_key = tuple(round(v, 2) for v in full_bbox)
                key = (rows_key, bbox_key)
                if key not in seen_keys:
                    seen_keys.add(key)
                    merged.append((table, full_bbox))
            break
    return merged


def _table_text_from_rows(rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return ""

    header = [str(col or "").strip() for col in rows[0]]
    body = rows[1:]
    if not body:
        body = rows
        header = [f"Column {idx}" for idx in range(1, len(rows[0]) + 1)]

    header_line = "| " + " | ".join(cell or f"Column {idx + 1}" for idx, cell in enumerate(header)) + " |"
    divider_line = "| " + " | ".join("---" for _ in header) + " |"
    body_lines = []
    for row in body:
        padded_row = list(row) + [""] * max(0, len(header) - len(row))
        body_lines.append("| " + " | ".join(_format_markdown_cell(str(value or "")) for value in padded_row) + " |")
    return "\n".join([header_line, divider_line, *body_lines])


def _merge_split_rows(rows: TableRows) -> TableRows:
    if not rows:
        return rows

    merged: TableRows = [list(rows[0])]
    for row in rows[1:]:
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


def _append_output_table(output_tables: List[str], page_no: int, table_no: int, table_rows: TableRows) -> None:
    table_text = _table_text_from_rows(_merge_split_rows(table_rows))
    if table_text:
        output_tables.append(f"### Page {page_no} table {table_no}\n{table_text}")
