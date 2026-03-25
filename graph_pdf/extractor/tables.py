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
    _detect_body_bounds,
    _filter_page_for_extraction,
    _is_layout_artifact,
    _normalize_cell_lines,
    _repair_watermark_bleed,
)


def _merge_cells(table: Sequence[Sequence[str]]) -> List[List[str]]:
    # pdfplumber can yield `None` cells, so normalize early to simple stripped strings.
    return [[str(cell or "").strip() for cell in row] for row in table]


def _collapse_structural_triplet_columns(table: Sequence[Sequence[str]]) -> List[List[str]]:
    # Remove vertically empty columns that are likely structural artifacts from extraction.
    rows = [list(row) for row in table]
    if not rows:
        return []

    col_count = max((len(row) for row in rows), default=0)
    padded_rows = [list(row) + [""] * (col_count - len(row)) for row in rows]
    kept_indices = [
        idx
        for idx in range(col_count)
        if any(_normalize_text(str(padded_rows[row_idx][idx]).strip()) for row_idx in range(len(padded_rows)))
    ]

    if not kept_indices:
        return [[] for _ in padded_rows]

    return [[row[idx] for idx in kept_indices] for row in padded_rows]


def _normalize_extracted_table(table: Sequence[Sequence[str]]) -> List[List[str]]:
    # Table normalization is deliberately cell-local so geometric table structure stays untouched.
    normalized: List[List[str]] = []
    for row in table:
        normalized_row = []
        for cell in row:
            normalized_row.append("\n".join(_normalize_cell_lines(str(cell or ""))))
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
    # Header detection is heuristic and only used for cross-page continuation handling.
    if not row:
        return False
    normalized = [_normalize_text(c) for c in row]
    tokens = [cell for cell in normalized if cell]
    if not tokens:
        return False
    alpha_like = sum(1 for token in tokens if re.fullmatch(r"[A-Za-z][A-Za-z0-9\s/&._:-]*", token))
    short = sum(1 for token in tokens if len(token) <= 24)
    return alpha_like >= len(tokens) * 0.8 and short >= len(tokens) * 0.8


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


def _is_parameter_description_layout(rows: Sequence[Sequence[str]]) -> bool:
    if not rows:
        return False

    header = [
        _normalize_text(cell)
        for cell in rows[0]
        if _normalize_text(cell)
    ]
    if len(header) < 2:
        return False
    header_text = " ".join(header).lower()
    if "parameter" not in header_text or "description" not in header_text:
        return False

    col_indexes = _effective_non_empty_column_indices(rows)
    if len(col_indexes) < 2 or len(col_indexes) > 3:
        return False

    for row in rows[1:]:
        active = [
            (idx, _normalize_text(cell))
            for idx, cell in enumerate(row)
            if _normalize_text(cell)
        ]
        if not active:
            continue
        if len(active) != 2:
            return False

        key_text = active[0][1]
        value_text = active[1][1]
        if not key_text or len(key_text) > 28:
            return False
        if len(value_text) < 20:
            return False

    return True


def _is_key_value_layout(rows: Sequence[Sequence[str]]) -> bool:
    if len(rows) < 2:
        return False

    col_indexes = _effective_non_empty_column_indices(rows)
    if len(col_indexes) < 2 or len(col_indexes) > 3:
        return False

    qualified_rows = 0
    two_or_three_cell_rows = 0
    short_key_rows = 0

    for row in rows:
        active = [
            (idx, _normalize_text(cell))
            for idx, cell in enumerate(row)
            if _normalize_text(cell)
        ]
        if not active:
            continue

        qualified_rows += 1
        if len(active) > 3:
            return False

        if len(active) in (2, 3):
            two_or_three_cell_rows += 1
            key_text = active[0][1]
            value_cells = [item[1] for item in active[1:]]
            if not key_text or len(key_text) > 40:
                return False
            if all(len(value_text) < 2 for value_text in value_cells):
                return False
            if len(key_text) <= 30:
                short_key_rows += 1

        if len(active) == 1:
            value_text = active[0][1]
            if len(value_text) > 3:
                return False

    if qualified_rows == 0 or two_or_three_cell_rows == 0:
        return False
    if two_or_three_cell_rows / qualified_rows < 0.7:
        return False
    if short_key_rows == 0:
        return False

    return True


def _is_single_column_like_rows(rows: Sequence[Sequence[str]]) -> bool:
    return len(_effective_non_empty_column_indices(rows)) <= 1


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
        .extract_words(x_tolerance=1.5, y_tolerance=2.0, keep_blank_chars=False)
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


def _extract_region_line_rows(
    page: pdfplumber.page.PageObject,
    bbox: Tuple[float, float, float, float],
) -> list[list[str]]:
    lines = _extract_region_lines(_extract_region_words(page, bbox))
    rows: list[list[str]] = []
    for words_in_line in lines:
        text = _line_text_from_words(words_in_line)
        if not text:
            continue
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


def _estimate_region_kind(
    table_rows: Sequence[Sequence[str]],
    region_words: Sequence[dict[str, Any]],
    internal_grid: tuple[int, int],
    *,
    min_long_line_words: int = 6,
    min_total_chars: int = 80,
) -> tuple[str, dict[str, Any]]:
    # Return "table" or "note" with lightweight, explainable signals.
    if _is_parameter_description_layout(table_rows):
        return "table", {
            "reason": "parameter_description_layout",
            "effective_col_count": len(_effective_non_empty_column_indices(table_rows)),
        }

    is_single_column_like = _is_single_column_like_rows(table_rows)
    row_count = len(table_rows)
    col_count = max((len(row) for row in table_rows), default=0)
    if row_count <= 1:
        fallback_kind, fallback_meta = _classify_single_column_rows_only(table_rows)
        if fallback_kind == "note":
            return fallback_kind, {
                "reason": "single_row_fallback",
                **fallback_meta,
            }
    if col_count <= 0 or not region_words:
        return "table", {"reason": "empty_content"}

    text_lines = [_normalize_text(_line_text_from_words(words)) for words in _extract_region_lines(region_words)]
    text_lines = [line for line in text_lines if line]
    if not text_lines:
        return "table", {"reason": "empty_lines"}

    line_word_counts = [len(line.split()) for line in text_lines]
    line_char_counts = [len(line) for line in text_lines]
    avg_words = sum(line_word_counts) / len(line_word_counts)
    max_words = max(line_word_counts)
    max_chars = max(line_char_counts)
    total_chars = sum(line_char_counts)
    long_lines = sum(1 for count in line_word_counts if count >= min_long_line_words)
    punctuation_ratio = sum(1 for line in text_lines if any(ch in line for ch in [".", ",", ";", ":", "(", ")"])) / len(text_lines)
    internal_vertical, internal_horizontal = internal_grid
    table_score = 0.0
    note_score = 0.0

    # Note routing is intentionally single-column-first.
    if not is_single_column_like:
        return "table", {
            "reason": "not_single_column_like",
            "col_count": col_count,
            "effective_col_count": len(_effective_non_empty_column_indices(table_rows)),
        }

    if row_count <= 3 and max(line_word_counts) <= 4 and total_chars < min_total_chars:
        table_score += 2.0
    if max_words <= 4 and long_lines == 0:
        table_score += 1.0
    if internal_vertical + internal_horizontal >= 2:
        table_score += 2.0
    if punctuation_ratio >= 0.2 and total_chars <= min_total_chars:
        table_score += 1.0

    # Prose-like dense lines indicate note-style single-column content.
    if row_count >= 2 and (avg_words >= 5 or max_chars >= 90 or total_chars >= min_total_chars):
        note_score += 2.0
    if long_lines >= 2:
        note_score += 2.0
    if avg_words >= 7 and total_chars >= min_total_chars:
        note_score += 1.0

    if note_score > table_score:
        return "note", {
            "reason": "prose_score",
            "note_score": note_score,
            "table_score": table_score,
            "avg_words": avg_words,
            "max_words": max_words,
            "line_count": len(text_lines),
            "total_chars": total_chars,
        }
    return "table", {
        "reason": "table_score",
        "note_score": note_score,
        "table_score": table_score,
        "avg_words": avg_words,
        "max_words": max_words,
        "line_count": len(text_lines),
        "total_chars": total_chars,
    }


def _classify_single_column_region(
    table_rows: Sequence[Sequence[str]],
    page: Optional[pdfplumber.page.PageObject] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> tuple[str, dict[str, Any]]:
    # Public wrapper used by both table filtering and debug reporting.
    col_count = max((len(row) for row in table_rows), default=0)
    if not _is_single_column_like_rows(table_rows):
        effective_cols = _effective_non_empty_column_indices(table_rows)
        return "table", {
            "reason": "not_single_column_like",
            "col_count": col_count,
            "effective_col_count": len(effective_cols),
        }
    if not page or not bbox:
        return _classify_single_column_rows_only(table_rows)

    region_words = _extract_region_words(page=page, bbox=bbox)
    internal_grid = _internal_grid_counts(page=page, bbox=bbox)
    internal_vertical, internal_horizontal = internal_grid
    if len(table_rows) <= 1 and internal_vertical + internal_horizontal >= 2:
        # A one-line block that already has a visible cell grid is treated as table metadata,
        # because it's structurally a table row/header regardless of prose-like length.
        return "table", {
            "reason": "single_row_grid",
            "table_score": 3.0,
            "note_score": 0.0,
            "internal_vertical": internal_vertical,
            "internal_horizontal": internal_horizontal,
        }
    return _estimate_region_kind(
        table_rows=table_rows,
        region_words=region_words,
        internal_grid=internal_grid,
    )


def _classify_single_column_rows_only(
    rows: Sequence[Sequence[str]],
) -> tuple[str, dict[str, Any]]:
    # Geometry-independent fallback for tests and call-sites that only have cells.
    if not rows:
        return "table", {"reason": "empty_row_region"}
    effective_col_count = len(_effective_non_empty_column_indices(rows))
    if effective_col_count != 1:
        return "table", {"reason": "not_single_column_like", "effective_col_count": effective_col_count}
    normalized_lines = []
    for row in rows:
        line = ""
        for cell in row:
            value = _normalize_text(cell)
            if value:
                line = value
                break
        if line:
            normalized_lines.append(line)
    normalized_lines = [line for line in normalized_lines if line]
    if not normalized_lines:
        return "table", {"reason": "empty_line_region"}

    if len(rows) == 1:
        max_chars = max(len(line) for line in normalized_lines)
        return (
            "note" if max_chars >= 30 else "table",
            {
                "reason": "single_line_row_length",
                "line_count": 1,
                "max_chars": max_chars,
                "max_words": max(len(line.split()) for line in normalized_lines),
            },
        )

    line_word_counts = [len(line.split()) for line in normalized_lines]
    line_char_counts = [len(line) for line in normalized_lines]
    total_chars = sum(line_char_counts)
    avg_words = sum(line_word_counts) / len(line_word_counts)
    avg_chars = total_chars / max(1, len(line_char_counts))
    punctuation_ratio = sum(
        1 for line in normalized_lines if any(ch in line for ch in [".", ",", ";", ":", "(", ")"])
    ) / len(normalized_lines)
    short_rows = sum(1 for count in line_word_counts if count <= 4)

    note_score = 0.0
    table_score = 0.0
    if len(normalized_lines) <= 2 and short_rows >= len(normalized_lines):
        table_score += 2.0
    if avg_chars >= 30 and total_chars >= 45:
        note_score += 1.5
    if avg_words >= 6:
        note_score += 1.5
    if total_chars >= 90:
        note_score += 1.0
    if punctuation_ratio >= 0.2:
        note_score += 0.5
    if table_score >= note_score:
        return "table", {
            "reason": "single_column_fallback",
            "note_score": note_score,
            "table_score": table_score,
            "line_count": len(normalized_lines),
            "avg_chars": avg_chars,
            "total_chars": total_chars,
        }

    return "note", {
        "reason": "single_column_fallback",
        "note_score": note_score,
        "table_score": table_score,
        "line_count": len(normalized_lines),
        "avg_chars": avg_chars,
        "total_chars": total_chars,
    }


def _looks_like_single_column_note(
    rows: Sequence[Sequence[str]],
    page: Optional[pdfplumber.page.PageObject] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> bool:
    # Backward-compatible API used by tests. Spatial hints are optional.
    kind, _ = _classify_single_column_region(table_rows=rows, page=page, bbox=bbox)
    return kind == "note"


def _single_column_note_body_text(rows: Sequence[Sequence[str]]) -> str:
    # Convert multi-line note-like rows into a single body sentence.
    parts: List[str] = []
    for row in rows:
        if not row:
            continue
        leading_text = ""
        for cell in row:
            normalized = _normalize_text(cell)
            if normalized:
                leading_text = normalized
                break
        if not leading_text:
            continue
        for line in _normalize_cell_lines(str(leading_text)):
            normalized = _normalize_text(line)
            if normalized:
                parts.append(normalized)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _rows_match(a: Sequence[str], b: Sequence[str]) -> bool:
    # Header rows are compared after whitespace normalization to avoid duplicate header output.
    if len(a) != len(b):
        return False
    return all(_normalize_text(x) == _normalize_text(y) for x, y in zip(a, b))


def _header_row_count(rows: Sequence[Sequence[str]], max_header_rows: int = 2) -> int:
    # Treat consecutive short alpha-like top rows as a multi-row header block.
    count = 0
    for row in rows[:max_header_rows]:
        if not _looks_like_header_row(row):
            break
        count += 1
    return count


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
        return curr_rows
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


def _extract_continuation_lines(
    page_text: str,
    repeated_header: str,
    next_row_label: str,
) -> List[str]:
    # Missing-first-column continuation rows borrow prose lines from the surrounding page text.
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
    # Reconstruct a 3-column shape when a continuation fragment drops the first column entirely.
    return [["", str(row[0] or "").strip(), str(row[1] or "").strip()] for row in rows if row]


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


def _gap_text_boxes_after_bbox(
    page: pdfplumber.page.PageObject,
    bbox: Tuple[float, float, float, float],
    table_bboxes: Sequence[Tuple[float, float, float, float]],
    header_margin: float,
    footer_margin: float,
) -> List[Tuple[float, float, float, float]]:
    # Text after a table candidate can block continuation into the next page.
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
    # Text before a table candidate can mean the new page starts with prose rather than a continuation table.
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
    gap_text_boxes: Sequence[Tuple[float, float, float, float]],
    edge_tolerance: float = 24.0,
    axis_tolerance: float = 1.0,
    prev_page_height: float | None = None,
) -> bool:
    # Merge only when geometry matches and no body text sits between the two fragments.
    _prev_x0, _prev_top, _prev_x1, prev_bottom = prev_bbox
    _curr_x0, curr_top, _curr_x1, _curr_bottom = curr_bbox

    shared_axes = [axis for axis in prev_axes if any(abs(axis - other) <= axis_tolerance for other in curr_axes)]
    if not shared_axes or gap_text_boxes:
        return False

    # Near-footer / near-header placement is the common continuation pattern, but shared axes already make this permissive.
    prev_near_footer = abs(body_bottom - prev_bottom) <= edge_tolerance
    curr_near_header = abs(curr_top - body_top) <= edge_tolerance
    if prev_near_footer or curr_near_header:
        # Keep current continuation behavior when the fragments are aligned near page edges.
        # This covers the usual table-break pattern while still allowing occasional header/footers offsets.
        return True

    if prev_page_height is None:
        return True

    # Cross-page continuation usually occurs near the top of the next page and near the bottom
    # of the previous page. A large geometric jump is likely a new table block, not a split.
    gap_across_pages = (float(prev_page_height) - prev_bottom) + (curr_top - body_top)
    return gap_across_pages <= 220.0


def _maybe_merge_missing_first_column_chunk(
    pending_table: TableRows | None,
    current_rows: TableRows,
    page_text: str,
) -> TableRows | None:
    # Some PDFs drop the leading column when a long row spills to the next page.
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
    # Markdown cells preserve logical line breaks with `<br>` while escaping literal pipe characters.
    lines = _normalize_cell_lines(value)
    if not lines:
        return ""
    return "<br>".join(line.replace("|", "\\|") for line in lines)


def _format_header_markdown_cell(value: str) -> str:
    # Header rows are already semantically separated, so preserve their row boundaries directly.
    parts = [str(part or "").strip().replace("|", "\\|") for part in str(value or "").splitlines()]
    parts = [part for part in parts if part]
    return "<br>".join(parts)


def _table_regions(
    page: pdfplumber.page.PageObject,
    y_tolerance: float = 65.0,
    min_lines: int = 3,
) -> List[tuple]:
    # Table discovery is driven by connected edge geometry rather than text layout alone.
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
        # Ignore weak components that do not look like a stable table grid.
        if len(component_lines) < min_lines or not shared_verticals:
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


def _thin_strip_rects(rects: Sequence[dict], max_height: float = 1.5) -> List[dict]:
    # Some PDFs use filled rect strips instead of stroked lines for box boundaries.
    return [
        rect
        for rect in rects
        if bool(rect.get("fill"))
        and float(rect.get("bottom", 0.0)) - float(rect.get("top", 0.0)) <= max_height
    ]


def _single_column_box_regions(page: pdfplumber.page.PageObject) -> List[Tuple[float, float, float, float]]:
    # Detect box-like regions that visually behave as one cell even if the PDF uses multiple fill rects inside.
    body_top, body_bottom = _detect_body_bounds(page, header_margin=90.0, footer_margin=40.0)
    fill_rects = [
        rect
        for rect in getattr(page, "rects", [])
        if bool(rect.get("fill"))
        and float(rect.get("bottom", 0.0)) > body_top
        and float(rect.get("top", 0.0)) < body_bottom
    ]
    boundary_rects = _thin_strip_rects(fill_rects)
    content_rects = [
        rect
        for rect in fill_rects
        if rect not in boundary_rects and not bool(rect.get("stroke"))
    ]
    candidates: List[Tuple[float, float, float, float]] = []
    for bbox in _merge_touching_fill_rects_by_color(content_rects):
        x0, top, x1, bottom = bbox
        width = x1 - x0
        if width < 120.0:
            continue

        top_strip = any(
            float(rect.get("x0", 0.0)) <= x0 + 4.0
            and float(rect.get("x1", 0.0)) >= x1 - 4.0
            and abs(float(rect.get("bottom", 0.0)) - top) <= 2.5
            for rect in boundary_rects
        )
        bottom_strip = any(
            float(rect.get("x0", 0.0)) <= x0 + 4.0
            and float(rect.get("x1", 0.0)) >= x1 - 4.0
            and abs(float(rect.get("top", 0.0)) - bottom) <= 2.5
            for rect in boundary_rects
        )
        if not top_strip or not bottom_strip:
            continue

        internal_verticals = _merge_numeric_positions(
            [
                float(edge["x0"])
                for edge in page.vertical_edges
                if bool(edge.get("stroke"))
                and float(edge.get("x0", 0.0)) > x0 + 2.0
                and float(edge.get("x0", 0.0)) < x1 - 2.0
                and float(edge.get("top", 0.0)) <= bottom
                and float(edge.get("bottom", 0.0)) >= top
            ],
            tolerance=1.0,
        )
        if internal_verticals:
            continue
        candidates.append(bbox)
    return candidates


def _extract_text_from_box_region(
    page: pdfplumber.page.PageObject,
    bbox: Tuple[float, float, float, float],
) -> str:
    # Box-like regions should collapse into one cell, preserving visual line breaks inside the box.
    filtered_page = _filter_page_for_extraction(page)
    words = (
        filtered_page
        .crop(bbox)
        .extract_words(x_tolerance=1.5, y_tolerance=2.0, keep_blank_chars=False)
        or []
    )
    grouped_lines: List[List[dict]] = []
    for word in sorted(words, key=lambda item: (float(item.get("top", 0.0)), float(item.get("x0", 0.0)))):
        cleaned = _repair_watermark_bleed(str(word.get("text") or "").strip())
        if not cleaned or _is_layout_artifact(cleaned):
            continue
        if not grouped_lines or abs(float(word.get("top", 0.0)) - float(grouped_lines[-1][0].get("top", 0.0))) > 2.5:
            grouped_lines.append([word])
            continue
        grouped_lines[-1].append(word)

    lines: List[str] = []
    for words_in_line in grouped_lines:
        ordered = sorted(words_in_line, key=lambda item: float(item.get("x0", 0.0)))
        text = " ".join(
            _repair_watermark_bleed(str(word.get("text") or "").strip())
            for word in ordered
            if _repair_watermark_bleed(str(word.get("text") or "").strip())
        ).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_overlap_ratio(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    intersection = _bbox_area((ix0, iy0, ix1, iy1))
    if intersection <= 0.0:
        return 0.0
    return intersection / max(min(_bbox_area(a), _bbox_area(b)), 1.0)


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


def _single_column_box_region_candidates(
    page: pdfplumber.page.PageObject,
) -> List[dict]:
    body_top, body_bottom = _detect_body_bounds(page, header_margin=90.0, footer_margin=40.0)
    fill_rects = [
        rect
        for rect in getattr(page, "rects", [])
        if bool(rect.get("fill"))
        and float(rect.get("bottom", 0.0)) > body_top
        and float(rect.get("top", 0.0)) < body_bottom
    ]
    merged_fill_rects = _dedupe_redundant_rectangles(_merge_touching_fill_rects_by_color(fill_rects, tolerance=1.0))
    boundary_rects = _thin_strip_rects(merged_fill_rects)
    content_rects = [
        rect
        for rect in merged_fill_rects
        if rect not in boundary_rects and not bool(rect.get("stroke"))
    ]

    candidates: List[dict] = []
    if not content_rects:
        return candidates

    for bbox in _merge_touching_fill_rects_by_color(content_rects, tolerance=1.0):
        x0, top, x1, bottom = bbox
        width = x1 - x0
        if width < 120.0:
            continue

        top_strip_ratio = _strip_coverage_ratio((x0, top, x1, bottom), boundary_rects, top, tolerance=2.8)
        bottom_strip_ratio = _strip_coverage_ratio((x0, top, x1, bottom), boundary_rects, bottom, tolerance=2.8)
        if top_strip_ratio < 0.45 and bottom_strip_ratio < 0.45:
            continue

        overlapping_content = [
            rect
            for rect in content_rects
            if not (
                float(rect.get("x1", 0.0)) < x0
                or float(rect.get("x0", 0.0)) > x1
                or float(rect.get("bottom", 0.0)) < top
                or float(rect.get("top", 0.0)) > bottom
            )
        ]
        content_colors: List[object] = []
        border_colors: List[object] = []
        for rect in overlapping_content:
            color = rect.get("non_stroking_color") if rect.get("non_stroking_color") is not None else rect.get("stroking_color")
            content_colors.append(color)
        for rect in boundary_rects:
            if not (
                float(rect.get("x1", 0.0)) < x0
                or float(rect.get("x0", 0.0)) > x1
                or float(rect.get("bottom", 0.0)) < top
                or float(rect.get("top", 0.0)) > bottom
            ):
                color = rect.get("non_stroking_color") if rect.get("non_stroking_color") is not None else rect.get("stroking_color")
                border_colors.append(color)

        has_non_white_border = any(not _is_nearly_white_color(color) for color in border_colors if color is not None)
        is_white_content = (
            bool(content_colors)
            and all(_is_nearly_white_color(color) for color in content_colors)
            and not has_non_white_border
        )

        internal_verticals = _merge_numeric_positions(
            [
                float(edge["x0"])
                for edge in page.vertical_edges
                if bool(edge.get("stroke"))
                and float(edge.get("x0", 0.0)) > x0 + 2.0
                and float(edge.get("x0", 0.0)) < x1 - 2.0
                and float(edge.get("top", 0.0)) <= bottom
                and float(edge.get("bottom", 0.0)) >= top
            ],
            tolerance=1.0,
        )
        if internal_verticals:
            continue

        candidates.append({
            "bbox": bbox,
            "is_white_content": is_white_content,
        })

    return candidates


def _single_column_box_regions(
    page: pdfplumber.page.PageObject,
) -> List[Tuple[float, float, float, float]]:
    # Keep public behavior for existing callers that only need candidate bboxes.
    return [entry["bbox"] for entry in _single_column_box_region_candidates(page)]


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


def _select_note_anchor_for_bbox(
    page: pdfplumber.page.PageObject,
    bbox: Tuple[float, float, float, float],
    image_regions: Sequence[Tuple[float, float, float, float]] | None = None,
) -> Tuple[float, float, float, float] | None:
    image_regions = list(image_regions or _candidate_image_regions_for_notes(page))
    if not image_regions:
        return None

    note_y0, note_y1 = bbox[1], bbox[3]
    note_x0, note_x1 = bbox[0], bbox[2]
    note_center_y = (note_y0 + note_y1) / 2.0
    best_anchor: Tuple[float, float, float, float] | None = None
    best_score = -1.0

    for region in image_regions:
        image_x0, image_top, image_x1, image_bottom = region
        x_overlap = _bbox_x_overlap_ratio(region, bbox)
        if x_overlap < 0.10:
            if image_x1 <= note_x0:
                gap = note_x0 - image_x1
            elif image_x0 >= note_x1:
                gap = image_x0 - note_x1
            else:
                gap = 0.0
            if gap > 24.0:
                continue
            x_overlap = max(0.1, 0.5 - (gap / 48.0))

        if x_overlap < 0.10:
            continue
        if image_bottom < note_y0 - 24.0 or image_top > note_y1 + 24.0:
            continue

        image_center_y = (image_top + image_bottom) / 2.0
        score = x_overlap * 100.0 - abs(note_center_y - image_center_y)
        if score > best_score:
            best_score = score
            best_anchor = region

    return best_anchor


def _candidate_note_band_for_bbox(
    page: pdfplumber.page.PageObject,
    bbox: Tuple[float, float, float, float],
) -> Tuple[float, float] | None:
    # Extend a note candidate from anchor image to nearest horizontal separators.
    image_regions = _candidate_image_regions_for_notes(page)
    if not image_regions:
        return None

    anchor = _select_note_anchor_for_bbox(page, bbox, image_regions=image_regions)
    if anchor is None:
        return None

    anchor_x0, anchor_top, anchor_x1, anchor_bottom = anchor
    anchor_area = (anchor_x0, anchor_top, anchor_x1, anchor_bottom)
    separator_lines = _horizontal_separator_lines(page)
    top_boundary = float("-inf")
    bottom_boundary = float("inf")

    for line_x0, line_y, line_x1, _line_bottom in separator_lines:
        if _bbox_x_overlap_ratio(anchor_area, (line_x0, line_y, line_x1, line_y)) < 0.35:
            continue
        if line_y <= anchor_top:
            top_boundary = max(top_boundary, line_y)
        else:
            bottom_boundary = min(bottom_boundary, line_y)

    if top_boundary == float("-inf"):
        top_boundary = anchor_top
    if bottom_boundary == float("inf"):
        return (anchor_top, anchor_bottom)
    return (max(0.0, top_boundary), min(page.height, bottom_boundary))


def _single_column_boxes_share_index(
    a_bbox: Tuple[float, float, float, float],
    b_bbox: Tuple[float, float, float, float],
) -> bool:
    a_x0, a_y0, a_x1, a_y1 = a_bbox
    b_x0, b_y0, b_x1, b_y1 = b_bbox
    if _bbox_x_overlap_ratio(a_bbox, b_bbox) < 0.65:
        return False

    a_width = a_x1 - a_x0
    b_width = b_x1 - b_x0
    return abs(a_width - b_width) <= 6.0 or abs(a_x0 - b_x0) <= 3.0




def _note_bands_are_adjacent_or_overlapping(
    a_band: tuple[float, float] | None,
    b_band: tuple[float, float] | None,
    *,
    gap_tolerance: float = 2.0,
) -> bool:
    if a_band is None or b_band is None:
        return False

    a_top, a_bottom = a_band
    b_top, b_bottom = b_band
    if a_top > a_bottom:
        a_top, a_bottom = a_bottom, a_top
    if b_top > b_bottom:
        b_top, b_bottom = b_bottom, b_top

    return max(a_top, b_top) <= min(a_bottom, b_bottom) + gap_tolerance


def _collect_single_column_candidates_for_notes(
    page: pdfplumber.page.PageObject,
) -> List[dict]:
    # Build reusable merged single-column box candidates once and share across table/note steps.
    candidate_rows: List[dict] = []
    image_regions = _candidate_image_regions_for_notes(page)
    for entry in _single_column_box_region_candidates(page):
        rows = [[_extract_text_from_box_region(page, entry["bbox"])]]
        if not rows[0][0]:
            continue
        raw_bbox = entry["bbox"]
        is_note_like = _looks_like_single_column_note(rows=rows, page=page, bbox=entry["bbox"])
        note_anchor = _select_note_anchor_for_bbox(page, raw_bbox, image_regions=image_regions)
        note_band = _candidate_note_band_for_bbox(page, raw_bbox) if note_anchor is not None else None
        if is_note_like and note_band is not None:
            raw_bbox = (
                entry["bbox"][0],
                min(entry["bbox"][1], note_band[0]),
                entry["bbox"][2],
                max(entry["bbox"][3], note_band[1]),
            )

        candidate_rows.append(
            {
                "bbox": raw_bbox,
                "raw_bbox": entry["bbox"],
                "rows": rows,
                "is_white_content": bool(entry.get("is_white_content")),
                "is_note_like": is_note_like,
                "note_anchor": (
                    tuple(round(value, 2) for value in note_anchor)
                    if note_anchor is not None
                    else None
                ),
                "note_band": note_band,
            }
        )

    merged_candidates: List[dict] = []
    for candidate in sorted(candidate_rows, key=lambda item: float(item["bbox"][1])):
        if candidate["is_white_content"]:
            merged_candidates.append(candidate)
            continue

        merged_into_existing = False
        for existing in merged_candidates:
            if existing["is_white_content"]:
                continue
            if not _single_column_boxes_share_index(existing["bbox"], candidate["bbox"]):
                continue
            # Merge vertically adjacent or separator-defined fragments that are likely the same logical box.
            same_anchor = (
                existing.get("note_anchor") is not None
                and existing["note_anchor"] == candidate.get("note_anchor")
            )
            same_band = _note_bands_are_adjacent_or_overlapping(
                existing.get("note_band"),
                candidate.get("note_band"),
            )
            if same_anchor or same_band or _bbox_overlap_ratio(existing["bbox"], candidate["bbox"]) > 0.0:
                merged_rows, merged_bbox = _merge_single_column_fragment_rows(
                    existing["bbox"],
                    existing["rows"],
                    candidate["bbox"],
                    candidate["rows"],
                )
                existing["rows"] = merged_rows
                existing["bbox"] = merged_bbox
                existing["is_note_like"] = _looks_like_single_column_note(
                    rows=merged_rows,
                    page=page,
                    bbox=merged_bbox,
                )
                merged_into_existing = True
                break

        if not merged_into_existing:
            merged_candidates.append(candidate)

    return merged_candidates


def _split_crop_bbox_by_excluded_bands(
    crop_bbox: Tuple[float, float, float, float],
    excluded_bands: Sequence[Tuple[float, float, float, float]],
    *,
    min_x_overlap_ratio: float = 0.20,
    y_merge_tolerance: float = 1.0,
    min_region_height: float = 3.5,
) -> List[Tuple[float, float, float, float]]:
    # Remove vertical note-like bands from a table crop to avoid parsing note content as table rows.
    x0, y0, x1, y1 = crop_bbox
    if not excluded_bands:
        return [crop_bbox]

    intervals: List[Tuple[float, float]] = []
    for band_x0, band_x1, band_top, band_bottom in excluded_bands:
        if _bbox_x_overlap_ratio((band_x0, band_top, band_x1, band_bottom), crop_bbox) < min_x_overlap_ratio:
            continue
        top = max(y0, band_top)
        bottom = min(y1, band_bottom)
        if bottom - top <= 0.0:
            continue
        intervals.append((top, bottom))

    if not intervals:
        return [crop_bbox]

    intervals.sort()
    merged: List[Tuple[float, float]] = []
    for top, bottom in intervals:
        if not merged or top > merged[-1][1] + y_merge_tolerance:
            merged.append((top, bottom))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], bottom))

    pieces: List[Tuple[float, float, float, float]] = []
    cursor = y0
    for top, bottom in merged:
        if top - cursor >= min_region_height:
            pieces.append((x0, cursor, x1, top))
        cursor = max(cursor, bottom)

    if y1 - cursor >= min_region_height:
        pieces.append((x0, cursor, x1, y1))

    return pieces

def _merge_single_column_fragment_rows(
    top_bbox: Tuple[float, float, float, float],
    top_rows: List[List[str]],
    bottom_bbox: Tuple[float, float, float, float],
    bottom_rows: List[List[str]],
) -> Tuple[List[List[str]], Tuple[float, float, float, float]]:
    # Preserve source order by vertical location and avoid accidental duplicated prose rows.
    if top_bbox[1] <= bottom_bbox[1]:
        ordered_rows = [
            [row[:] for row in top_rows],
            [row[:] for row in bottom_rows],
        ]
    else:
        ordered_rows = [
            [row[:] for row in bottom_rows],
            [row[:] for row in top_rows],
        ]

    merged_rows: List[List[str]] = []
    for row in (ordered_rows[0] + ordered_rows[1]):
        if not row:
            continue
        normalized = _first_non_empty_cell_value(row)
        if not normalized:
            continue
        if merged_rows and _first_non_empty_cell_value(merged_rows[-1]) == normalized:
            continue
        merged_rows.append(row)

    merged_bbox = (
        min(top_bbox[0], bottom_bbox[0]),
        min(top_bbox[1], bottom_bbox[1]),
        max(top_bbox[2], bottom_bbox[2]),
        max(top_bbox[3], bottom_bbox[3]),
    )
    return merged_rows, merged_bbox


def _extract_tables_from_crop(
    page: pdfplumber.page.PageObject,
    crop_bbox: Tuple[float, float, float, float],
    *,
    fallback_to_text_rows: bool = False,
) -> List[TableChunk]:
    # Crop-level extraction gives table_settings a tighter region and improves recovery of border-light tables.
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
        # Try line-driven extraction first, then a text-assisted fallback inside the same crop.
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

    if fallback_to_text_rows:
        line_rows = _compact_fallback_rows(_extract_region_line_rows(page, crop_bbox))
        if line_rows:
            return [(line_rows, crop_bbox)]

    return []


def _extract_tables(
    page: pdfplumber.page.PageObject,
    force_table: bool = False,
) -> List[TableChunk]:
    # Region-based extraction is preferred because full-page fallback tends to over-merge adjacent content.
    page = _filter_page_for_extraction(page)
    seen_keys = set()
    merged: List[TableChunk] = []
    candidate_rows = _collect_single_column_candidates_for_notes(page)
    table_regions = _table_regions(page)
    note_exclusion_bands: List[Tuple[float, float, float, float]] = []
    for candidate in candidate_rows:
        if candidate["is_white_content"] or not candidate.get("is_note_like", False):
            continue
        band_top, band_bottom = (
            candidate.get("note_band")
            if candidate.get("note_band") is not None
            else (candidate["bbox"][1], candidate["bbox"][3])
        )
        note_exclusion_bands.append(
            (
                candidate["bbox"][0],
                candidate["bbox"][2],
                band_top,
                band_bottom,
            )
        )

    for x0, x1, lines in table_regions:
        y0 = min(edge["top"] for edge in lines) - 2
        y1 = max(edge["top"] for edge in lines) + 2
        crop_bbox = (max(0.0, x0), max(0.0, y0), min(page.width, x1), min(page.height, y1))
        split_regions = _split_crop_bbox_by_excluded_bands(crop_bbox, note_exclusion_bands)
        for split_bbox in split_regions:
            for table, crop_box in _extract_tables_from_crop(
                page, split_bbox, fallback_to_text_rows=False
            ):
                table = _normalize_extracted_table(table)
                rows_key = tuple(tuple(row) for row in table)
                bbox_key = tuple(round(v, 2) for v in crop_box)
                key = (rows_key, bbox_key)
                if key not in seen_keys:
                    seen_keys.add(key)
                    merged.append((table, crop_box))

    if not table_regions:
        # Some documents render callout-like notes as filled boxes without explicit borders.
        for entry in _single_column_box_region_candidates(page):
            x0, y0, x1, y1 = entry["bbox"]
            crop_box = (x0, y0, x1, y1)
            for table, _crop_box in _extract_tables_from_crop(
                page,
                crop_box,
                fallback_to_text_rows=True,
            ):
                rows_key = tuple(tuple(row) for row in table)
                key = (rows_key, tuple(round(v, 2) for v in _crop_box))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged.append((table, _crop_box))

    merged_candidates = candidate_rows
    if table_regions:
        removed_indices: set[int] = set()
        for candidate in merged_candidates:
            if candidate["is_white_content"] or not candidate.get("is_note_like", False):
                continue
            for idx, (rows, table_bbox) in enumerate(merged):
                if idx in removed_indices:
                    continue
                if not _looks_like_single_column_note(rows=rows, page=page, bbox=table_bbox):
                    continue
                if not _single_column_boxes_share_index(table_bbox, candidate["bbox"]):
                    continue
                candidate_overlap_bbox = (
                    candidate["raw_bbox"]
                    if candidate.get("note_band") is not None
                    else candidate["bbox"]
                )
                if _bbox_overlap_ratio(table_bbox, candidate_overlap_bbox) < 0.8:
                    continue
                removed_indices.add(idx)

        if removed_indices:
            merged = [entry for idx, entry in enumerate(merged) if idx not in removed_indices]

    # Single-column callout-like boxes can represent prose notes even when table geometry exists.
    # Treat note-like non-overlapping box candidates as standalone notes so they are rendered
    # as note references instead of being silently dropped from output.
    if merged_candidates:
        merged_table_bboxes = [table_bbox for _rows, table_bbox in merged]
        for candidate in merged_candidates:
            if candidate["is_white_content"] or not candidate.get("is_note_like", False):
                continue

            candidate_bbox = (
                candidate["raw_bbox"]
                if candidate.get("note_band") is not None
                else candidate["bbox"]
            )
            overlaps_with_table = any(
                _bbox_overlap_ratio(candidate_bbox, table_bbox) >= 0.25
                for table_bbox in merged_table_bboxes
            )
            if overlaps_with_table and candidate.get("note_band") is None:
                continue

            merged.append((candidate["rows"], candidate_bbox))

    if merged or not force_table:
        return merged

    # The caller can opt into a more aggressive page-wide fallback when geometric table regions are absent.
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
    # Convert normalized row data into markdown table text used by the final artifacts.
    if not rows:
        return ""

    note_like = _looks_like_single_column_note(rows)
    if note_like:
        header = [f"Column {idx}" for idx in range(1, len(rows[0]) + 1)]
        body = rows
    else:
        header_row_count = _header_row_count(rows)
        header_rows = rows[:header_row_count] if header_row_count else rows[:1]
        header = _collapse_header_rows(header_rows)
        body = rows[header_row_count:] if header_row_count else rows[1:]
        if not body:
            body = rows
            header = [f"Column {idx}" for idx in range(1, len(rows[0]) + 1)]

    header_line = "| " + " | ".join(_format_header_markdown_cell(cell or f"Column {idx + 1}") for idx, cell in enumerate(header)) + " |"
    divider_line = "| " + " | ".join("---" for _ in header) + " |"
    body_lines = []
    for row in body:
        padded_row = list(row) + [""] * max(0, len(header) - len(row))
        body_lines.append("| " + " | ".join(_format_markdown_cell(str(value or "")) for value in padded_row) + " |")
    return "\n".join([header_line, divider_line, *body_lines])


def _merge_split_rows(rows: TableRows) -> TableRows:
    # Post-process rows that were extracted as separate fragments even though they belong to the same logical row.
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
    # Table numbering is derived at append time so merged cross-page tables keep one output block.
    merged_rows = _merge_split_rows(table_rows)
    collapsed_rows = _collapse_structural_triplet_columns(merged_rows)
    table_text = _table_text_from_rows(collapsed_rows)
    if table_text:
        output_tables.append(f"### Page {page_no} table {table_no}\n{table_text}")
