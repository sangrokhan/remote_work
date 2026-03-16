from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from sample_fixture import load_demo_fixture

LineItem = Tuple[str, int]
TableRow = Tuple[str, str, str]

def _fixture_tables() -> Dict[str, Tuple[Tuple[str, str, str], Tuple[TableRow, ...]]]:
    fixture = load_demo_fixture()
    return {
        table["id"]: (
            tuple(table["columns"]),
            tuple(tuple(str(cell) for cell in row) for row in table["rows"]),
        )
        for table in fixture["tables"]
    }


def get_demo_tables() -> Dict[str, Tuple[Tuple[str, str, str], Tuple[TableRow, ...]]]:
    return _fixture_tables()


def get_demo_text_lines() -> Tuple[str, ...]:
    body = load_demo_fixture()["body"]
    return tuple(body["intro"] + body["after_item_table"] + body["after_stage_table"] + body["footer_lines"])


def _fixture_layout() -> dict:
    fixture = load_demo_fixture()
    return {
        "watermark_text": fixture["watermark_text"],
        "header_left": tuple(fixture["header_left"]),
        "header_right": tuple(fixture["header_right"]),
        "footer_left": tuple(fixture["footer_left"]),
        "footer_right": tuple(fixture["footer_right"]),
        "body": fixture["body"],
        "tables": _fixture_tables(),
    }


def _split_cell_lines(text: str) -> List[str]:
    lines = str(text or "").split("\n")
    if not lines:
        return [""]
    return [line.strip() for line in lines]


def _wrap_visual_line(text: str, max_width: float, font_name: str, font_size: float) -> List[str]:
    words = str(text or "").split()
    if not words:
        return [""]

    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
    lines.append(current)
    return lines


def _layout_cell_lines(text: str, max_width: float, font_name: str, font_size: float) -> List[str]:
    rendered: List[str] = []
    for logical_line in _split_cell_lines(text):
        if not logical_line:
            rendered.append("")
            continue
        rendered.extend(_wrap_visual_line(logical_line, max_width, font_name, font_size))
    return rendered or [""]


def _estimate_row_heights(rows: Sequence[TableRow], col_widths: Sequence[float], font_size: float) -> List[float]:
    heights: List[float] = []
    font_name = "Helvetica"
    for row in rows:
        wrapped_lengths = [
            _layout_cell_lines(cell, max(col_width - 8.0, 24.0), font_name, font_size)
            for cell, col_width in zip(row, col_widths)
        ]
        lines = max(len(lines_) for lines_ in wrapped_lengths)
        content_h = lines * (font_size + 2.0)
        heights.append(max(24.0, content_h + 4.0))
    return heights


class DemoPdfBuilder:
    def __init__(self, output_path: Path) -> None:
        self.fixture = _fixture_layout()
        self.path = output_path
        self.width, self.height = letter
        self.margin_x = 36
        self.body_top = self.height - 96
        self.body_bottom = 78
        self.left_footer_y = 44
        self.right_footer_y = 44
        self.cursor_y = self.body_top

        self.page_no = 0
        self.canvas = canvas.Canvas(str(output_path), pagesize=letter)
        self._start_new_page()

    def _start_new_page(self) -> None:
        if self.page_no > 0:
            self.canvas.showPage()

        self.page_no += 1
        self.canvas.setFillColor(colors.black)
        self.canvas.setFont("Helvetica", 10)

        header_left = self.fixture["header_left"]
        header_right = self.fixture["header_right"]

        self.canvas.drawString(self.margin_x, self.height - 26, header_left[0])
        self.canvas.drawString(self.margin_x, self.height - 42, header_left[1])
        self.canvas.drawRightString(
            self.width - self.margin_x,
            self.height - 26,
            header_right[0].format(page_no=self.page_no),
        )
        self.canvas.drawRightString(
            self.width - self.margin_x,
            self.height - 42,
            header_right[1],
        )

        footer_left = self.fixture["footer_left"]
        footer_right = self.fixture["footer_right"]
        self.canvas.drawString(self.margin_x, self.left_footer_y + 14, footer_left[0])
        self.canvas.drawString(self.margin_x, self.left_footer_y, footer_left[1])
        self.canvas.drawRightString(
            self.width - self.margin_x,
            self.right_footer_y + 14,
            footer_right[0],
        )
        self.canvas.drawRightString(
            self.width - self.margin_x,
            self.right_footer_y,
            footer_right[1].format(page_no=self.page_no),
        )

        self._draw_watermark(self.width / 2, self.height / 2, size=44)

        self.cursor_y = self.body_top

    def _ensure_space(self, height_needed: float) -> None:
        if self.cursor_y - height_needed >= self.body_bottom:
            return
        self._start_new_page()

    def _draw_watermark(self, x: float, y: float, size: int = 48) -> None:
        self.canvas.saveState()
        self.canvas.setFillColor(colors.grey)
        self.canvas.setFillAlpha(0.13)
        self.canvas.setFont("Helvetica-Bold", size)
        self.canvas.translate(x, y)
        self.canvas.rotate(55)
        self.canvas.drawCentredString(0, 0, self.fixture["watermark_text"])
        self.canvas.restoreState()

    def add_body_text(self, lines: Sequence[LineItem | str], line_height: float = 14.0) -> None:
        self.canvas.setFillColor(colors.black)
        self.canvas.setFont("Helvetica", 11)

        estimated_height = len(lines) * line_height
        self._ensure_space(estimated_height)

        for item in lines:
            indent = 0
            text = item
            if isinstance(item, tuple):
                text, indent = item
            self.canvas.drawString(self.margin_x + indent, self.cursor_y, text)
            self.cursor_y -= line_height

    def add_gap(self, height: float) -> None:
        self.cursor_y -= max(0.0, float(height))
        if self.cursor_y < self.body_bottom + 8:
            self._start_new_page()

    def _draw_table_block(
        self,
        header: Sequence[str],
        rows: Sequence[TableRow],
        header_font_size: float = 9.0,
        row_font_size: float = 8.0,
        include_header: bool = True,
        split_pages: bool = True,
        include_outer_vertical: bool = False,
        with_watermark: bool = False,
        merge_first_col: bool = False,
    ) -> None:
        if not rows:
            return

        body_width = self.width - 2 * self.margin_x
        if not body_width:
            return

        column_weights = [0.28, 0.2, 0.52]
        col_widths = [w * body_width for w in column_weights]
        col_x = [self.margin_x + col_widths[0], self.margin_x + col_widths[0] + col_widths[1]]
        header_height = 20.0
        line_h = row_font_size + 1.2

        prepared_rows = []
        current_group_id = -1
        for source_idx, row in enumerate(rows):
            if merge_first_col and str(row[0]).strip():
                current_group_id += 1

            cell_lines = [
                _layout_cell_lines(cell, max(col_width - 8.0, 24.0), "Helvetica", row_font_size)
                for cell, col_width in zip(row, col_widths)
            ]
            line_count = max(len(lines) for lines in cell_lines)
            prepared_rows.append(
                {
                    "source_idx": source_idx,
                    "cells": [str(cell or "") for cell in row],
                    "cell_lines": cell_lines,
                    "line_count": line_count,
                    "height": max(24.0, line_count * (row_font_size + 2.0) + 4.0),
                    "group_id": current_group_id if merge_first_col and current_group_id >= 0 else None,
                }
            )

        idx = 0
        while idx < len(prepared_rows):
            if self.cursor_y - self.body_bottom < header_height + 24.0:
                self._start_new_page()

            available = self.cursor_y - self.body_bottom
            used = header_height if include_header else 0.0
            chunk_rows = []
            chunk_heights = []
            chunk_group_ids = []

            while idx < len(prepared_rows):
                row_data = prepared_rows[idx]
                row_height = float(row_data["height"])
                remaining = available - used
                if row_height <= remaining:
                    chunk_rows.append(tuple(row_data["cells"]))
                    chunk_heights.append(row_height)
                    chunk_group_ids.append(row_data["group_id"])
                    used += row_height
                    idx += 1
                    continue

                if not split_pages:
                    break

                can_split_row = int(row_data["line_count"]) >= 6 or row_height >= 80.0
                if not can_split_row:
                    break

                max_lines_fit = int((remaining - 4.0) // (row_font_size + 2.0))
                if max_lines_fit <= 0:
                    break

                total_lines = max(int(row_data["line_count"]), 1)
                if max_lines_fit >= total_lines:
                    chunk_rows.append(tuple(row_data["cells"]))
                    chunk_heights.append(row_height)
                    chunk_group_ids.append(row_data["group_id"])
                    used += row_height
                    idx += 1
                    continue

                top_cells: List[str] = []
                bottom_cells: List[str] = []
                for col_idx, lines in enumerate(row_data["cell_lines"]):
                    top_part = lines[:max_lines_fit]
                    bottom_part = lines[max_lines_fit:]
                    if col_idx == 0 and merge_first_col:
                        bottom_cells.append("")
                    elif col_idx == 1:
                        bottom_cells.append(row_data["cells"][col_idx] if bottom_part else "")
                    else:
                        bottom_cells.append("\n".join(bottom_part).strip())
                    top_cells.append("\n".join(top_part).strip())

                chunk_rows.append(tuple(top_cells))
                chunk_heights.append(max(24.0, max_lines_fit * (row_font_size + 2.0) + 4.0))
                chunk_group_ids.append(row_data["group_id"])

                prepared_rows[idx] = {
                    **row_data,
                    "cells": bottom_cells,
                    "cell_lines": [
                        _layout_cell_lines(cell, max(col_width - 8.0, 24.0), "Helvetica", row_font_size)
                        for cell, col_width in zip(bottom_cells, col_widths)
                    ],
                }
                prepared_rows[idx]["line_count"] = max(len(lines) for lines in prepared_rows[idx]["cell_lines"])
                prepared_rows[idx]["height"] = max(
                    24.0,
                    prepared_rows[idx]["line_count"] * (row_font_size + 2.0) + 4.0,
                )
                break

            if not chunk_rows:
                self._start_new_page()
                continue

            merged_spans: List[Tuple[int, int]] = []
            if merge_first_col and chunk_group_ids:
                run_start = 0
                while run_start < len(chunk_group_ids):
                    gid = chunk_group_ids[run_start]
                    run_end = run_start
                    while run_end + 1 < len(chunk_group_ids) and chunk_group_ids[run_end + 1] == gid:
                        run_end += 1
                    if gid is not None:
                        prev_gid = chunk_group_ids[run_start - 1] if run_start > 0 else None
                        next_gid = chunk_group_ids[run_end + 1] if run_end + 1 < len(chunk_group_ids) else (
                            prepared_rows[idx]["group_id"] if idx < len(prepared_rows) else None
                        )
                        if run_end > run_start or prev_gid == gid or next_gid == gid:
                            merged_spans.append((run_start, run_end))
                    run_start = run_end + 1

            table_top = self.cursor_y
            table_bottom = table_top - used

            self._render_table(
                x=self.margin_x,
                y_top=table_top,
                col_x=col_x,
                col_widths=col_widths,
                body_width=body_width,
                header=header if include_header else (),
                rows=chunk_rows,
                row_heights=chunk_heights,
                include_outer_vertical=include_outer_vertical,
                include_header=include_header,
                header_font_size=header_font_size,
                row_font_size=row_font_size,
                merge_first_col=merge_first_col,
                merged_first_col_spans=merged_spans or None,
            )
            self.cursor_y = table_bottom
            if idx < len(prepared_rows):
                self._start_new_page()

    def _render_table(
        self,
        *,
        x: float,
        y_top: float,
        col_x: Sequence[float],
        col_widths: Sequence[float],
        body_width: float,
        header: Sequence[str],
        rows: Sequence[TableRow],
        row_heights: Sequence[float],
        include_outer_vertical: bool,
        include_header: bool,
        merge_first_col: bool,
        merged_first_col_spans: Sequence[Tuple[int, int]] | None,
        header_font_size: float,
        row_font_size: float,
    ) -> None:
        self.canvas.setStrokeColor(colors.black)
        self.canvas.setLineWidth(0.8)

        header_h = 20.0 if include_header else 0.0
        table_h = header_h + sum(row_heights)
        row_tops = [0.0]
        for row_h in row_heights:
            row_tops.append(row_tops[-1] + row_h)

        # Highlight merged first-column spans to make the visual grouping explicit.
        # A merged span starts with a non-empty first column and continues while the
        # first-column cell remains empty.
        if merge_first_col and rows:
            if merged_first_col_spans is not None:
                spans = list(merged_first_col_spans)
            else:
                spans = []
                i = 0
                while i < len(rows):
                    if str(rows[i][0]).strip():
                        start = i
                        end = i
                        j = i + 1
                        while j < len(rows) and not str(rows[j][0]).strip():
                            end = j
                            j += 1
                        if end > start:
                            spans.append((start, end))
                        i = j
                    else:
                        i += 1

            if spans:
                self.canvas.saveState()
                fill_color = colors.HexColor("#eceff6")
                border_color = colors.HexColor("#7c8799")
                self.canvas.setFillColor(fill_color)
                self.canvas.setStrokeColor(border_color)
                self.canvas.setLineWidth(0.7)
                body_top = y_top - header_h
                for start, end in spans:
                    y_span_top = body_top - row_tops[start]
                    y_span_bottom = body_top - row_tops[end + 1]
                    group_height = y_span_top - y_span_bottom

                    self.canvas.rect(
                        x,
                        y_span_bottom,
                        col_x[0] - x,
                        group_height,
                        fill=1,
                        stroke=0,
                    )

                    # Keep a clear boundary for merged spans to distinguish it from
                    # ordinary blank-cell rows in monochrome extraction passes.
                    self.canvas.setStrokeColor(border_color)
                    self.canvas.line(x, y_span_top, col_x[0], y_span_top)
                    self.canvas.line(x, y_span_bottom, col_x[0], y_span_bottom)
                    self.canvas.line(col_x[0], y_span_top, col_x[0], y_span_bottom)
                self.canvas.restoreState()

        y = y_top
        self.canvas.line(x, y_top, x + body_width, y_top)

        if include_header and header:
            self.canvas.setFont("Helvetica-Bold", header_font_size)
            self.canvas.drawString(x + 6, y_top - 14, header[0])
            self.canvas.drawString(col_x[0] + 6, y_top - 14, header[1] if len(header) > 1 else "")
            self.canvas.drawString(col_x[1] + 6, y_top - 14, header[2] if len(header) > 2 else "")
            y -= header_h

        # Horizontal lines (including header/body split and bottom line).
        table_start = y
        for idx, row_h in enumerate(row_heights):
            next_row_merges_first_col = False
            if idx + 1 < len(rows):
                if merged_first_col_spans is not None:
                    next_row_merges_first_col = any(start <= idx < end for start, end in merged_first_col_spans)
                else:
                    next_row = rows[idx + 1]
                    next_row_merges_first_col = bool(next_row) and not str(next_row[0]).strip()

            if next_row_merges_first_col:
                self.canvas.line(col_x[0], y, x + body_width, y)
            else:
                self.canvas.line(x, y, x + body_width, y)

            y -= row_h

        # If the first chunk starts a merged continuation row (empty first cell), do not
        # draw the top border across the merged first column.
        if not include_header and rows and not str(rows[0][0]).strip():
            self.canvas.line(x + col_widths[0], table_start, x + body_width, table_start)

        self.canvas.line(x, y, x + body_width, y)

        # Vertical lines.
        if include_outer_vertical:
            self.canvas.line(x, y_top, x, y)
            self.canvas.line(x + body_width, y_top, x + body_width, y)
        self.canvas.line(col_x[0], y_top, col_x[0], y)
        self.canvas.line(col_x[1], y_top, col_x[1], y)

        # Body text.
        self.canvas.setFont("Helvetica", row_font_size)
        line_h = row_font_size + 1.2
        row_cursor = y_top - header_h
        span_starts: dict[int, tuple[int, float, float]] = {}
        rows_in_spans: set[int] = set()

        if merge_first_col and merged_first_col_spans:
            body_top = y_top - header_h
            for start, end in merged_first_col_spans:
                span_top = body_top - row_tops[start]
                span_bottom = body_top - row_tops[end + 1]
                span_starts[start] = (end, span_top, span_bottom)
                for row_idx in range(start, end + 1):
                    rows_in_spans.add(row_idx)

        for row_idx, (row, row_h) in enumerate(zip(rows, row_heights)):
            wrap_texts = [
                _layout_cell_lines(cell, max(col_width - 8.0, 24.0), "Helvetica", row_font_size)
                for cell, col_width in zip(row, col_widths)
            ]
            max_line_count = max(len(t) for t in wrap_texts)
            row_baseline_top = row_cursor - 11

            draw_first_col_per_row = row_idx not in rows_in_spans
            for i in range(max_line_count):
                if draw_first_col_per_row:
                    self.canvas.drawString(
                        x + 4,
                        row_baseline_top - (i * line_h),
                        wrap_texts[0][i] if i < len(wrap_texts[0]) else "",
                    )
                self.canvas.drawString(col_x[0] + 4, row_baseline_top - (i * line_h), wrap_texts[1][i] if i < len(wrap_texts[1]) else "")
                self.canvas.drawString(col_x[1] + 4, row_baseline_top - (i * line_h), wrap_texts[2][i] if i < len(wrap_texts[2]) else "")

            if row_idx in span_starts and str(row[0]).strip():
                _end, span_top, span_bottom = span_starts[row_idx]
                first_col_lines = wrap_texts[0]
                span_height = span_top - span_bottom
                text_block_height = max(len(first_col_lines), 1) * line_h
                block_top = span_top - max((span_height - text_block_height) / 2.0, 8.0) - 4.0
                for i, line in enumerate(first_col_lines):
                    self.canvas.drawString(x + 4, block_top - (i * line_h), line)

            row_cursor -= row_h

    def save(self) -> None:
        self.canvas.save()


def create_demo_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    builder = DemoPdfBuilder(path)
    body = builder.fixture["body"]
    tables = builder.fixture["tables"]
    intro_lines = tuple(body["intro"])
    after_item_table = tuple(body["after_item_table"])
    after_stage_table = tuple(body["after_stage_table"])
    footer_lines = tuple(body["footer_lines"])

    builder.add_body_text(
        (
            *intro_lines[:3],
            ("  - nested detail: line 2 confirms indentation", 12),
            ("    - deeper detail: line 3 confirms paragraph wrap and line breaks", 24),
            *intro_lines[5:],
        )
    )

    builder._draw_table_block(
        header=tables["item"][0],
        rows=tables["item"][1],
        include_header=True,
        split_pages=False,
        include_outer_vertical=False,
        with_watermark=True,
    )
    builder.add_gap(24.0)

    builder.add_body_text(after_item_table)
    builder.add_gap(100.0)

    builder._draw_table_block(
        header=tables["stage"][0],
        rows=tables["stage"][1],
        include_header=True,
        split_pages=True,
        include_outer_vertical=False,
        with_watermark=False,
        merge_first_col=True,
    )
    builder.add_gap(24.0)

    builder.add_body_text(
        (
            *after_stage_table[:2],
            (after_stage_table[2], 12),
        )
    )

    builder._draw_table_block(
        header=tables["area"][0],
        rows=tables["area"][1],
        include_header=True,
        split_pages=False,
        include_outer_vertical=False,
        with_watermark=False,
    )
    builder.add_gap(24.0)

    builder.add_body_text(footer_lines)

    builder.save()
