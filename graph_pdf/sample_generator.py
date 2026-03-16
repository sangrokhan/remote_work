from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


WATERMARK_TEXT = "CONFIDENTIAL"
HEADER_LEFT = (
    "Graph PDF Demo Header: sample source",
    "Prepared for table + text extraction tests",
)
HEADER_RIGHT = (
    "Page {page_no}",
    "Header checks: layout boundary review",
)
FOOTER_LEFT = (
    "Graph PDF Demo Footer / Left",
    "Footer details: keep header/footer clean",
)
FOOTER_RIGHT = (
    "Footer line 1: generated data",
    "Footer page marker: {page_no}",
)

LineItem = Tuple[str, int]
TableRow = Tuple[str, str, str]

ITEM_ROWS: Tuple[TableRow, ...] = (
    (
        "Laptop\n- line 1",
        "12",
        "$120",
    ),
    (
        "Keyboard\n- line 1\n- line 2",
        "8",
        "$32",
    ),
    (
        "Monitor",
        "5\n- line A\n- line B",
        "$260",
    ),
    (
        "Mouse",
        "18\n- optical\n- silent click",
        "$8",
    ),
    (
        "Dock",
        "3",
        "$45\n- includes\n- 2 ports",
    ),
)

STAGE_ROWS: Tuple[TableRow, ...] = (
    (
        "Phase A",
        "Discovery",
        "Kickoff scope lock\n- gather baseline\n- define risks\n- align dependencies",
    ),
    (
        "",
        "Design",
        "UX skeleton review\n- navigation\n- component map\n- spacing audit",
    ),
    (
        "",
        "Frontend",
        "Prototype pass\n- mobile spec\n- accessibility path\n- responsive breakpoints",
    ),
    (
        "",
        "Backend",
        "Core API design\n- auth contract\n- payload schema\n- endpoint ownership",
    ),
    (
        "",
        "QA",
        "Quality validation\n- smoke\n- integration matrix\n- rollback checks",
    ),
    (
        "",
        "Docs",
        "Scenario matrix draft\n- migration\n- release notes\n- stakeholder sync",
    ),
    (
        "",
        "Release Notes",
        "Publish cadence plan\n- draft audience\n- review windows\n- archive distribution",
    ),
    (
        "Phase B",
        "Operations",
        "Runbook draft\n- infra checklist\n- alert thresholds\n- on-call routing",
    ),
    (
        "",
        "Security",
        "Threat model\n- token handling\n- permission matrix\n- incident runbook",
    ),
    (
        "",
        "Release",
        "Scenario matrix\n- smoke\n- negative cases\n- release windows",
    ),
    (
        "",
        "Platform",
        "Support playbooks\n- upgrade path\n- dependency freeze\n- fallback scripts",
    ),
    (
        "",
        "Observability",
        "Dashboard rollout\n- metric taxonomy\n- error alert policy\n- tracing contracts",
    ),
    (
        "",
        "Performance",
        "Scale and profiling pass\n- load profile\n- memory pressure\n- queue saturation targets",
    ),
    (
        "",
        "Compliance",
        "Policy sweep\n- controls list\n- retention rules\n- audit trails",
    ),
    (
        "Phase C",
        "Documentation",
        "Publish handoff pack\n- API docs\n- release playbook\n- migration path",
    ),
    (
        "",
        "Legal",
        "Terms and compliance checks\n- consent language\n- governance review\n- retention policy",
    ),
    (
        "",
        "Accessibility",
        "Review deep pass\n- contrast baseline\n- keyboard order\n- narration labels",
    ),
    (
        "",
        "Operations",
        "Post-launch tasks\n- monitor alerts\n- close checklist\n- confirm rollback route",
    ),
    (
        "",
        "Finance",
        "Invoice trail\n- line item reconciliation\n- grant tracking\n- budget burn alerts",
    ),
)

COMPACT_ROWS: Tuple[TableRow, ...] = (
    ("Docs", "READY", "Finalize\n- sample\n- archive"),
    ("QA", "TODO", "Confirm\n- edge case\n- fallback"),
    ("Ops", "OK", "Archive path\n- cleanup\n- index refresh"),
)

DEMO_TABLES: Dict[str, Tuple[Tuple[str, str, str], Tuple[TableRow, ...]]] = {
    "item": (("Item", "Qty", "Price"), ITEM_ROWS),
    "stage": (("Stage", "Team", "Notes"), STAGE_ROWS),
    "area": (("Area", "Status", "Action"), COMPACT_ROWS),
}

DEMO_BODY_LINES: Tuple[str, ...] = (
    "Chapter 1: Deep Structure Verification",
    "This section starts a body flow with multiple lines and clear indentation to test ordered extraction.",
    "- 1st level bullet: layout and spacing checks",
    "- nested detail: line 2 confirms indentation",
    "- deeper detail: line 3 confirms paragraph wrap and line breaks",
    "The extraction should remove header/footer and watermark while preserving indented body content.",
    "- level 1: body copy, one of many lines",
    "- level 2: second nested line",
    "- level 3: third line to test depth",
    "Body text here is cleaned for ingestion by GraphRAG or similar index pipelines.",
)

DEMO_AFTER_TABLE_LINES: Tuple[str, ...] = (
    "The first table must fit entirely inside body bounds and end before the footer region.",
    "After table, body lines must continue and never overlap following elements.",
)

DEMO_SPAN_TABLE_TAIL_LINES: Tuple[str, ...] = (
    "Page 2 continues the flow if table rows spill over from the first table page.",
    "This paragraph is intentionally after the spanning table so table continuation remains earlier on the next page.",
    "- 1st-level continuation bullet",
)

DEMO_FOOTER_LINES: Tuple[str, ...] = (
    "Appendix chapter line for verification:",
    "- 1st level appendix bullet",
    "- 2nd level appendix bullet",
    "Final lines ensure normal paragraph flow does not overlap completed tables.",
)


def get_demo_tables() -> Dict[str, Tuple[Tuple[str, str, str], Tuple[TableRow, ...]]]:
    return DEMO_TABLES


def get_demo_text_lines() -> Tuple[str, ...]:
    return (
        *DEMO_BODY_LINES,
        *DEMO_AFTER_TABLE_LINES,
        *DEMO_SPAN_TABLE_TAIL_LINES,
        *DEMO_FOOTER_LINES,
    )


def _split_cell_lines(text: str) -> List[str]:
    lines = str(text or "").split("\n")
    if not lines:
        return [""]
    return [line.strip() for line in lines]


def _estimate_row_heights(rows: Sequence[TableRow], col_widths: Sequence[float], font_size: float) -> List[float]:
    heights: List[float] = []
    for row in rows:
        wrapped_lengths = [_split_cell_lines(cell) for cell in row]
        lines = max(len(lines_) for lines_ in wrapped_lengths)
        content_h = lines * (font_size + 2.0)
        heights.append(max(24.0, content_h + 4.0))
    return heights


class DemoPdfBuilder:
    def __init__(self, output_path: Path) -> None:
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

        header_left = (
            *HEADER_LEFT,
        )
        header_right = (
            *HEADER_RIGHT,
        )

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

        footer_left = (
            *FOOTER_LEFT,
        )
        footer_right = (
            *FOOTER_RIGHT,
        )
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
        self.canvas.drawCentredString(0, 0, WATERMARK_TEXT)
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
    ) -> None:
        if not rows:
            return

        body_width = self.width - 2 * self.margin_x
        if not body_width:
            return

        column_weights = [0.28, 0.2, 0.52]
        col_widths = [w * body_width for w in column_weights]
        col_x = [self.margin_x + col_widths[0], self.margin_x + col_widths[0] + col_widths[1]]

        row_heights = _estimate_row_heights(
            rows=rows,
            col_widths=col_widths,
            font_size=row_font_size,
        )
        header_height = 20.0

        chunk_start = 0
        first_chunk = True

        while chunk_start < len(rows):
            first_chunk_rows = len(rows) - chunk_start
            available = self.cursor_y - self.body_bottom

            if available < 25:
                self._start_new_page()
                available = self.cursor_y - self.body_bottom

            rows_for_chunk: List[Tuple[int, float]] = []
            used = header_height if first_chunk and include_header else 0.0

            idx = chunk_start
            while idx < len(rows):
                row_h = row_heights[idx]
                if used + row_h <= available:
                    rows_for_chunk.append((idx, row_h))
                    used += row_h
                    idx += 1
                    continue

                if idx == chunk_start:
                    # force at least one row in a new page fragment
                    if first_chunk and include_header:
                        self._start_new_page()
                        available = self.cursor_y - self.body_bottom
                        first_chunk = False
                        used = 0.0
                        continue
                    break
                break

            if not rows_for_chunk:
                self._start_new_page()
                continue

            chunk_end = rows_for_chunk[-1][0] + 1
            chunk_rows = rows[chunk_start:chunk_end]
            chunk_heights = [rows_for_chunk[i][1] for i in range(len(chunk_rows))]

            table_top = self.cursor_y
            table_bottom = table_top - (used)

            if first_chunk and with_watermark:
                # Place watermark slightly above the table block to validate
                # watermark presence without corrupting cell extraction.
                self._draw_watermark(self.margin_x + body_width / 2, table_top + 18, size=44)

            self._render_table(
                x=self.margin_x,
                y_top=table_top,
                col_x=col_x,
                col_widths=col_widths,
                body_width=body_width,
                header=header if first_chunk and include_header else (),
                rows=chunk_rows,
                row_heights=chunk_heights,
                include_outer_vertical=include_outer_vertical,
                include_header=first_chunk and include_header,
                header_font_size=header_font_size,
                row_font_size=row_font_size,
            )

            self.cursor_y = table_bottom

            chunk_start = chunk_end
            if chunk_start < len(rows):
                self._start_new_page()
                first_chunk = False

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
        header_font_size: float,
        row_font_size: float,
    ) -> None:
        self.canvas.setStrokeColor(colors.black)
        self.canvas.setLineWidth(0.8)

        header_h = 20.0 if include_header else 0.0
        table_h = header_h + sum(row_heights)

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
                next_row = rows[idx + 1]
                next_row_merges_first_col = bool(next_row) and not str(next_row[0]).strip()

            if idx == 0 and not include_header and rows and str(rows[0][0]).strip():
                # Non-header continuation chunks keep first row text start as normal.
                next_row_merges_first_col = next_row_merges_first_col

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

        for row, row_h in zip(rows, row_heights):
            wrap_texts = [
                _split_cell_lines(cell)
                for cell in row
            ]
            max_line_count = max(len(t) for t in wrap_texts)
            row_baseline_top = row_cursor - 11

            for i in range(max_line_count):
                self.canvas.drawString(x + 4, row_baseline_top - (i * line_h), wrap_texts[0][i] if i < len(wrap_texts[0]) else "")
                self.canvas.drawString(col_x[0] + 4, row_baseline_top - (i * line_h), wrap_texts[1][i] if i < len(wrap_texts[1]) else "")
                self.canvas.drawString(col_x[1] + 4, row_baseline_top - (i * line_h), wrap_texts[2][i] if i < len(wrap_texts[2]) else "")

            row_cursor -= row_h

    def save(self) -> None:
        self.canvas.save()


def create_demo_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    builder = DemoPdfBuilder(path)

    builder.add_body_text(
        (
            *DEMO_BODY_LINES[:3],
            ("  - nested detail: line 2 confirms indentation", 12),
            ("    - deeper detail: line 3 confirms paragraph wrap and line breaks", 24),
            *DEMO_BODY_LINES[5:],
        )
    )

    builder._draw_table_block(
        header=("Item", "Qty", "Price"),
        rows=DEMO_TABLES["item"][1],
        include_header=True,
        split_pages=False,
        include_outer_vertical=False,
        with_watermark=True,
    )
    builder.add_gap(24.0)

    builder.add_body_text(
        DEMO_AFTER_TABLE_LINES
    )
    builder.add_gap(120.0)

    builder._draw_table_block(
        header=("Stage", "Team", "Notes"),
        rows=DEMO_TABLES["stage"][1],
        include_header=True,
        split_pages=True,
        include_outer_vertical=False,
        with_watermark=False,
    )
    builder.add_gap(24.0)

    builder.add_body_text(
        (
            *DEMO_SPAN_TABLE_TAIL_LINES[:2],
            (DEMO_SPAN_TABLE_TAIL_LINES[2], 12),
        )
    )

    builder._draw_table_block(
        header=("Area", "Status", "Action"),
        rows=DEMO_TABLES["area"][1],
        include_header=True,
        split_pages=False,
        include_outer_vertical=False,
        with_watermark=False,
    )
    builder.add_gap(24.0)

    builder.add_body_text(
        DEMO_FOOTER_LINES
    )

    builder.save()
