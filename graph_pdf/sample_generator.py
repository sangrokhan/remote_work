from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def _draw_header_and_footer(
    c: canvas.Canvas,
    page_no: int,
    width: float,
    height: float,
) -> None:
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 10)
    header_left_lines = (
        "Graph PDF Demo Header: sample source",
        "Prepared for table + text extraction tests",
        "Header checks: line 3 validates layout alignment",
    )
    header_right_lines = (
        f"Page {page_no} / 2",
        "Header line 2: extraction boundary checks",
        "Header line 3: left/right split text blocks",
    )
    c.drawString(36, height - 28, header_left_lines[0])
    c.drawString(36, height - 44, header_left_lines[1])
    c.drawString(36, height - 60, header_left_lines[2])
    c.drawRightString(width - 36, height - 28, header_right_lines[0])
    c.drawRightString(width - 36, height - 44, header_right_lines[1])
    c.drawRightString(width - 36, height - 60, header_right_lines[2])

    footer_left_lines = (
        "Graph PDF Demo Footer / Left",
        "Footer details: keep header/footer clean",
        "Footer note: ignore this for body extraction",
    )
    footer_right_lines = (
        "Footer line 1: generated data",
        f"Footer page marker: {page_no}",
        "Footer line 3: right-side metadata block",
    )
    c.drawString(36, 40, footer_left_lines[0])
    c.drawString(36, 28, footer_left_lines[1])
    c.drawString(36, 16, footer_left_lines[2])
    c.drawRightString(width - 36, 40, footer_right_lines[0])
    c.drawRightString(width - 36, 28, footer_right_lines[1])
    c.drawRightString(width - 36, 16, footer_right_lines[2])


def _draw_watermark(c: canvas.Canvas, width: float, height: float) -> None:
    c.saveState()
    c.setFillColor(colors.grey)
    c.setFillAlpha(0.12)
    c.setFont("Helvetica-Bold", 44)
    c.translate(width / 2, height * 0.78)
    c.rotate(30)
    c.drawCentredString(0, 0, "CONFIDENTIAL")
    c.restoreState()


LineItem = Union[str, Tuple[str, int]]
TableRow = Tuple[str, str, str]


def _draw_body_text(c: canvas.Canvas, width: float, start_y: float, lines: Sequence[LineItem]) -> None:
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 11)
    x = 36
    y = start_y
    for line in lines:
        text = line
        indent = 0
        if isinstance(line, tuple):
            text, indent = line
        c.drawString(x + indent, y, text)
        y -= 16


def _draw_wrapped_table_row(
    c: canvas.Canvas,
    x0: float,
    y: float,
    lines: Sequence[str],
    font_size: int,
    column_positions: Sequence[int],
    line_height: float,
) -> float:
    c.setFont("Helvetica", font_size)
    split_lines = [str(line).split("\n") if line else [""] for line in lines]
    max_lines = max((len(line) for line in split_lines), default=1)

    text_y = y - 12
    for line_idx in range(max_lines):
        c.drawString(
            x0 + 6,
            text_y,
            split_lines[0][line_idx] if line_idx < len(split_lines[0]) else "",
        )
        c.drawString(
            x0 + column_positions[0] + 6,
            text_y,
            split_lines[1][line_idx] if line_idx < len(split_lines[1]) else "",
        )
        c.drawString(
            x0 + column_positions[1] + 6,
            text_y,
            split_lines[2][line_idx] if line_idx < len(split_lines[2]) else "",
        )
        text_y -= line_height

    content_height = max_lines * line_height
    return max(line_height * 1.5, content_height)


def _draw_table(
    c: canvas.Canvas,
    x0: float,
    y0: float,
    header: Sequence[str],
    rows: Sequence[TableRow],
    row_height: float = 24,
    column_positions: Sequence[int] = (140, 260),
    table_width_tail: float = 130.0,
    include_header: bool = True,
    include_outer_vertical: bool = False,
) -> None:
    """
    Draw a table that has top/bottom/inner lines but no outer vertical borders.

    column_positions are x-offsets from x0 for inner vertical lines only.
    """
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.8)

    table_width = float(column_positions[-1]) + table_width_tail
    row_count = len(rows) + (1 if include_header else 0)
    total_height = row_count * row_height

    # Top, middle and bottom horizontal lines.
    for row in range(row_count + 1):
        y = y0 - row * row_height
        c.line(x0, y, x0 + table_width, y)

    # Vertical lines.
    for col_x in column_positions:
        c.line(x0 + col_x, y0, x0 + col_x, y0 - total_height)
    if include_outer_vertical:
        c.line(x0, y0, x0, y0 - total_height)
        c.line(x0 + table_width, y0, x0 + table_width, y0 - total_height)

    if include_header and header:
        c.setFont("Helvetica-Bold", 9)
        c.drawString(x0 + 6, y0 - 16, header[0])
        c.drawString(x0 + column_positions[0] + 6, y0 - 16, header[1] if len(header) > 1 else "")
        c.drawString(x0 + column_positions[1] + 6, y0 - 16, header[2] if len(header) > 2 else "")

    c.setFont("Helvetica", 9)
    y = y0 - row_height
    for row in rows:
        row_used = _draw_wrapped_table_row(
            c=c,
            x0=x0,
            y=y,
            lines=row,
            font_size=8,
            column_positions=column_positions,
            line_height=11,
        )
        y -= max(row_height, row_used)


def create_demo_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter

    # Page 1
    _draw_header_and_footer(c, 1, width, height)
    _draw_watermark(c, width, height)

    left_rows: Sequence[TableRow] = [
        (
            "Widget",
            "12\n- stock check\n- reorder ready",
            "$120",
        ),
        (
            "Keyboard",
            "8\n- mechanical\n- compact layout",
            "$32",
        ),
        (
            "Monitor",
            "5\n- 4k review\n- flicker test",
            "$260",
        ),
        (
            "Mouse",
            "18\n- 3 button\n- optical sensor",
            "$8",
        ),
    ]

    right_rows: Sequence[TableRow] = [
        (
            "Alpha",
            "OK",
            "DONE\nline 1\nline 2",
        ),
        (
            "Beta",
            "WARN",
            "REVIEW\nitemized\nneeds followup",
        ),
        (
            "Gamma",
            "FAIL",
            "PENDING\n- check logs\n- rerun checks",
        ),
    ]

    spanning_header: Sequence[str] = ("Stage", "Team", "Notes")
    spanning_rows_page1: Sequence[TableRow] = [
        (
            "Phase A",
            "Discovery",
            "Kickoff scope lock\n- gather baseline\n- define risks",
        ),
        (
            "",
            "Design",
            "UX skeleton review\n- navigation\n- component map",
        ),
        (
            "",
            "Frontend",
            "Prototype pass\n- mobile spec\n- accessibility path",
        ),
        (
            "Phase B",
            "Backend",
            "Core API design\n- auth contract\n- payload schema",
        ),
        (
            "",
            "Ops",
            "Runbook draft\n- infra checklist\n- alert thresholds",
        ),
        (
            "",
            "Security",
            "Threat model\n- token handling\n- permission matrix",
        ),
    ]

    spanning_rows_page2: Sequence[TableRow] = [
        (
            "",
            "QA",
            "Scenario matrix\n- smoke\n- negative cases",
        ),
        (
            "",
            "Release",
            "Rollout coordination\n- canary\n- monitoring",
        ),
        (
            "",
            "Docs",
            "Version notes\n- migration\n- rollout guide",
        ),
    ]

    _draw_body_text(
        c,
        width,
        start_y=height - 70,
        lines=[
            "Chapter 1: Deep Structure Verification",
            "This page intentionally includes 3+ line body paragraph to validate multi-line normalization",
            "for graph ingestion and chapter-aware chunking across paragraphs.",
            ("- 1st level bullet: layout and spacing checks", 12),
            ("  - nested detail: line 2 confirms indentation", 24),
            ("  - nested detail: line 3 confirms paragraph wrap and line breaks", 24),
            "The extraction should remove header/footer/watermark while preserving indented body content and table text.",
            ("This section has nested indentation:", 0),
            ("- level 1: body copy, one of many lines", 12),
            ("- level 2: second nested line", 24),
            ("- level 3: third line to test depth", 36),
            "Body text here is cleaned for ingestion by GraphRAG or similar index pipeline.",
        ],
    )

    # Two tables with different sizes on page 1.
    _draw_table(
        c,
        x0=32,
        y0=320,
        header=("Item", "Qty", "Price"),
        rows=left_rows,
        row_height=38,
        column_positions=(72, 160),
        table_width_tail=150.0,
    )

    _draw_table(
        c,
        x0=334,
        y0=325,
        header=("Group", "State", "Comment"),
        rows=right_rows,
        row_height=44,
        column_positions=(84, 160),
        table_width_tail=140.0,
    )

    # Spanning table starts on page 1 (header + first part).
    _draw_table(
        c,
        x0=32,
        y0=190,
        header=spanning_header,
        rows=spanning_rows_page1[:4],
        row_height=46,
        column_positions=(110, 240),
        table_width_tail=170.0,
        include_outer_vertical=True,
    )

    c.showPage()

    # Page 2
    _draw_header_and_footer(c, 2, width, height)
    _draw_watermark(c, width, height)

    _draw_body_text(
        c,
        width,
        start_y=height - 90,
        lines=[
            "Page 2 continues document structure and validates page-spanning table continuity.",
            "A spanning table row set is intentionally split by page boundary and should be merged as one logical table in extraction.",
            "Body line 1: continuation context confirms cleanup and ordering.",
            "Body line 2: verify indentation and multiple bullet styles are retained as raw text.",
            ("- 1st-level continuation bullet", 12),
            ("- 2nd-level continuation bullet", 24),
            "The output should still extract all rows and keep the merged column rows coherent.",
        ],
    )

    # Continuation of the spanning table starts from page 2 (no header on this page).
    _draw_table(
        c,
        x0=32,
        y0=730,
        header=(),
        rows=spanning_rows_page1[4:] + spanning_rows_page2,
        row_height=46,
        column_positions=(110, 240),
        table_width_tail=170.0,
        include_header=False,
        include_outer_vertical=True,
    )

    # A compact table near bottom keeps another size variant.
    _draw_table(
        c,
        x0=360,
        y0=290,
        header=("Area", "Status", "Action"),
        rows=[
            (
                "Docs",
                "READY",
                "Finalize\n- sample\n- archive",
            ),
            (
                "QA",
                "TODO",
                "Confirm\n- edge case\n- fallback",
            ),
        ],
        row_height=40,
        column_positions=(90, 170),
        table_width_tail=110.0,
    )

    c.save()
