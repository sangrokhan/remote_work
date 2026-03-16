from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

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
    c.drawString(36, height - 28, "Graph PDF Demo Header: sample source")
    c.drawString(36, 22, f"Graph PDF Demo Footer / Page {page_no}")


def _draw_watermark(c: canvas.Canvas, width: float, height: float) -> None:
    c.saveState()
    c.setFillColor(colors.grey)
    c.setFillAlpha(0.12)
    c.setFont("Helvetica-Bold", 44)
    c.translate(width / 2, height * 0.78)
    c.rotate(30)
    c.drawCentredString(0, 0, "CONFIDENTIAL")
    c.restoreState()


def _draw_body_text(c: canvas.Canvas, width: float, start_y: float, lines: Sequence[str]) -> None:
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 11)
    x = 36
    y = start_y
    for line in lines:
        c.drawString(x, y, line)
        y -= 16


def _draw_table(
    c: canvas.Canvas,
    x0: float,
    y0: float,
    rows: Sequence[Tuple[str, str, str]],
    row_height: float = 22,
    column_positions: Sequence[int] = (140, 260),
) -> None:
    """
    Draw a table that has top/bottom/inner lines but no outer vertical borders.

    column_positions are x-offsets from x0 for inner vertical lines only.
    """
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.8)

    row_count = len(rows) + 1
    table_width = column_positions[-1] + 120
    total_height = row_count * row_height

    # Top, middle and bottom horizontal lines.
    for row in range(row_count + 1):
        y = y0 - row * row_height
        c.line(x0, y, x0 + table_width, y)

    # Inner vertical lines only; no line at x0 and x0+table_width
    for col_x in column_positions:
        c.line(x0 + col_x, y0, x0 + col_x, y0 - total_height)

    c.setFont("Helvetica-Bold", 9)
    c.drawString(x0 + 6, y0 - 16, "Item")
    c.drawString(x0 + column_positions[0] + 6, y0 - 16, "Qty")
    c.drawString(x0 + column_positions[1] + 6, y0 - 16, "Price")

    c.setFont("Helvetica", 9)
    y = y0 - row_height - 14
    for row in rows:
        c.drawString(x0 + 6, y, row[0])
        c.drawString(x0 + column_positions[0] + 6, y, row[1])
        c.drawString(x0 + column_positions[1] + 6, y, row[2])
        y -= row_height


def create_demo_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter

    # Page 1
    _draw_header_and_footer(c, 1, width, height)
    _draw_watermark(c, width, height)

    left_rows = [
        ("Widget", "12", "$120"),
        ("Keyboard", "8", "$32"),
        ("Monitor", "5", "$260"),
        ("Mouse", "18", "$8"),
    ]

    right_rows = [
        ("Alpha", "OK", "DONE"),
        ("Beta", "WARN", "REVIEW"),
        ("Gamma", "FAIL", "PENDING"),
    ]

    _draw_body_text(
        c,
        width,
        start_y=height - 70,
        lines=[
            "PDF extraction sample for body cleanup and structured table parsing.",
            "This page contains header, footer, watermark, and tables near the left and right edges.",
            "The tables are drawn with top/bottom and inner lines only; no outer vertical lines.",
            "Body text here is cleaned for ingestion by GraphRAG or similar index pipeline.",
        ],
    )

    _draw_table(c, x0=32, y0=320, rows=left_rows, column_positions=(90, 180))
    _draw_table(c, x0=340, y0=320, rows=right_rows, column_positions=(70, 130))

    c.showPage()

    # Page 2
    _draw_header_and_footer(c, 2, width, height)
    _draw_watermark(c, width, height)

    _draw_body_text(
        c,
        width,
        start_y=height - 90,
        lines=[
            "Second page only validates body text extraction without embedded tables.",
            "Even without table content, body text should stay while removing header/footer/watermark.",
            "The extracted output should be suitable for text chunking into vectors or graph nodes.",
        ],
    )

    c.save()
