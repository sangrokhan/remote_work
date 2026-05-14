import io
from typing import Generator
from docx import Document
from docx.oxml.ns import qn
from core.models import ParagraphElement, TableElement, ImageElement, Run

Element = ParagraphElement | TableElement | ImageElement


def _is_page_break_para(para) -> bool:
    for run in para.runs:
        for br in run._r.findall(qn("w:br")):
            if br.get(qn("w:type")) == "page":
                return True
    return False


def _para_runs(para) -> list[Run]:
    runs = []
    for r in para.runs:
        size = None
        if r.font.size is not None:
            size = r.font.size.pt
        runs.append(Run(text=r.text, font_size=size, bold=bool(r.bold)))
    return runs


def _table_rows(tbl) -> list[list[str]]:
    return [[cell.text for cell in row.cells] for row in tbl.rows]


def stream_elements(docx_data: bytes) -> Generator[Element, None, None]:
    doc = Document(io.BytesIO(docx_data))
    body = doc.element.body
    page_approx = 1
    last_was_page_break = False

    for child in body.iterchildren():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if tag == "p":
            from docx.text.paragraph import Paragraph
            para = Paragraph(child, doc)
            is_pb = _is_page_break_para(para)
            if is_pb:
                page_approx += 1
            runs = _para_runs(para)
            elem = ParagraphElement(
                text=para.text,
                style_name=para.style.name if para.style else "Normal",
                runs=runs,
                page_approx=page_approx,
                is_page_break=is_pb,
            )
            last_was_page_break = is_pb
            yield elem

        elif tag == "tbl":
            from docx.table import Table
            tbl = Table(child, doc)
            rows = _table_rows(tbl)
            col_count = max(len(r) for r in rows) if rows else 0
            elem = TableElement(
                rows=rows,
                col_count=col_count,
                page_approx=page_approx,
                preceded_by_page_break=last_was_page_break,
            )
            last_was_page_break = False
            yield elem

        else:
            last_was_page_break = False
