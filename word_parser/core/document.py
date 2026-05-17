import io
import logging
from typing import Generator
from docx import Document
from docx.oxml.ns import qn
from core.models import ParagraphElement, TableElement, ImageElement, Run

Element = ParagraphElement | TableElement | ImageElement

_A_BLIP = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
_R_EMBED = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
_V_IMAGEDATA = "{urn:schemas-microsoft-com:vml}imagedata"
_KNOWN_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/bmp",
    "image/tiff",
    "image/x-emf",
    "image/emf",
    "image/x-wmf",
    "image/wmf",
    "image/svg+xml",
    "image/webp",
}


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
    result = []
    for tr in tbl._tbl.tr_lst:
        row_data: list[str] = []
        for tc in tr.tc_lst:
            tc_pr = tc.find(qn("w:tcPr"))
            v_merge = tc_pr.find(qn("w:vMerge")) if tc_pr is not None else None
            grid_span_el = tc_pr.find(qn("w:gridSpan")) if tc_pr is not None else None
            grid_span = int(grid_span_el.get(qn("w:val"), 1)) if grid_span_el is not None else 1

            if v_merge is not None and v_merge.get(qn("w:val")) != "restart":
                cell_text = "^"
            else:
                cell_text = "".join(t.text or "" for t in tc.iter(qn("w:t")))
            row_data.append(cell_text)
            for _ in range(grid_span - 1):
                row_data.append("")
        result.append(row_data)
    return result


def _extract_drawing_image(
    para, doc, page_approx: int, logger: logging.Logger | None = None
) -> ImageElement | None:
    # DrawingML path: w:drawing → a:blip r:embed
    drawing = para._p.find(".//" + qn("w:drawing"))
    if drawing is not None:
        blip = drawing.find(".//" + _A_BLIP)
        if blip is None:
            if logger:
                logger.debug("[document] w:drawing found but no a:blip — skipping paragraph image")
            return None
        r_id = blip.get(_R_EMBED)
        if r_id is None:
            if logger:
                logger.debug("[document] a:blip found but no r:embed attribute — skipping paragraph image")
            return None
        try:
            rel = doc.part.rels[r_id]
            content_type = rel.target_part.content_type
            if content_type not in _KNOWN_IMAGE_TYPES:
                if logger:
                    logger.warning(
                        f"[document] Unrecognized image content type: {content_type!r} (rId={r_id}) — skipped"
                    )
                return None
            return ImageElement(
                relationship_id=r_id,
                content_type=content_type,
                data=rel.target_part.blob,
                page_approx=page_approx,
            )
        except (KeyError, AttributeError) as e:
            if logger:
                logger.debug(f"[document] Failed to resolve image relationship {r_id!r}: {e}")
            return None

    # VML path: w:pict → v:imagedata r:id
    pict = para._p.find(".//" + qn("w:pict"))
    if pict is not None:
        imagedata = pict.find(".//" + _V_IMAGEDATA)
        if imagedata is None:
            if logger:
                logger.debug("[document] w:pict found but no v:imagedata — skipping paragraph image")
            return None
        r_id = imagedata.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        if r_id is None:
            if logger:
                logger.debug("[document] v:imagedata found but no r:id attribute — skipping paragraph image")
            return None
        try:
            rel = doc.part.rels[r_id]
            content_type = rel.target_part.content_type
            if content_type not in _KNOWN_IMAGE_TYPES:
                if logger:
                    logger.warning(
                        f"[document] Unrecognized VML image content type: {content_type!r} (rId={r_id}) — skipped"
                    )
                return None
            return ImageElement(
                relationship_id=r_id,
                content_type=content_type,
                data=rel.target_part.blob,
                page_approx=page_approx,
            )
        except (KeyError, AttributeError) as e:
            if logger:
                logger.debug(f"[document] Failed to resolve VML image relationship {r_id!r}: {e}")
            return None

    return None


def attach_captions(elements: list[Element]) -> list[Element]:
    result: list[Element] = []
    i = 0
    while i < len(elements):
        elem = elements[i]
        if (
            isinstance(elem, ImageElement)
            and i + 1 < len(elements)
            and isinstance(elements[i + 1], ParagraphElement)
            and "caption" in elements[i + 1].style_name.lower()
        ):
            elem.caption = elements[i + 1].text
            result.append(elem)
            i += 2
        else:
            result.append(elem)
            i += 1
    return result


def stream_elements(
    docx_data: bytes,
    logger: logging.Logger | None = None,
) -> Generator[Element, None, None]:
    doc = Document(io.BytesIO(docx_data))
    body = doc.element.body
    state = {"page_approx": 1, "last_was_page_break": False}

    def _iter_children(parent):
        for child in parent.iterchildren():
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

            if tag == "sdt":
                # Recurse into content control — paragraphs/tables live in w:sdtContent
                sdt_content = child.find(qn("w:sdtContent"))
                if sdt_content is not None:
                    yield from _iter_children(sdt_content)
                continue

            if tag == "p":
                from docx.text.paragraph import Paragraph
                para = Paragraph(child, doc)
                is_pb = _is_page_break_para(para)
                if is_pb:
                    state["page_approx"] += 1
                    if logger:
                        logger.info(f"[document] Page {state['page_approx']} started")
                img = _extract_drawing_image(para, doc, state["page_approx"], logger=logger)
                if img is not None:
                    state["last_was_page_break"] = False
                    yield img
                    continue
                runs = _para_runs(para)
                elem = ParagraphElement(
                    text=para.text,
                    style_name=para.style.name if para.style else "Normal",
                    runs=runs,
                    page_approx=state["page_approx"],
                    is_page_break=is_pb,
                )
                state["last_was_page_break"] = is_pb
                yield elem

            elif tag == "tbl":
                from docx.table import Table
                tbl = Table(child, doc)
                rows = _table_rows(tbl)
                col_count = max(len(r) for r in rows) if rows else 0
                elem = TableElement(
                    rows=rows,
                    col_count=col_count,
                    page_approx=state["page_approx"],
                    preceded_by_page_break=state["last_was_page_break"],
                )
                state["last_was_page_break"] = False
                yield elem

            else:
                if logger:
                    logger.debug(f"[document] Skipping unknown element: {tag!r}")
                state["last_was_page_break"] = False

    yield from _iter_children(body)
