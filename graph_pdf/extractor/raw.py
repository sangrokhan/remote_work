from __future__ import annotations

import base64
import json
import tempfile
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Iterator, Optional, Sequence

import pdfplumber
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, BooleanObject, ByteStringObject, DictionaryObject, FloatObject, IndirectObject, NameObject, NullObject, NumberObject, StreamObject, TextStringObject


def _object_to_json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        return [_object_to_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_object_to_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _object_to_json_safe(item) for key, item in value.items()}
    return str(value)


def _serialize_pdf_object(
    value: object,
    *,
    seen: set[tuple[int, int]] | None = None,
    depth: int = 0,
    max_depth: int = 16,
) -> object:
    seen = seen or set()
    if depth > max_depth:
        return {"type": "max_depth"}
    if isinstance(value, BooleanObject):
        return bool(value)
    if value is None or isinstance(value, (bool, int, float, str, BooleanObject, FloatObject, NumberObject, NameObject, TextStringObject)):
        if isinstance(value, (FloatObject, NumberObject)):
            return float(value)
        return str(value) if isinstance(value, NameObject) else value
    if isinstance(value, NullObject):
        return None
    if isinstance(value, ByteStringObject):
        return {"type": "bytes", "base64": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, IndirectObject):
        ref = (int(value.idnum), int(value.generation))
        if ref in seen:
            return {"type": "indirect_ref", "idnum": ref[0], "generation": ref[1]}
        seen.add(ref)
        return {
            "type": "indirect",
            "idnum": ref[0],
            "generation": ref[1],
            "value": _serialize_pdf_object(value.get_object(), seen=seen, depth=depth + 1, max_depth=max_depth),
        }
    if isinstance(value, StreamObject):
        return {
            "type": "stream",
            "dict": {
                str(key): _serialize_pdf_object(item, seen=seen, depth=depth + 1, max_depth=max_depth)
                for key, item in value.items()
            },
            "data_base64": base64.b64encode(value.get_data()).decode("ascii"),
        }
    if isinstance(value, DictionaryObject):
        return {
            str(key): _serialize_pdf_object(item, seen=seen, depth=depth + 1, max_depth=max_depth)
            for key, item in value.items()
        }
    if isinstance(value, ArrayObject):
        return [_serialize_pdf_object(item, seen=seen, depth=depth + 1, max_depth=max_depth) for item in value]
    if isinstance(value, (list, tuple)):
        return [_serialize_pdf_object(item, seen=seen, depth=depth + 1, max_depth=max_depth) for item in value]
    if isinstance(value, bytes):
        return {"type": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    return str(value)


def _page_content_stream_base64(reader_page) -> str:
    contents = reader_page.get_contents()
    if contents is None:
        return ""
    if hasattr(contents, "get_data"):
        return base64.b64encode(contents.get_data()).decode("ascii")
    return ""


def _page_embedded_images(reader_page) -> list[dict]:
    payload: list[dict] = []
    for image_file in getattr(reader_page, "images", []):
        payload.append(
            {
                "name": str(getattr(image_file, "name", "") or ""),
                "data_base64": base64.b64encode(bytes(getattr(image_file, "data", b""))).decode("ascii"),
            }
        )
    return payload


def _page_object_dump(plumber_page: "pdfplumber.page.Page") -> dict:
    return {
        "chars": _object_to_json_safe(list(getattr(plumber_page, "chars", []) or [])),
        "lines": _object_to_json_safe(list(getattr(plumber_page, "lines", []) or [])),
        "rects": _object_to_json_safe(list(getattr(plumber_page, "rects", []) or [])),
        "curves": _object_to_json_safe(list(getattr(plumber_page, "curves", []) or [])),
        "images": _object_to_json_safe(list(getattr(plumber_page, "images", []) or [])),
        "horizontal_edges": _object_to_json_safe(list(getattr(plumber_page, "horizontal_edges", []) or [])),
        "vertical_edges": _object_to_json_safe(list(getattr(plumber_page, "vertical_edges", []) or [])),
        "words": _object_to_json_safe(
            plumber_page.extract_words(
                x_tolerance=1.5,
                y_tolerance=2.0,
                keep_blank_chars=False,
                extra_attrs=["size", "fontname"],
            )
            or []
        ),
    }


def _subset_pdf_bytes(pdf_path: Path, page_numbers: Sequence[int]) -> bytes:
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    for page_no in page_numbers:
        writer.add_page(reader.pages[page_no - 1])
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def dump_pdf_to_raw_file(
    pdf_path: Path,
    raw_path: Path,
    pages: Optional[Sequence[int]] = None,
) -> Path:
    selected_pages = [int(page_no) for page_no in (pages or [])]
    with pdfplumber.open(str(pdf_path)) as plumber_pdf:
        total_pages = len(plumber_pdf.pages)
        if not selected_pages:
            selected_pages = list(range(1, total_pages + 1))

    subset_pdf = _subset_pdf_bytes(pdf_path, selected_pages)
    reader = PdfReader(BytesIO(subset_pdf))
    page_payloads: list[dict] = []
    with pdfplumber.open(BytesIO(subset_pdf)) as plumber_pdf:
        for raw_index, (reader_page, plumber_page) in enumerate(zip(reader.pages, plumber_pdf.pages), start=1):
            original_page_no = selected_pages[raw_index - 1]
            page_payloads.append(
                {
                    "raw_page_number": raw_index,
                    "source_page_number": original_page_no,
                    "width": float(plumber_page.width),
                    "height": float(plumber_page.height),
                    "rotation": int(reader_page.get("/Rotate", 0) or 0),
                    "mediabox": [float(value) for value in reader_page.mediabox],
                    "cropbox": [float(value) for value in reader_page.cropbox],
                    "content_stream_base64": _page_content_stream_base64(reader_page),
                    "resources": _serialize_pdf_object(reader_page.get("/Resources")),
                    "embedded_images": _page_embedded_images(reader_page),
                    "objects": _page_object_dump(plumber_page),
                }
            )

    payload = {
        "schema_version": "1.0",
        "source_pdf": str(pdf_path),
        "source_name": pdf_path.name,
        "selected_pages": selected_pages,
        "page_count": len(page_payloads),
        "document_pdf_base64": base64.b64encode(subset_pdf).decode("ascii"),
        "pages": page_payloads,
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return raw_path


def load_raw_payload(raw_path: Path) -> dict:
    return json.loads(Path(raw_path).read_text(encoding="utf-8"))


@contextmanager
def materialize_raw_dump(raw_path: Path) -> Iterator[tuple[Path, dict]]:
    payload = load_raw_payload(raw_path)
    pdf_bytes = base64.b64decode(str(payload.get("document_pdf_base64") or ""))
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_name = Path(str(payload.get("source_name") or "raw_document.pdf")).with_suffix(".pdf").name
        materialized_path = Path(tmpdir) / pdf_name
        materialized_path.write_bytes(pdf_bytes)
        yield materialized_path, payload
