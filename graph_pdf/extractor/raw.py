from __future__ import annotations

import base64
import json
import tempfile
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Iterator, Sequence

from pypdf import PdfReader, PdfWriter


RAW_SCHEMA_VERSION = "1.1"


def _subset_pdf_bytes(pdf_path: Path, page_numbers: Sequence[int]) -> bytes:
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    for page_no in page_numbers:
        writer.add_page(reader.pages[page_no - 1])
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _normalize_selected_pages(total_pages: int, pages: Sequence[int] | None) -> list[int]:
    selected_pages = [int(page_no) for page_no in (pages or [])]
    if not selected_pages:
        return list(range(1, total_pages + 1))
    return selected_pages


def dump_pdf_to_raw_file(
    pdf_path: Path,
    raw_path: Path,
    pages: Sequence[int] | None = None,
) -> Path:
    reader = PdfReader(str(pdf_path))
    selected_pages = _normalize_selected_pages(len(reader.pages), pages)
    subset_pdf = _subset_pdf_bytes(pdf_path, selected_pages)

    payload = {
        "schema_version": RAW_SCHEMA_VERSION,
        "document_pdf_base64": base64.b64encode(subset_pdf).decode("ascii"),
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return raw_path


def load_raw_payload(raw_path: Path) -> dict:
    return json.loads(Path(raw_path).read_text(encoding="utf-8"))


@contextmanager
def materialize_raw_dump(raw_path: Path) -> Iterator[tuple[Path, dict]]:
    payload = load_raw_payload(raw_path)
    raw_base64 = payload.get("document_pdf_base64")
    if not isinstance(raw_base64, str) or not raw_base64:
        raise ValueError("raw dump is missing required document_pdf_base64")

    pdf_bytes = base64.b64decode(raw_base64)
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_name = Path(str(payload.get("source_name") or "raw_document.pdf")).with_suffix(".pdf").name
        materialized_path = Path(tmpdir) / pdf_name
        materialized_path.write_bytes(pdf_bytes)
        yield materialized_path, payload
