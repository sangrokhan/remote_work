from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pdfplumber
from pypdf import PdfReader

from .text import _detect_body_bounds


def _image_intersects_body(
    image_meta: dict,
    body_top: float,
    body_bottom: float,
) -> bool:
    # Header/footer artwork is ignored; only images overlapping the body band are exported.
    top = float(image_meta.get("top", 0.0))
    bottom = float(image_meta.get("bottom", top))
    return bottom > body_top and top < body_bottom


def _extract_embedded_images(
    pdf_path: Path,
    out_image_dir: Path,
    stem: str,
    pages: Optional[Sequence[int]] = None,
) -> List[Path]:
    # Image extraction is intentionally independent from table/text extraction so it can be reused or debugged separately.
    out_image_dir.mkdir(parents=True, exist_ok=True)

    image_files: List[Path] = []
    selected_pages = set(int(page_no) for page_no in (pages or []))
    reader = PdfReader(str(pdf_path))
    with pdfplumber.open(str(pdf_path)) as plumber_pdf:
        for page_idx, (page, plumber_page) in enumerate(zip(reader.pages, plumber_pdf.pages), start=1):
            if selected_pages and page_idx not in selected_pages:
                continue

            body_top, body_bottom = _detect_body_bounds(
                plumber_page,
                header_margin=90.0,
                footer_margin=40.0,
            )
            allowed_names = {
                str(image_meta.get("name") or "")
                for image_meta in plumber_page.images
                if _image_intersects_body(image_meta, body_top=body_top, body_bottom=body_bottom)
            }

            kept_idx = 0
            for image_file in page.images:
                # pypdf and pdfplumber expose image identifiers slightly differently, so compare both forms.
                image_name = Path(image_file.name or "").name
                image_stem = Path(image_name).stem
                if image_stem not in allowed_names and image_name not in allowed_names:
                    continue
                kept_idx += 1
                suffix = Path(image_name).suffix or ".bin"
                out_path = out_image_dir / f"{stem}_page_{page_idx:02d}_image_{kept_idx:02d}{suffix}"
                out_path.write_bytes(image_file.data)
                image_files.append(out_path)

    return image_files
