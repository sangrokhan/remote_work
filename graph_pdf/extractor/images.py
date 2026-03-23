from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pdfplumber
from pypdf import PdfReader

from .text import _detect_body_bounds


def _crop_page_region(
    page_image: "pdfplumber.display.PageImage",
    page_height: float,
    bbox: Tuple[float, float, float, float],
    resolution: float,
) -> "Image":
    # `PageImage` removed `.crop()` in current versions, so crop via Pillow with
    # explicit point-to-pixel conversion using pdfplumber's top-origin coordinates.
    x0, top, x1, bottom = bbox
    left = float(min(x0, x1))
    right = float(max(x0, x1))
    y0 = float(min(top, bottom))
    y1 = float(max(top, bottom))

    scale = float(resolution) / 72.0
    image_width, image_height = page_image.original.size
    left_px = max(0, min(int(round(left * scale)), image_width))
    right_px = max(0, min(int(round(right * scale)), image_width))
    top_px = max(0, min(int(round(y0 * scale)), image_height))
    bottom_px = max(0, min(int(round(y1 * scale)), image_height))

    # `PageImage.original` uses top-left origin, matching pdfplumber `top`/`bottom`.
    return page_image.original.crop((left_px, top_px, right_px, bottom_px))


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
    drawing_regions_by_page: Optional[dict[int, Sequence[Tuple[float, float, float, float]]]] = None,
) -> List[Path]:
    # Image extraction is intentionally independent from table/text extraction so it can be reused or debugged separately.
    out_image_dir.mkdir(parents=True, exist_ok=True)

    image_files: List[Path] = []
    selected_pages = set(int(page_no) for page_no in (pages or []))
    drawing_regions_by_page = drawing_regions_by_page or {}
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

            drawing_image_idx = 0
            for region in drawing_regions_by_page.get(page_idx, []):
                x0, top, x1, bottom = region
                width = float(plumber_page.width or 0.0)
                height = float(plumber_page.height or 0.0)
                left = max(0.0, min(float(x0), width))
                right = max(0.0, min(float(x1), width))
                top_y = max(0.0, min(float(top), height))
                bottom_y = max(0.0, min(float(bottom), height))
                if right <= left or bottom_y <= top_y:
                    continue
                try:
                    page_image = plumber_page.to_image(resolution=180)
                    region_image = _crop_page_region(
                        page_image=page_image,
                        page_height=height,
                        bbox=(left, top_y, right, bottom_y),
                        resolution=180.0,
                    )
                    drawing_image_idx += 1
                    out_path = out_image_dir / f"{stem}_page_{page_idx:02d}_drawing_{drawing_image_idx:02d}.png"
                    region_image.save(str(out_path), format="PNG")
                    image_files.append(out_path)
                except Exception:
                    continue

    return image_files
