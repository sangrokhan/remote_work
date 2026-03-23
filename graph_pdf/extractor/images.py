from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import math
from typing import List, Optional, Sequence, Tuple

import pdfplumber
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader

from .shared import _bboxes_intersect, _char_rotation_degrees
from .text import _detect_body_bounds, _is_non_watermark_obj


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


def _bbox_for_obj(obj: dict) -> Tuple[float, float, float, float]:
    return (
        float(obj.get("x0", 0.0)),
        float(obj.get("top", 0.0)),
        float(obj.get("x1", obj.get("x0", 0.0))),
        float(obj.get("bottom", obj.get("top", 0.0))),
    )


def _pdf_color_to_rgb(color: object, default: Tuple[int, int, int] = (0, 0, 0)) -> Tuple[int, int, int]:
    if isinstance(color, (int, float)):
        gray = float(color)
        if gray <= 1.0:
            level = int(round(max(0.0, min(gray, 1.0)) * 255))
            return (level, level, level)
        level = int(round(max(0.0, min(gray, 255.0))))
        return (level, level, level)

    if isinstance(color, (tuple, list)) and color:
        values = [float(value) for value in color[:3]]
        if len(values) == 1:
            level = values[0]
            if level <= 1.0:
                level = round(max(0.0, min(level, 1.0)) * 255)
            return (int(level), int(level), int(level))
        if max(values) <= 1.0:
            return tuple(int(round(max(0.0, min(value, 1.0)) * 255)) for value in values[:3])
        return tuple(int(round(max(0.0, min(value, 255.0)))) for value in values[:3])

    return default


def _scale_point(x: float, y: float, left: float, top: float, scale: float) -> Tuple[float, float]:
    return ((float(x) - left) * scale, (float(y) - top) * scale)


def _pil_font_path(fontname: str) -> Path | None:
    lower = str(fontname or "").lower()
    is_bold = "bold" in lower
    is_italic = "italic" in lower or "oblique" in lower
    candidates = {
        (False, False): Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        (True, False): Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        (False, True): Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
        (True, True): Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf"),
    }
    path = candidates[(is_bold, is_italic)]
    return path if path.exists() else None


@lru_cache(maxsize=64)
def _load_pil_font(fontname: str, size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    font_path = _pil_font_path(fontname)
    if font_path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(str(font_path), max(1, int(size)))


def _sample_cubic_bezier(
    start: Tuple[float, float],
    control1: Tuple[float, float],
    control2: Tuple[float, float],
    end: Tuple[float, float],
    steps: int = 24,
) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for step in range(steps + 1):
        t = step / steps
        one_minus_t = 1.0 - t
        x = (
            (one_minus_t**3) * start[0]
            + 3.0 * (one_minus_t**2) * t * control1[0]
            + 3.0 * one_minus_t * (t**2) * control2[0]
            + (t**3) * end[0]
        )
        y = (
            (one_minus_t**3) * start[1]
            + 3.0 * (one_minus_t**2) * t * control1[1]
            + 3.0 * one_minus_t * (t**2) * control2[1]
            + (t**3) * end[1]
        )
        points.append((x, y))
    return points


def _path_points(path: Sequence[tuple]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    current: Tuple[float, float] | None = None
    subpath_start: Tuple[float, float] | None = None

    for step in path:
        operator = step[0]
        if operator == "m" and len(step) >= 2:
            current = (float(step[1][0]), float(step[1][1]))
            subpath_start = current
            points.append(current)
            continue
        if operator == "l" and len(step) >= 2:
            current = (float(step[1][0]), float(step[1][1]))
            points.append(current)
            continue
        if operator == "c" and len(step) >= 4 and current is not None:
            control1 = (float(step[1][0]), float(step[1][1]))
            control2 = (float(step[2][0]), float(step[2][1]))
            end = (float(step[3][0]), float(step[3][1]))
            points.extend(_sample_cubic_bezier(current, control1, control2, end)[1:])
            current = end
            continue
        if operator == "h" and subpath_start is not None:
            current = subpath_start
            points.append(subpath_start)

    return points


def _draw_graphic_path(
    draw: ImageDraw.ImageDraw,
    obj: dict,
    region_bbox: Tuple[float, float, float, float],
    scale: float,
) -> None:
    left, top, _right, _bottom = region_bbox
    stroke_color = _pdf_color_to_rgb(obj.get("stroking_color"), default=(0, 0, 0))
    fill_color = _pdf_color_to_rgb(obj.get("non_stroking_color"), default=(255, 255, 255))
    line_width = max(1, int(round(float(obj.get("linewidth", 1.0)) * scale)))
    points = _path_points(obj.get("path") or [])
    if not points:
        return

    scaled_points = [_scale_point(x, y, left, top, scale) for x, y in points]
    if obj.get("fill") and len(scaled_points) >= 3:
        draw.polygon(scaled_points, fill=fill_color)
    if obj.get("stroke") and len(scaled_points) >= 2:
        draw.line(scaled_points, fill=stroke_color, width=line_width)


def _draw_rect(
    draw: ImageDraw.ImageDraw,
    rect: dict,
    region_bbox: Tuple[float, float, float, float],
    scale: float,
) -> None:
    left, top, _right, _bottom = region_bbox
    x0, y0, x1, y1 = _bbox_for_obj(rect)
    scaled_bbox = (
        (x0 - left) * scale,
        (y0 - top) * scale,
        (x1 - left) * scale,
        (y1 - top) * scale,
    )
    fill = _pdf_color_to_rgb(rect.get("non_stroking_color"), default=(255, 255, 255)) if rect.get("fill") else None
    outline = _pdf_color_to_rgb(rect.get("stroking_color"), default=(0, 0, 0)) if rect.get("stroke") else None
    width = max(1, int(round(float(rect.get("linewidth", 1.0)) * scale)))
    draw.rectangle(scaled_bbox, fill=fill, outline=outline, width=width if outline is not None else 0)


def _draw_char(
    image: Image.Image,
    char: dict,
    region_bbox: Tuple[float, float, float, float],
    scale: float,
) -> None:
    if not _is_non_watermark_obj(char):
        return
    text = str(char.get("text") or "")
    if not text:
        return

    left, top, _right, _bottom = region_bbox
    x0, char_top, _x1, _char_bottom = _bbox_for_obj(char)
    font_size = max(1, int(round(float(char.get("size", 0.0)) * scale)))
    font = _load_pil_font(str(char.get("fontname") or ""), font_size)
    fill = _pdf_color_to_rgb(char.get("non_stroking_color") or char.get("stroking_color"), default=(0, 0, 0))
    x = (x0 - left) * scale
    y = (char_top - top) * scale
    angle = _char_rotation_degrees(char)

    if abs(angle) <= 0.1:
        ImageDraw.Draw(image).text((x, y), text, font=font, fill=fill)
        return

    padding = max(4, font_size)
    layer = Image.new("RGBA", (font_size * 4, font_size * 4), (255, 255, 255, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((padding, padding), text, font=font, fill=(*fill, 255))
    rotated = layer.rotate(-angle, expand=True)
    image.alpha_composite(rotated, (int(round(x - padding)), int(round(y - padding))))


def _render_drawing_region_image(
    page: "pdfplumber.page.Page",
    bbox: Tuple[float, float, float, float],
    resolution: float,
) -> Image.Image:
    left, top, right, bottom = bbox
    scale = float(resolution) / 72.0
    width_px = max(1, int(math.ceil((right - left) * scale)))
    height_px = max(1, int(math.ceil((bottom - top) * scale)))
    image = Image.new("RGBA", (width_px, height_px), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    for rect in getattr(page, "rects", []):
        if _bboxes_intersect(_bbox_for_obj(rect), bbox):
            _draw_rect(draw, rect, bbox, scale)

    for obj in [*getattr(page, "lines", []), *getattr(page, "curves", [])]:
        if _bboxes_intersect(_bbox_for_obj(obj), bbox):
            _draw_graphic_path(draw, obj, bbox, scale)

    for char in getattr(page, "chars", []):
        if _bboxes_intersect(_bbox_for_obj(char), bbox):
            _draw_char(image, char, bbox, scale)

    return image.convert("RGB")


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
                    region_image = _render_drawing_region_image(
                        page=plumber_page,
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
