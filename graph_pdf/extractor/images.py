from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import math
from typing import List, Optional, Sequence, Tuple

import pdfplumber
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader

from .shared import _bboxes_intersect, _char_rotation_degrees
from .text import _detect_body_bounds, _is_non_watermark_obj, _selected_drawing_image_groups


def _normalize_pdf_image_name(image_name: object) -> str:
    return Path(str(image_name or "")).name


def _normalize_pdf_image_stem(image_name: object) -> str:
    normalized = _normalize_pdf_image_name(image_name)
    lower = normalized.lower()
    for suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"):
        if lower.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _image_ref_bbox(image_meta: dict) -> Tuple[float, float, float, float] | None:
    if not image_meta:
        return None
    x0 = image_meta.get("x0")
    top = image_meta.get("top")
    x1 = image_meta.get("x1")
    bottom = image_meta.get("bottom")
    if x0 is None or top is None or x1 is None or bottom is None:
        return None
    try:
        return float(x0), float(top), float(x1), float(bottom)
    except (TypeError, ValueError):
        return None


def _match_embedded_image_by_name(
    image_name: str,
    image_stem: str,
    candidates: Sequence[object],
) -> dict | None:
    for candidate in candidates:
        candidate_name = _normalize_pdf_image_name(
            candidate.get("name") if isinstance(candidate, dict) else getattr(candidate, "name", "")
        )
        candidate_stem = _normalize_pdf_image_stem(candidate_name)
        if candidate_name == image_name or candidate_stem == image_stem:
            if isinstance(candidate, dict):
                return dict(candidate)
            return {
                "name": str(getattr(candidate, "name", image_name)),
                "data": getattr(candidate, "data", None),
            }
    return None


def _collect_embedded_image_refs(
    pdf_path: Path,
    pages: Optional[Sequence[int]] = None,
    header_margin: float = 90.0,
    footer_margin: float = 40.0,
) -> dict[int, list[dict]]:
    # Shared helper for content-flow reference generation and image extraction.
    selected_pages = set(int(page_no) for page_no in (pages or []))
    refs_by_page: dict[int, list[dict]] = {}

    reader = PdfReader(str(pdf_path))
    with pdfplumber.open(str(pdf_path)) as plumber_pdf:
        for page_idx, (page, plumber_page) in enumerate(zip(reader.pages, plumber_pdf.pages), start=1):
            if selected_pages and page_idx not in selected_pages:
                continue

            body_top, body_bottom = _detect_body_bounds(
                plumber_page,
                header_margin=header_margin,
                footer_margin=footer_margin,
            )
            plumber_images = [
                image
                for image in getattr(plumber_page, "images", [])
                if _image_intersects_body(image, body_top=body_top, body_bottom=body_bottom)
            ]
            if not plumber_images:
                continue

            allowed_names = {
                _normalize_pdf_image_name(image.get("name"))
                for image in plumber_images
                if str(image.get("name") or "").strip()
            }
            allowed_stems = {_normalize_pdf_image_stem(name) for name in allowed_names}

            entries: list[dict] = []
            image_entries = []
            for image_file in page.images:
                image_name = _normalize_pdf_image_name(getattr(image_file, "name", None))
                image_stem = _normalize_pdf_image_stem(image_name)
                if not image_name:
                    continue
                if image_stem not in allowed_stems and image_name not in allowed_names:
                    continue

                matched = _match_embedded_image_by_name(image_name, image_stem, plumber_images)
                if matched is None:
                    continue
                bbox = _image_ref_bbox(matched)
                if bbox is None:
                    continue

                image_entries.append(
                    {
                        "name": image_name,
                        "name_stem": image_stem,
                        "suffix": Path(image_name).suffix or ".bin",
                        "bbox": bbox,
                        "pdf_name": str(getattr(image_file, "name", "")),
                    }
                )

            # Stable order matches page rendering order and keeps index assignment deterministic.
            image_entries = sorted(
                image_entries,
                key=lambda entry: (float(entry["bbox"][1]), float(entry["bbox"][0])),
            )
            for index, entry in enumerate(image_entries, start=1):
                copied = dict(entry)
                copied["index"] = index
                entries.append(copied)
            if entries:
                refs_by_page[page_idx] = entries

    return refs_by_page


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


def _union_bboxes(bboxes: Sequence[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _expanded_drawing_render_bbox(
    page: "pdfplumber.page.Page",
    image_bbox: Tuple[float, float, float, float],
    header_margin: float,
    footer_margin: float,
) -> Tuple[float, float, float, float]:
    groups = _selected_drawing_image_groups(
        page=page,
        header_margin=header_margin,
        footer_margin=footer_margin,
    )
    for group in groups:
        if not _bboxes_intersect(group["image_bbox"], image_bbox):
            continue
        object_bboxes = [tuple(obj["bbox"]) for obj in group["objects"]]
        render_bbox = _union_bboxes(object_bboxes)
        text_bboxes = [
            _bbox_for_obj(char)
            for char in getattr(page, "chars", [])
            if _is_non_watermark_obj(char) and _bboxes_intersect(_bbox_for_obj(char), render_bbox)
        ]
        if text_bboxes:
            render_bbox = _union_bboxes([render_bbox, *text_bboxes])
        return render_bbox
    return image_bbox


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
    image_refs_by_page: Optional[dict[int, Sequence[dict]]] = None,
    excluded_regions_by_page: Optional[dict[int, Sequence[Tuple[float, float, float, float]]]] = None,
) -> List[Path]:
    # 이미지 저장은 본문/표 생성과 분리해 둔다.
    # 같은 PDF에서도 body markdown 생성 규칙과 이미지 export 규칙을 독립적으로 디버깅할 수 있어야 하기 때문이다.
    out_image_dir.mkdir(parents=True, exist_ok=True)

    image_files: List[Path] = []
    image_no = 1
    selected_pages = set(int(page_no) for page_no in (pages or []))
    drawing_regions_by_page = drawing_regions_by_page or {}
    excluded_regions_by_page = excluded_regions_by_page or {}
    reader = PdfReader(str(pdf_path))

    def _is_excluded_region(bbox: Tuple[float, float, float, float], page_no: int) -> bool:
        excluded_regions = excluded_regions_by_page.get(page_no, ())
        return any(_bboxes_intersect(region, bbox) for region in excluded_regions)

    with pdfplumber.open(str(pdf_path)) as plumber_pdf:
        for page_idx, (page, plumber_page) in enumerate(zip(reader.pages, plumber_pdf.pages), start=1):
            if selected_pages and page_idx not in selected_pages:
                continue

            header_margin = 90.0
            footer_margin = 40.0
            body_top, body_bottom = _detect_body_bounds(
                plumber_page,
                header_margin=header_margin,
                footer_margin=footer_margin,
            )
            allowed_names = {
                str(image_meta.get("name") or "")
                for image_meta in plumber_page.images
                if _image_intersects_body(image_meta, body_top=body_top, body_bottom=body_bottom)
                and not _is_excluded_region(_bbox_for_obj(image_meta), page_idx)
            }

            if image_refs_by_page is None:
                for image_file in page.images:
                    # pypdf와 pdfplumber의 image 이름 표현이 조금 달라서
                    # stem과 원본 이름 둘 다 비교해 매칭한다.
                    image_name = _normalize_pdf_image_name(image_file.name or "")
                    image_stem = _normalize_pdf_image_stem(image_name)
                    if image_stem not in allowed_names and image_name not in allowed_names:
                        continue
                    suffix = Path(image_name).suffix or ".bin"
                    out_path = out_image_dir / f"{stem}_image_{image_no}{suffix}"
                    image_no += 1
                    out_path.write_bytes(image_file.data)
                    image_files.append(out_path)
            else:
                for image_ref in image_refs_by_page.get(page_idx, []):
                    image_name = str(image_ref.get("name") or "")
                    image_stem = str(image_ref.get("name_stem") or _normalize_pdf_image_stem(image_name))
                    if image_stem not in allowed_names and image_name not in allowed_names:
                        continue
                    image_bbox = image_ref.get("bbox")
                    if (
                        not isinstance(image_bbox, Sequence)
                        or len(image_bbox) != 4
                        or _is_excluded_region((float(image_bbox[0]), float(image_bbox[1]), float(image_bbox[2]), float(image_bbox[3])), page_idx)
                    ):
                        continue
                    suffix = str(image_ref.get("suffix") or Path(image_name).suffix or ".bin")
                    out_path = out_image_dir / f"{stem}_image_{image_no}{suffix}"
                    image_no += 1
                    image_payload = _match_embedded_image_by_name(image_name, image_stem, page.images)
                    if image_payload is None:
                        continue
                    image_data = image_payload.get("data")
                    if image_data is None:
                        continue
                    out_path.write_bytes(image_data)
                    image_files.append(out_path)

            for region in drawing_regions_by_page.get(page_idx, []):
                x0, top, x1, bottom = region
                if _is_excluded_region((x0, top, x1, bottom), page_idx):
                    continue
                width = float(plumber_page.width or 0.0)
                height = float(plumber_page.height or 0.0)
                left = max(0.0, min(float(x0), width))
                right = max(0.0, min(float(x1), width))
                top_y = max(0.0, min(float(top), height))
                bottom_y = max(0.0, min(float(bottom), height))
                if right <= left or bottom_y <= top_y:
                    continue
                render_left, render_top, render_right, render_bottom = _expanded_drawing_render_bbox(
                    page=plumber_page,
                    image_bbox=(left, top_y, right, bottom_y),
                    header_margin=header_margin,
                    footer_margin=footer_margin,
                )
                render_left = max(0.0, min(float(render_left), width))
                render_right = max(0.0, min(float(render_right), width))
                render_top = max(0.0, min(float(render_top), height))
                render_bottom = max(0.0, min(float(render_bottom), height))
                if render_right <= render_left or render_bottom <= render_top:
                    continue
                try:
                    # drawing 이미지는 PDF object를 직접 다시 그려서 raster 이미지로 만든다.
                    region_image = _render_drawing_region_image(
                        page=plumber_page,
                        bbox=(render_left, render_top, render_right, render_bottom),
                        resolution=180.0,
                    )
                    out_path = out_image_dir / f"{stem}_image_{image_no}.png"
                    image_no += 1
                    region_image.save(str(out_path), format="PNG")
                    image_files.append(out_path)
                except Exception:
                    continue

    return image_files
