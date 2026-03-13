import argparse
import json
import logging
import math
import re
import tempfile
from collections import Counter
from pathlib import Path

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
_LOGGER = logging.getLogger("read_pdf")

_SUPPORTED_ROTATIONS = {0, 90, 180, 270}
_WATERMARK_ROTATION_DEGREE = 55
_WATERMARK_ROTATION_TOLERANCE = 2.5
_BULLET_REPLACEMENTS = {
    "•": "-",
    "◦": "-",
    "·": "-",
    "▪": "-",
    "‣": "-",
    "∙": "-",
    "○": "-",
    "◉": "-",
    "★": "-",
}

_KOREAN_FONT_HINTS = (
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJKkr-Regular.otf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/System/Library/Fonts/Supplemental/AppleSDGothicNeo.ttc",
    "/Library/Fonts/AppleGothic.ttf",
    "/Library/Fonts/AppleMyungjo.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgunb.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
)
_RECONSTRUCTION_KOREAN_FONT = None


def _contains_korean(text):
    for ch in text:
        codepoint = ord(ch)
        if (
            0x1100 <= codepoint <= 0x11FF
            or 0x3130 <= codepoint <= 0x318F
            or 0xAC00 <= codepoint <= 0xD7A3
            or 0x2E80 <= codepoint <= 0x2FD5
            or 0x2FF0 <= codepoint <= 0x2FFF
            or 0x3000 <= codepoint <= 0x303F
            or 0x31C0 <= codepoint <= 0x31EF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0xFE30 <= codepoint <= 0xFE4F
            or 0x20000 <= codepoint <= 0x2FA1F
            or 0x2F800 <= codepoint <= 0x2FA1F
        ):
            return True
    return False


def _get_reconstruct_fontfile():
    global _RECONSTRUCTION_KOREAN_FONT

    if _RECONSTRUCTION_KOREAN_FONT is not None:
        return _RECONSTRUCTION_KOREAN_FONT

    for font_path in _KOREAN_FONT_HINTS:
        try:
            if Path(font_path).is_file():
                _RECONSTRUCTION_KOREAN_FONT = str(font_path)
                return _RECONSTRUCTION_KOREAN_FONT
        except OSError:
            continue

    search_roots = (
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "/Library/Fonts",
        "/System/Library/Fonts",
        "C:/Windows/Fonts",
    )
    font_name_markers = (
        "noto",
        "nanum",
        "malgun",
        "applegothic",
        "applegothicneoregular",
        "batang",
        "gulim",
        "dotum",
        "msung",
        "msjh",
        "msyhl",
        "hei",
        "microsoftyi",
        "wqy",
        "sourcehans",
    )
    for root_path in search_roots:
        try:
            root = Path(root_path)
            if not root.is_dir():
                continue
        except OSError:
            continue

        for ext in ("*.ttf", "*.ttc", "*.otf"):
            for path in root.rglob(ext):
                try:
                    path_str = str(path).lower()
                except OSError:
                    continue
                if any(marker in path_str for marker in font_name_markers):
                    try:
                        _RECONSTRUCTION_KOREAN_FONT = str(path)
                        return _RECONSTRUCTION_KOREAN_FONT
                    except OSError:
                        continue

    _RECONSTRUCTION_KOREAN_FONT = ""
    return _RECONSTRUCTION_KOREAN_FONT


def _normalize_bullets(text):
    if not text:
        return text
    return "".join(_BULLET_REPLACEMENTS.get(ch, ch) for ch in text)


def _round_float(value, ndigits=2):
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def _parse_pages(pages):
    if pages is None:
        return None

    normalized = []
    seen = set()
    for part in str(pages).split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            begin_str, end_str = part.split("-", 1)
            if not begin_str.strip() or not end_str.strip():
                raise ValueError(f"Invalid page range '{part}'")

            start = int(begin_str.strip())
            end = int(end_str.strip())
            if start < 1 or end < 1:
                raise ValueError(f"Page numbers must be >= 1: '{part}'")

            step = 1 if start <= end else -1
            for page_no in range(start, end + step, step):
                if page_no in seen:
                    continue
                seen.add(page_no)
                normalized.append(page_no)
        else:
            page_no = int(part)
            if page_no < 1:
                raise ValueError(f"Page numbers must be >= 1: '{part}'")
            if page_no in seen:
                continue
            seen.add(page_no)
            normalized.append(page_no)

    return normalized


def _coerce_page_numbers(doc, requested_pages, max_pages):
    if requested_pages is not None:
        if not requested_pages:
            return []

        total_pages = int(doc.page_count or 0)
        page_numbers = [p for p in requested_pages if 1 <= p <= total_pages]
        if len(page_numbers) != len(requested_pages):
            _LOGGER.warning(
                "Some requested page numbers are out of range. source=%s requested=%s filtered=%s total=%s",
                getattr(doc, "name", ""),
                requested_pages,
                page_numbers,
                total_pages,
            )
        return sorted(page_numbers)

    if max_pages is None:
        return list(range(1, int(doc.page_count) + 1))

    if max_pages <= 0:
        return []

    total_pages = int(doc.page_count or 0)
    end_page = min(int(max_pages), total_pages)
    return list(range(1, end_page + 1))


def _surrounding_snippet(text, position, radius=40):
    start = max(0, position - radius)
    end = min(len(text), position + radius + 1)
    segment = text[start:end]
    return segment.encode("unicode_escape").decode("ascii")


def _sanitize_text(value, context=None):
    if value is None:
        return value

    raw = str(value)
    matches = list(_SURROGATE_RE.finditer(raw))
    if not matches:
        return _normalize_bullets(raw)

    if context is None:
        return _normalize_bullets(_SURROGATE_RE.sub("", raw))

    positions = [match.start() for match in matches]
    preview = [_surrounding_snippet(raw, position) for position in positions[:3]]
    _LOGGER.warning(
        "Removed surrogate code units in PDF text. source=%s page=%s line=%s span=%s count=%s snippets=%s",
        context.get("source"),
        context.get("page"),
        context.get("line"),
        context.get("span"),
        len(positions),
        ", ".join(preview),
    )
    return _normalize_bullets(_SURROGATE_RE.sub("", raw))


def _normalize_line(line):
    return re.sub(r"\s+", " ", _sanitize_text(line)).strip()


def _looks_like_repeated_watermark(line):
    normalized = _normalize_line(line)
    if not normalized:
        return False
    if len(normalized) > 120:
        return False

    if len(normalized) < 4:
        return False

    alpha_or_digit = [ch for ch in normalized if ch.isalpha() or ch.isdigit()]
    if len(alpha_or_digit) < 4:
        return False

    if len(normalized.split()) > 8:
        return False

    return True


def _collect_repeated_watermark_lines(
    doc,
    source,
    header_ratio,
    footer_ratio,
    ratio_threshold,
    exclude_locations=("header", "footer"),
    debug=False,
):
    all_lines = []
    page_count = int(doc.page_count or 0)
    for page_no in range(1, page_count + 1):
        try:
            lines = _extract_page_lines(
                doc[page_no - 1],
                page_no,
                source,
                header_ratio,
                footer_ratio,
                preserve_newlines=False,
                strip_markdown_lines=False,
                debug=debug,
            )
        except Exception:
            continue
        for line in lines:
            line["source"] = source
        all_lines.append(lines)

    if not all_lines:
        return set()

    return _collect_repeated_lines(
        all_lines,
        ratio_threshold=ratio_threshold,
        exclude_locations=exclude_locations,
    )


def _collect_repeated_lines(pages, ratio_threshold, exclude_locations=("header", "footer")):
    if not pages:
        return set()

    page_count = len(pages)
    occurrence = Counter()

    for lines in pages:
        seen = set()
        for line in lines:
            location = line.get("location")
            if location in exclude_locations:
                continue

            normalized = _normalize_line(line.get("raw", "")).casefold()
            if not normalized or not _looks_like_repeated_watermark(normalized):
                continue

            if normalized in seen:
                continue
            occurrence[normalized] += 1
            seen.add(normalized)

    min_count = max(2, math.ceil(page_count * ratio_threshold))
    return {
        key
        for key, count in occurrence.items()
        if count >= min_count
    }


def _compile_patterns(raw_patterns):
    return [
        re.compile(pattern, re.IGNORECASE)
        for pattern in raw_patterns
        if pattern and pattern.strip()
    ]


def _first_pattern_hit(line, patterns):
    for pattern in patterns:
        match = pattern.search(line)
        if match:
            return pattern, match.start()
    return None, None


def _style_token(span):
    font = (span.get("font") or "").lower()
    flags = int(span.get("flags") or 0)
    is_bold = "bold" in font or bool(flags & 16)
    is_italic = "italic" in font or "oblique" in font or bool(flags & 2)
    is_mono = any(token in font for token in ("mono", "courier", "consola", "consolas"))
    return is_bold, is_italic, is_mono


def _span_to_markdown(raw, span):
    if not raw:
        return ""

    is_bold, is_italic, is_mono = _style_token(span)
    text = raw.strip("\n\r")
    if not text:
        return ""

    if is_bold and is_italic:
        return f"***{text}***"
    if is_bold:
        return f"**{text}**"
    if is_italic:
        return f"*{text}*"
    if is_mono:
        return f"`{text}`"
    return text


def _is_markdown_like(text):
    normalized = _normalize_line(text)
    if not normalized:
        return False

    if "|" in normalized and normalized.count("|") >= 2:
        marker_count = sum(1 for ch in normalized if ch in "|-+:")
        if marker_count >= normalized.count("|") and marker_count >= 2:
            return True
        if re.fullmatch(r"\|[-\s\|:]+\|", normalized):
            return True

    if re.search(r"^\s*`{1,3}.+`{1,3}\s*$", normalized):
        return True
    if re.fullmatch(r"\s*[`*_-]{3,}\s*", normalized):
        return True
    return False


def _is_rotation_match(rotation, target_rotation, tolerance):
    if target_rotation is None or tolerance is None:
        return False

    try:
        target = float(target_rotation)
        tolerance_value = float(tolerance)
    except (TypeError, ValueError):
        return False

    if tolerance_value < 0:
        tolerance_value = abs(tolerance_value)

    value = int(rotation or 0) % 360
    return abs((value - target + 180) % 360 - 180) <= tolerance_value


def _collect_removal_rects(
    page,
    page_no,
    source,
    header_ratio,
    footer_ratio,
    watermark_angle,
    watermark_tolerance,
    remove_rotation_markdown=True,
    strip_body_rotation=False,
    remove_markdown_lines=False,
    debug=False,
):
    lines = _extract_page_lines(
        page,
        page_no,
        source,
        header_ratio,
        footer_ratio,
        preserve_newlines=False,
        strip_markdown_lines=False,
        debug=debug,
    )

    removal_rects = []
    stats = {
        "checked_lines": 0,
        "rotation_matches": 0,
        "markdown_matches": 0,
        "removed": 0,
    }

    for line in lines:
        raw_text = _normalize_line(line.get("raw") or "")
        if not raw_text:
            continue
        stats["checked_lines"] += 1

        rotation = line.get("rotation")
        location = line.get("location", "body")
        rotation_match = _is_rotation_match(rotation, watermark_angle, watermark_tolerance)
        is_markdown = _is_markdown_like(raw_text)
        should_remove = False
        removed_reason = None

        if remove_rotation_markdown and rotation_match:
            stats["rotation_matches"] += 1
            remove_all_rotation = bool(strip_body_rotation)
            if location == "body" or remove_all_rotation:
                should_remove = True
                removed_reason = "watermark-rotation-body" if location == "body" else "watermark-rotation-header-footer"

        if remove_markdown_lines and is_markdown:
            should_remove = True
            removed_reason = removed_reason or "markdown"

        if not should_remove:
            if debug and rotation_match:
                _LOGGER.debug(
                    "Skipping rotation-match line in preview: source=%s page=%s line=%s location=%s markdown_like=%s text=%r",
                    source,
                    page_no,
                    line.get("line"),
                    location,
                    is_markdown,
                    raw_text[:120],
                )
            continue

        span_rects = []
        for span in line.get("spans") or []:
            span_bbox = span.get("bbox")
            if not (isinstance(span_bbox, (list, tuple)) and len(span_bbox) == 4):
                continue
            try:
                sx0, sy0, sx1, sy1 = [float(v) for v in span_bbox]
            except (TypeError, ValueError):
                continue
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            span_rects.append(pymupdf.Rect(sx0, sy0, sx1, sy1))

        if not span_rects:
            bbox = line.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            try:
                x0, y0, x1, y1 = [float(v) for v in bbox]
            except (TypeError, ValueError):
                continue
            if x1 <= x0 or y1 <= y0:
                continue
            span_rects.append(pymupdf.Rect(x0, y0, x1, y1))

        for rect in span_rects:
            if debug and removed_reason:
                _LOGGER.debug(
                    "Preview watermark-removal rect: source=%s page=%s line=%s location=%s reason=%s rotation=%s rect=%s",
                    source,
                    page_no,
                    line.get("line"),
                    location,
                    removed_reason,
                    rotation,
                    f"{_round_float(rect.x0)},{_round_float(rect.y0)},{_round_float(rect.x1)},{_round_float(rect.y1)}",
                )
            removal_rects.append(rect)

        stats["removed"] += 1

    if debug:
        _LOGGER.debug(
            "Preview removal summary: source=%s page=%s checked=%s rotation_matches=%s removed=%s markdown_like=%s markdown_option=%s",
            source,
            page_no,
            stats["checked_lines"],
            stats["rotation_matches"],
            stats["removed"],
            stats["markdown_matches"],
            remove_markdown_lines,
        )

    return removal_rects


def _append_span(parts, text):
    if not text:
        return
    if parts and not text.startswith(" ") and not parts[-1].endswith(" "):
        parts.append(" ")
    parts.append(text)


def _line_bbox(block_line, spans):
    bbox = block_line.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return tuple(float(v) for v in bbox)

    xs = []
    ys = []
    for span in spans:
        span_bbox = span.get("bbox")
        if isinstance(span_bbox, (list, tuple)) and len(span_bbox) == 4:
            x0, y0, x1, y1 = span_bbox
            xs.extend([float(x0), float(x1)])
            ys.extend([float(y0), float(y1)])
    if not xs or not ys:
        return (0.0, 0.0, 0.0, 0.0)

    return (min(xs), min(ys), max(xs), max(ys))


def _line_center(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0, 0.0
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _rotation_axis(rotation):
    normalized = int(rotation or 0) % 360
    if normalized in (90, 270):
        return "x"
    return "y"


def _is_watermark_rotation(rotation):
    value = int(rotation or 0) % 360
    return abs((value - _WATERMARK_ROTATION_DEGREE + 180) % 360 - 180) <= _WATERMARK_ROTATION_TOLERANCE


def _classify_region(page_rect, bbox, rotation, header_ratio, footer_ratio):
    if page_rect is None:
        return "body"

    x0, y0, x1, y1 = bbox
    axis = _rotation_axis(rotation)
    if axis == "x":
        center = (x0 + x1) / 2.0
        axis_size = float(page_rect.width)
    else:
        center = (y0 + y1) / 2.0
        axis_size = float(page_rect.height)

    if axis_size <= 0:
        return "body"

    if center <= axis_size * header_ratio:
        return "header"
    if center >= axis_size * (1 - footer_ratio):
        return "footer"
    return "body"


def _estimate_row_tolerance(lines):
    sizes = [float(line.get("size", 0.0) or 0.0) for line in lines if line.get("size")]
    if not sizes:
        return 2.5

    sizes.sort()
    median = sizes[len(sizes) // 2]
    if median <= 0:
        return 2.5
    return max(1.0, median * 0.7)


def _line_axis_span(line):
    bbox = line.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None, None

    x0, y0, x1, y1 = [float(v) for v in bbox]
    axis = (line.get("baseline") or {}).get("axis", "y")
    if axis == "x":
        start, end = min(x0, x1), max(x0, x1)
    else:
        start, end = min(y0, y1), max(y0, y1)

    return start, end


def _spans_overlap(a_start, a_end, b_start, b_end, tolerance):
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return False
    return b_start <= a_end + tolerance and a_start <= b_end + tolerance


def _assign_row_ids(lines):
    if not lines:
        return

    order = list(range(len(lines)))
    order.sort(
        key=lambda idx: (
            _round_float(lines[idx].get("baseline", {}).get("value", 0.0) or 0.0),
            _round_float(lines[idx].get("position", {}).get("x", 0.0) or 0.0),
        )
    )

    tolerance = _estimate_row_tolerance(lines)
    current = None
    row_no = 0
    for idx in order:
        line = lines[idx]
        axis = (line.get("baseline") or {}).get("axis", "y")
        value = (line.get("baseline") or {}).get("value")
        span_start, span_end = _line_axis_span(line)

        if value is None or span_start is None or span_end is None:
            row_no += 1
            current = None
            line["row_no"] = row_no
            continue

        span_start = float(span_start)
        span_end = float(span_end)

        if (
            current is None
            or current["axis"] != axis
            or not _spans_overlap(
                current["start"],
                current["end"],
                span_start,
                span_end,
                tolerance,
            )
        ):
            row_no += 1
            current = {"axis": axis, "start": span_start, "end": span_end}
        else:
            if span_start < current["start"]:
                current["start"] = span_start
            if span_end > current["end"]:
                current["end"] = span_end

        line["row_no"] = row_no


def _line_baseline(spans, axis):
    centers = []
    for span in spans:
        span_bbox = span.get("bbox")
        if not isinstance(span_bbox, (list, tuple)) or len(span_bbox) != 4:
            continue
        x0, y0, x1, y1 = span_bbox
        if axis == "x":
            centers.append((float(x0) + float(x1)) / 2.0)
        else:
            centers.append((float(y0) + float(y1)) / 2.0)

    if not centers:
        return 0.0
    return float(sum(centers) / len(centers))


def _line_tilt_angle(line_obj, spans):
    line_direction = line_obj.get("dir")
    if line_direction and len(line_direction) == 2:
        dx, dy = line_direction
        if dx or dy:
            return float(math.degrees(math.atan2(-dy, dx)))

    angles = []
    for span in spans:
        direction = span.get("dir")
        if not direction or len(direction) != 2:
            continue
        dx, dy = direction
        if not dx and not dy:
            continue
        angle = math.degrees(math.atan2(-dy, dx))
        angles.append(angle)

    if not angles:
        return 0.0
    return float(sum(angles) / len(angles))


def _to_point(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _to_rgb_color(value, default=(0.0, 0.0, 0.0)):
    if value is None:
        return default

    if isinstance(value, (list, tuple)):
        if len(value) >= 3:
            try:
                rgb = [float(component) for component in value[:3]]
            except (TypeError, ValueError):
                return default
            return tuple(max(0.0, min(1.0, component)) for component in rgb)
        return default

    try:
        v = float(value)
    except (TypeError, ValueError):
        return default

    if 0.0 <= v <= 1.0:
        return (v, v, v)

    if v < 0:
        return default

    iv = int(v)
    if iv <= 0xFFFFFF:
        return (
            ((iv >> 16) & 0xFF) / 255.0,
            ((iv >> 8) & 0xFF) / 255.0,
            (iv & 0xFF) / 255.0,
        )

    return default


def _to_rect_from_re(args):
    if not args:
        return None

    first = args[0]
    if isinstance(first, (list, tuple)) and len(first) == 4:
        try:
            return [float(first[0]), float(first[1]), float(first[2]), float(first[3])]
        except (TypeError, ValueError):
            return None

    if len(args) == 4:
        try:
            return [float(args[0]), float(args[1]), float(args[2]), float(args[3])]
        except (TypeError, ValueError):
            return None

    return None


def _coerce_number(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_shape_lines_from_drawing(drawing):
    if not isinstance(drawing, dict):
        return []

    items = drawing.get("items")
    if not isinstance(items, list):
        return []

    linewidth = _coerce_number(drawing.get("linewidth"), 0.0)
    color = drawing.get("color")
    if color is None and isinstance(drawing.get("fill"), (int, float)):
        color = drawing.get("fill")
    if color is None:
        color = drawing.get("stroke")

    segments = []
    cursor = None
    for item in items:
        if not item:
            continue

        op = item[0]
        args = item[1:]
        if op == "m" and args:
            cursor = _to_point(args[0]) if len(args) == 1 else _to_point(args)
            continue

        if op == "l" and args and cursor is not None:
            point = _to_point(args[0]) if len(args) == 1 else _to_point(args)
            if point is None:
                continue

            x0, y0 = cursor
            x1, y1 = point
            segments.append((x0, y0, x1, y1, linewidth, color))
            cursor = point
            continue

        if op == "re":
            rect = _to_rect_from_re(args)
            if rect is None:
                continue
            x0, y0, x1, y1 = rect
            if x0 == x1 or y0 == y1:
                continue
            segments.append((x0, y0, x0, y1, linewidth, color))
            segments.append((x0, y1, x1, y1, linewidth, color))
            segments.append((x1, y1, x1, y0, linewidth, color))
            segments.append((x1, y0, x0, y0, linewidth, color))
            continue

        if op in ("v", "y") and cursor is not None:
            if len(args) == 1:
                cursor = _to_point(args[0]) or cursor
            else:
                cursor = _to_point(args) or cursor
            continue

        if op == "h":
            if cursor is not None:
                segments.append((cursor[0], cursor[1], cursor[0], cursor[1], linewidth, color))
            continue

        if op.startswith("c") and args:
            # ignore Bézier control data unless it starts with a point-like argument
            if op in ("c",):
                end = _to_point(args[-1]) if len(args) >= 6 else _to_point(args)
                if end is not None and cursor is not None:
                    x0, y0 = cursor
                    x1, y1 = end
                    segments.append((x0, y0, x1, y1, linewidth, color))
                cursor = end

    return segments


def _segment_length(x0, y0, x1, y1):
    return math.hypot(x1 - x0, y1 - y0)


def _segment_orientation(x0, y0, x1, y1, tolerance=1.5):
    dx = x1 - x0
    dy = y1 - y0
    if abs(dy) <= tolerance and abs(dx) > tolerance:
        return "horizontal"
    if abs(dx) <= tolerance and abs(dy) > tolerance:
        return "vertical"
    return "other"


def _merge_numeric(values, tolerance):
    if not values:
        return []

    sorted_values = sorted(float(v) for v in values if v is not None)
    if not sorted_values:
        return []

    merged = []
    current = [sorted_values[0]]
    for value in sorted_values[1:]:
        if value <= current[-1] + tolerance:
            current.append(value)
        else:
            merged.append(sum(current) / len(current))
            current = [value]
    merged.append(sum(current) / len(current))
    return merged


def _collect_span_cells(lines, bbox=None):
    filter_bbox = None
    if (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and all(v is not None for v in bbox)
    ):
        try:
            filter_bbox = [float(v) for v in bbox]
            if filter_bbox[0] == filter_bbox[2] or filter_bbox[1] == filter_bbox[3]:
                filter_bbox = None
        except (TypeError, ValueError):
            filter_bbox = None

    if filter_bbox is not None:
        fx0, fy0, fx1, fy1 = filter_bbox

    if not lines:
        return []

    spans = []
    for line in lines:
        if line.get("is_watermark_rotation"):
            continue

        location = line.get("location", "body")
        if location == "watermark":
            continue

        for span in line.get("spans", []) or []:
            raw = _normalize_line((span.get("text") or "").strip())
            if not raw:
                continue

            span_bbox = span.get("bbox")
            if not isinstance(span_bbox, (list, tuple)) or len(span_bbox) != 4:
                continue

            try:
                x0, y0, x1, y1 = [float(v) for v in span_bbox]
            except (TypeError, ValueError):
                continue
            if x0 == x1 or y0 == y1:
                continue
            if filter_bbox is not None and (
                x1 < fx0
                or x0 > fx1
                or y1 < fy0
                or y0 > fy1
            ):
                continue

            spans.append(
                {
                    "text": raw,
                    "bbox": [x0, y0, x1, y1],
                    "x0": x0,
                    "x1": x1,
                    "y0": y0,
                    "y1": y1,
                    "cx": (x0 + x1) / 2.0,
                    "cy": (y0 + y1) / 2.0,
                    "line_no": line.get("line"),
                }
            )
    return spans


def _build_text_grid_table(
    lines,
    page_rect,
    page_no,
    source,
    rotation=0,
    debug=False,
    bbox=None,
    min_rows=2,
    min_row_candidates=2,
    min_support_ratio=0.25,
    infer_method="text-grid",
    override_row_lines=None,
    override_vertical_lines=None,
):
    spans = _collect_span_cells(lines, bbox=bbox)
    if not spans:
        return []

    row_tolerance = max(2.5, _estimate_row_tolerance(lines))
    rows = []
    for span in sorted(spans, key=lambda item: item["cy"]):
        if rows and abs(span["cy"] - rows[-1]["cy"]) <= row_tolerance:
            row = rows[-1]
            row["items"].append(span)
            row["count"] += 1
            row["cy"] = (row["cy"] * (row["count"] - 1) + span["cy"]) / row["count"]
            row["y0"] = min(row["y0"], span["y0"])
            row["y1"] = max(row["y1"], span["y1"])
        else:
            rows.append(
                {
                    "cy": span["cy"],
                    "count": 1,
                    "items": [span],
                    "y0": span["y0"],
                    "y1": span["y1"],
                }
            )

    if len(rows) < min_rows:
        if debug:
            _LOGGER.debug(
                "%s inference skipped: too few row clusters. source=%s page=%s rows=%s",
                infer_method,
                source,
                page_no,
                len(rows),
            )
        return []

    row_candidates = [row for row in rows if len(row["items"]) >= 2]
    if len(row_candidates) < min_row_candidates:
        if debug:
            _LOGGER.debug(
                "%s inference skipped: too few multi-cell rows. source=%s page=%s candidate_rows=%s",
                infer_method,
                source,
                page_no,
                len(row_candidates),
            )
        return []

    column_count_stats = Counter(len(row["items"]) for row in row_candidates)
    dominant_col_count, _ = column_count_stats.most_common(1)[0]
    if dominant_col_count < 2:
        return []

    support_threshold = max(2, dominant_col_count - 1)
    required_support = max(
        2,
        min(len(row_candidates), max(2, math.ceil(len(row_candidates) * min_support_ratio))),
    )
    support_rows = [
        row for row in row_candidates
        if len(row["items"]) >= support_threshold
    ]
    if len(support_rows) < required_support:
        if debug:
            _LOGGER.debug(
                "%s inference skipped: unstable row consistency. source=%s page=%s support_rows=%s required=%s",
                infer_method,
                source,
                page_no,
                len(support_rows),
                required_support,
            )
        return []

    column_points = [[] for _ in range(dominant_col_count + 1)]
    selected_rows = []
    for row in support_rows:
        cells = sorted(row["items"], key=lambda item: item["x0"])
        cells = cells[:dominant_col_count]
        if len(cells) < dominant_col_count:
            continue

        column_points[0].append(cells[0]["x0"])
        for idx in range(1, dominant_col_count):
            boundary = (cells[idx - 1]["x1"] + cells[idx]["x0"]) / 2.0
            column_points[idx].append(boundary)
        column_points[dominant_col_count].append(cells[dominant_col_count - 1]["x1"])
        selected_rows.append(
            {
                "y0": row["y0"],
                "y1": row["y1"],
                "items": cells,
            }
        )

    if len(selected_rows) < 2:
        return []

    merged_columns = [
        _merge_numeric(points, max(2.0, row_tolerance * 0.75))
        for points in column_points
    ]
    merged_columns = [points for points in merged_columns if points]
    if not merged_columns:
        return []

    x_positions = sorted(
        x
        for points in merged_columns
        for x in points
    )
    if not x_positions:
        return []

    tx0 = min(x_positions)
    tx1 = max(x_positions)

    y_values = []
    for row in selected_rows:
        y_values.append(row["y0"])
        y_values.append(row["y1"])

    ty0 = min(y_values)
    ty1 = max(y_values)
    if tx1 <= tx0 or ty1 <= ty0:
        return []

    row_boundaries = _merge_numeric(
        y_values,
        max(1.5, row_tolerance * 0.75),
    )
    if not row_boundaries:
        return []

    if override_row_lines and len(override_row_lines) >= 2:
        row_lines = list(override_row_lines)
    else:
        row_lines = [
            {
                "orientation": "horizontal",
                "x0": _round_float(tx0),
                "y0": _round_float(y),
                "x1": _round_float(tx1),
                "y1": _round_float(y),
            }
            for y in row_boundaries
        ]

    vertical_positions = _merge_numeric(
        x_positions,
        max(2.0, row_tolerance * 0.75),
    )
    if not vertical_positions:
        return []

    if override_vertical_lines and len(override_vertical_lines) >= 2:
        vertical_lines = list(override_vertical_lines)
    else:
        vertical_lines = [
            {
                "orientation": "vertical",
                "x0": _round_float(x),
                "y0": _round_float(ty0),
                "x1": _round_float(x),
                "y1": _round_float(ty1),
            }
            for x in vertical_positions
        ]

    row_texts = []
    for row in selected_rows:
        cells = sorted(row["items"], key=lambda item: item["x0"])
        row_texts.append(" | ".join(_sanitize_text(cell.get("text", "")) for cell in cells[:dominant_col_count]))

    if not row_texts:
        return []

    return [
        {
            "page": page_no,
            "start_page": page_no,
            "table_no": 1,
            "bbox": [tx0, ty0, tx1, ty1],
            "row_count": len(selected_rows),
            "col_count": dominant_col_count,
            "rotation": int(rotation or 0),
            "rows_text": row_texts,
            "x": _round_float((tx0 + tx1) / 2.0),
            "y": _round_float((ty0 + ty1) / 2.0),
            "page_width": float(page_rect.width) if page_rect else None,
            "page_height": float(page_rect.height) if page_rect else None,
            "text": "\n".join(row_texts),
            "row_lines": row_lines,
            "vertical_lines": vertical_lines,
            "components": {
                "bbox": [tx0, ty0, tx1, ty1],
                "row_lines": row_lines,
                "vertical_lines": vertical_lines,
            },
            "source": source,
            "infer_method": infer_method,
            "score": len(selected_rows) * dominant_col_count,
            "debug": {
                "candidate_rows": len(row_candidates),
                "selected_rows": len(selected_rows),
                "dominant_col_count": dominant_col_count,
            },
        }
    ]


def _extract_page_drawings(
    page,
    page_no,
    source,
    debug=False,
    header_ratio=0.08,
    footer_ratio=0.08,
):
    try:
        drawings = page.get_drawings()
    except Exception:
        if debug:
            _LOGGER.debug(
                "Could not read drawing objects: source=%s page=%s",
                source,
                page_no,
                exc_info=True,
            )
        return []

    lines = []
    page_rect = page.rect
    page_rotation = int(page.rotation or 0)
    for drawing in drawings:
        for x0, y0, x1, y1, linewidth, color in _extract_shape_lines_from_drawing(drawing):
            length = _segment_length(x0, y0, x1, y1)
            if length <= 1.5:
                continue
            if x0 == x1 and y0 == y1:
                continue

            min_x, min_y, max_x, max_y = (
                min(x0, x1),
                min(y0, y1),
                max(x0, x1),
                max(y0, y1),
            )
            region = _classify_region(
                page_rect,
                (min_x, min_y, max_x, max_y),
                page_rotation,
                header_ratio,
                footer_ratio,
            )
            orientation = _segment_orientation(x0, y0, x1, y1)
            lines.append(
                {
                    "type": "shape-line",
                    "page": page_no,
                    "source": source,
                    "region": region,
                    "x0": _round_float(x0),
                    "y0": _round_float(y0),
                    "x1": _round_float(x1),
                    "y1": _round_float(y1),
                    "x": _round_float((x0 + x1) / 2.0),
                    "y": _round_float((y0 + y1) / 2.0),
                    "length": _round_float(length),
                    "orientation": orientation,
                    "linewidth": _round_float(linewidth),
                    "color": color,
                }
            )

    if debug:
        _LOGGER.debug(
            "Extracted shape lines: source=%s page=%s count=%s",
            source,
            page_no,
            len(lines),
        )

    return lines


def _cell_value(cell, name, default=None):
    if cell is None:
        return default
    if isinstance(cell, dict):
        return cell.get(name, default)
    return getattr(cell, name, default)


def _cell_bbox(cell):
    raw_bbox = _cell_value(cell, "bbox")
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        try:
            return [float(v) for v in raw_bbox]
        except (TypeError, ValueError):
            pass

    x0 = _cell_value(cell, "x0")
    y0 = _cell_value(cell, "y0")
    x1 = _cell_value(cell, "x1")
    y1 = _cell_value(cell, "y1")
    if None in (x0, y0, x1, y1):
        return None
    try:
        return [float(x0), float(y0), float(x1), float(y1)]
    except (TypeError, ValueError):
        return None


def _merge_positions(values, tolerance):
    if not values:
        return []

    sorted_values = sorted(values)
    merged = [sorted_values[0]]
    for value in sorted_values[1:]:
        if value <= merged[-1] + tolerance:
            continue
        merged.append(value)
    return merged


def _infer_table_components(table, row_count=None, col_count=None):
    cells = _cell_value(table, "cells", []) or []
    table_bbox = _cell_value(table, "bbox")
    if not (
        isinstance(table_bbox, (list, tuple))
        and len(table_bbox) == 4
        and all(v is not None for v in table_bbox)
    ):
        return {"bbox": [], "row_lines": [], "vertical_lines": []}

    try:
        tx0, ty0, tx1, ty1 = [float(v) for v in table_bbox]
    except (TypeError, ValueError):
        return {"bbox": [], "row_lines": [], "vertical_lines": []}

    table_row_count = (
        _coerce_number(row_count)
        or _coerce_number(_cell_value(table, "row_count"))
        or _coerce_number(_cell_value(table, "rows"))
        or 0.0
    )
    table_col_count = (
        _coerce_number(col_count)
        or _coerce_number(_cell_value(table, "col_count"))
        or _coerce_number(_cell_value(table, "cols"))
        or 0.0
    )

    raw_y_starts = []
    raw_y_ends = []
    raw_x_starts = []
    raw_x_ends = []
    row_cells = []
    col_cells = []

    for cell in cells:
        row = _cell_value(
            cell,
            "row",
            _cell_value(
                cell,
                "row_idx",
                _cell_value(cell, "row_i", _cell_value(cell, "r", None)),
            ),
        )
        col = _cell_value(
            cell,
            "col",
            _cell_value(
                cell,
                "col_idx",
                _cell_value(cell, "col_i", _cell_value(cell, "c", None)),
            ),
        )

        bbox = _cell_bbox(cell)
        if bbox is None:
            continue

        c_x0, c_y0, c_x1, c_y1 = bbox
        raw_y_starts.append(c_y0)
        raw_y_ends.append(c_y1)
        raw_x_starts.append(c_x0)
        raw_x_ends.append(c_x1)

        if row is not None:
            row_cells.append((row, c_y0, c_y1))
        if col is not None:
            col_cells.append((col, c_x0, c_x1))

    row_heights = [float(end - start) for start, end in zip(raw_y_starts, raw_y_ends)]
    col_widths = [float(end - start) for start, end in zip(raw_x_starts, raw_x_ends)]
    row_tol = max(0.75, (sorted(row_heights)[len(row_heights) // 2] * 0.25)) if row_heights else 1.5
    col_tol = max(0.75, (sorted(col_widths)[len(col_widths) // 2] * 0.25)) if col_widths else 1.5

    y_positions = []
    x_positions = []

    if row_cells:
        y_values = []
        for _, top, bottom in row_cells:
            y_values.extend([top, bottom])
        y_positions = _merge_positions(y_values, row_tol)

    if col_cells:
        x_values = []
        for _, left, right in col_cells:
            x_values.extend([left, right])
        x_positions = _merge_positions(x_values, col_tol)

    if not y_positions and raw_y_starts:
        y_positions = _merge_positions(raw_y_starts + raw_y_ends, row_tol)

    if not x_positions and raw_x_starts:
        x_positions = _merge_positions(raw_x_starts + raw_x_ends, col_tol)

    if not y_positions and table_row_count > 0:
        if table_row_count > 0 and table_row_count == int(table_row_count):
            row_step = (ty1 - ty0) / table_row_count
            y_positions = [ty0 + row_step * i for i in range(int(table_row_count) + 1)]

    if not x_positions and table_col_count > 0:
        if table_col_count > 0 and table_col_count == int(table_col_count):
            col_step = (tx1 - tx0) / table_col_count
            x_positions = [tx0 + col_step * i for i in range(int(table_col_count) + 1)]

    if not y_positions:
        y_positions = [ty0, ty1]
    if not x_positions:
        x_positions = [tx0, tx1]

    y_positions = _merge_positions(sorted(set(_coerce_number(v, 0.0) for v in y_positions)), row_tol)
    x_positions = _merge_positions(sorted(set(_coerce_number(v, 0.0) for v in x_positions)), col_tol)
    if ty0 not in y_positions:
        y_positions.insert(0, ty0)
    if ty1 not in y_positions:
        y_positions.append(ty1)
    if tx0 not in x_positions:
        x_positions.insert(0, tx0)
    if tx1 not in x_positions:
        x_positions.append(tx1)

    row_lines = [
        {
            "orientation": "horizontal",
            "x0": _round_float(tx0),
            "y0": _round_float(y),
            "x1": _round_float(tx1),
            "y1": _round_float(y),
        }
        for y in y_positions
    ]

    vertical_lines = [
        {
            "orientation": "vertical",
            "x0": _round_float(x),
            "y0": _round_float(ty0),
            "x1": _round_float(x),
            "y1": _round_float(ty1),
        }
        for x in x_positions
    ]

    return {
        "bbox": [tx0, ty0, tx1, ty1],
        "row_lines": row_lines,
        "vertical_lines": vertical_lines,
    }


def _rows_from_cells(cells):
    if not cells:
        return []

    rows = []
    for cell in cells:
        cell_row = _cell_value(
            cell,
            "row",
            _cell_value(cell, "row_idx", _cell_value(cell, "row_i", _cell_value(cell, "r", None))),
        )
        cell_col = _cell_value(
            cell,
            "col",
            _cell_value(cell, "col_idx", _cell_value(cell, "col_i", _cell_value(cell, "c", None))),
        )
        if cell_row is None or cell_col is None:
            continue

        try:
            row_index = int(cell_row)
            col_index = int(cell_col)
        except (TypeError, ValueError):
            continue

        while len(rows) <= row_index:
            rows.append([])
        row_cells = rows[row_index]
        while len(row_cells) <= col_index:
            row_cells.append("")

        row_cells[col_index] = _sanitize_text(
            _cell_value(
                cell,
                "text",
                _cell_value(cell, "content", _cell_value(cell, "text_content", "")),
            )
        )

    return rows


def _extract_text_from_cell_clip(page, clip):
    if page is None or not clip:
        return ""

    try:
        x0, y0, x1, y1 = [float(v) for v in clip]
    except (TypeError, ValueError):
        return ""

    if x1 <= x0 or y1 <= y0:
        return ""

    try:
        rect = pymupdf.Rect(x0, y0, x1, y1)
    except Exception:
        return ""

    try:
        cell_text = page.get_text("text", clip=rect)
    except Exception:
        cell_text = ""

    normalized = _normalize_line(cell_text or "")
    if normalized:
        return normalized

    try:
        words = page.get_text("words", clip=rect)
    except Exception:
        return ""

    if not words:
        return ""

    rows = []
    line_tokens = []
    current_y = None
    line_tolerance = 1.2
    for word in sorted(words, key=lambda item: (_coerce_number(item[1], 0.0), _coerce_number(item[0], 0.0))):
        if len(word) < 5:
            continue

        try:
            x0w, y0w, _, _, value = word[:5]
        except Exception:
            continue

        text = _normalize_line(_sanitize_text(value))
        if not text:
            continue

        try:
            y0w = float(y0w)
            x0w = float(x0w)
        except (TypeError, ValueError):
            continue

        if current_y is None or abs(y0w - current_y) > line_tolerance:
            if line_tokens:
                rows.append(" ".join(token for _, token in sorted(line_tokens, key=lambda item: item[0])))
            line_tokens = [(x0w, text)]
            current_y = y0w
        else:
            line_tokens.append((x0w, text))

    if line_tokens:
        rows.append(" ".join(token for _, token in sorted(line_tokens, key=lambda item: item[0])))

    return _normalize_line(" ".join(rows))


def _reextract_table_rows(page, table, row_count=None, col_count=None):
    components = _infer_table_components(table, row_count=row_count, col_count=col_count)
    row_lines = components.get("row_lines") or []
    vertical_lines = components.get("vertical_lines") or []
    if len(row_lines) < 2 or len(vertical_lines) < 2:
        return [], components

    row_positions = []
    for line in row_lines:
        if not isinstance(line, dict):
            continue
        y0 = _coerce_number(line.get("y0"))
        y1 = _coerce_number(line.get("y1"))
        if y0 is None or y1 is None:
            continue
        row_positions.append(min(y0, y1))

    col_positions = []
    for line in vertical_lines:
        if not isinstance(line, dict):
            continue
        x0 = _coerce_number(line.get("x0"))
        x1 = _coerce_number(line.get("x1"))
        if x0 is None or x1 is None:
            continue
        col_positions.append(min(x0, x1))

    row_positions = _merge_positions(sorted(set(row_positions)), 1.0)
    col_positions = _merge_positions(sorted(set(col_positions)), 1.0)
    if len(row_positions) < 2 or len(col_positions) < 2:
        return [], components

    if row_count:
        target_rows = int(max(row_count, 0))
        if target_rows >= 1 and len(row_positions) > target_rows + 1:
            row_positions = row_positions[: target_rows + 1]

    if col_count:
        target_cols = int(max(col_count, 0))
        if target_cols >= 1 and len(col_positions) > target_cols + 1:
            col_positions = col_positions[: target_cols + 1]

    rows = []
    for row_index in range(len(row_positions) - 1):
        y0 = row_positions[row_index]
        y1 = row_positions[row_index + 1]
        if y1 <= y0:
            continue

        row_cells = []
        for col_index in range(len(col_positions) - 1):
            x0 = col_positions[col_index]
            x1 = col_positions[col_index + 1]
            if x1 <= x0:
                row_cells.append("")
                continue

            cell_text = _extract_text_from_cell_clip(page, (x0, y0, x1, y1))
            row_cells.append(cell_text)

        if any(cell for cell in row_cells):
            rows.append(row_cells)

    return rows, components


def _build_shape_grid_table(shape_lines, lines, page_rect, page_no, source, rotation=0, debug=False):
    shape_lines = [line for line in (shape_lines or []) if line.get("type") == "shape-line"]
    if not shape_lines or not lines:
        return []

    spans = _collect_span_cells(lines)
    if not spans:
        return []

    tx0 = min(span["x0"] for span in spans)
    ty0 = min(span["y0"] for span in spans)
    tx1 = max(span["x1"] for span in spans)
    ty1 = max(span["y1"] for span in spans)
    pad = 30.0
    span_bbox = (tx0 - pad, ty0 - pad, tx1 + pad, ty1 + pad)

    page_width = float(page_rect.width) if page_rect else (tx1 - tx0 if tx1 > tx0 else 0.0)
    page_height = float(page_rect.height) if page_rect else (ty1 - ty0 if ty1 > ty0 else 0.0)
    line_min_len = max(5.0, min(page_width, page_height) * 0.02 or 0.0)

    horizontal = []
    vertical = []
    for line in shape_lines:
        x0 = line.get("x0")
        y0 = line.get("y0")
        x1 = line.get("x1")
        y1 = line.get("y1")
        if not all(v is not None for v in (x0, y0, x1, y1)):
            continue
        try:
            x0 = float(x0)
            y0 = float(y0)
            x1 = float(x1)
            y1 = float(y1)
        except (TypeError, ValueError):
            continue

        length = _segment_length(x0, y0, x1, y1)
        if length < line_min_len:
            continue

        orientation = line.get("orientation")
        intersects_text = not (
            max(x0, x1) < span_bbox[0]
            or min(x0, x1) > span_bbox[2]
            or max(y0, y1) < span_bbox[1]
            or min(y0, y1) > span_bbox[3]
        )
        if not intersects_text:
            continue

        if orientation == "horizontal" or abs(y1 - y0) < 3.0:
            horizontal.append({"y": (y0 + y1) / 2.0, "x0": min(x0, x1), "x1": max(x0, x1)})
        elif orientation == "vertical" or abs(x1 - x0) < 3.0:
            vertical.append({"x": (x0 + x1) / 2.0, "y0": min(y0, y1), "y1": max(y0, y1)})

    if len(horizontal) < 2 or len(vertical) < 2:
        return []

    tol = max(2.0, _estimate_row_tolerance(lines))
    h_positions = _merge_numeric([entry["y"] for entry in horizontal], tol)
    v_positions = _merge_numeric([entry["x"] for entry in vertical], tol)

    if len(h_positions) < 2 or len(v_positions) < 2:
        return []

    # keep only drawing lines near the extracted text bounding area
    h_positions = [
        pos for pos in h_positions if ty0 - pad <= pos <= ty1 + pad
    ]
    v_positions = [
        pos for pos in v_positions if tx0 - pad <= pos <= tx1 + pad
    ]
    if len(h_positions) < 2 or len(v_positions) < 2:
        return []

    cand_x0 = min(v_positions)
    cand_x1 = max(v_positions)
    cand_y0 = min(h_positions)
    cand_y1 = max(h_positions)
    if cand_x1 <= cand_x0 or cand_y1 <= cand_y0:
        return []

    shape_row_lines = [
        {"orientation": "horizontal", "x0": cand_x0, "y0": y, "x1": cand_x1, "y1": y}
        for y in h_positions
    ]
    shape_vertical_lines = [
        {"orientation": "vertical", "x0": x, "y0": cand_y0, "x1": x, "y1": cand_y1}
        for x in v_positions
    ]

    tables = _build_text_grid_table(
        lines,
        page_rect,
        page_no,
        source,
        rotation=rotation,
        debug=debug,
        bbox=(cand_x0, cand_y0, cand_x1, cand_y1),
        min_rows=2,
        min_row_candidates=2,
        min_support_ratio=0.2,
        infer_method="shape-grid",
        override_row_lines=shape_row_lines,
        override_vertical_lines=shape_vertical_lines,
    )
    if not tables:
        return []

    result = []
    for table in tables:
        if table.get("bbox") is None:
            table["bbox"] = [cand_x0, cand_y0, cand_x1, cand_y1]
        table["source"] = source
        table["components"] = table.get("components", {})
        table["components"].setdefault("bbox", table.get("bbox"))
        table["components"]["row_lines"] = table.get("row_lines", [])
        table["components"]["vertical_lines"] = table.get("vertical_lines", [])
        result.append(table)

    return result


def _extract_page_tables(
    page,
    page_no,
    source,
    lines=None,
    shape_lines=None,
    debug=False,
    table_mode="auto",
):
    try:
        has_find_tables = hasattr(page, "find_tables")
    except Exception:
        has_find_tables = False

    if not has_find_tables:
        _LOGGER.warning("Page has no table detection API: source=%s page=%s", source, page_no)
        return []

    strategies = [("default", {})]
    if table_mode in ("auto", "lines"):
        strategies.append(("lines", {"vertical_strategy": "lines", "horizontal_strategy": "lines"}))
    if table_mode in ("auto", "text"):
        strategies.append(("text", {"vertical_strategy": "text", "horizontal_strategy": "text"}))
    if table_mode not in ("auto", "default", "lines", "text"):
        table_mode = "auto"

    selected = None
    selected_label = None
    fallback_selected = None
    fallback_label = None
    for label, options in strategies:
        try:
            tables = page.find_tables(**options)
        except TypeError:
            if debug:
                _LOGGER.debug(
                    "find_tables options unsupported, skipping: source=%s page=%s strategy=%s options=%s",
                    source,
                    page_no,
                    label,
                    options,
                )
            continue
        except Exception:
            if debug:
                _LOGGER.warning(
                    "find_tables() failed: source=%s page=%s strategy=%s options=%s",
                    source,
                    page_no,
                    label,
                    options,
                    exc_info=True,
                )
            continue

        rows = getattr(tables, "tables", None)
        count = len(rows or [])
        if debug:
            _LOGGER.debug(
                "find_tables returned %s table(s): source=%s page=%s strategy=%s options=%s",
                count,
                source,
                page_no,
                label,
                options,
            )
        if count:
            selected = tables
            selected_label = label
            break

        if fallback_selected is None:
            fallback_selected = tables
            fallback_label = label

        if table_mode != "auto":
            selected = tables
            selected_label = label
            break

    if selected is None:
        selected = fallback_selected
        selected_label = fallback_label

    if not selected or not getattr(selected, "tables", None):
        inferred = []
        inferred.extend(
            _build_shape_grid_table(
                shape_lines=shape_lines or [],
                lines=lines or [],
                page_rect=getattr(page, "rect", None),
                page_no=page_no,
                source=source,
                rotation=page.rotation,
                debug=debug,
            )
        )
        if not inferred:
            inferred = _build_text_grid_table(
                lines=lines or [],
                page_rect=getattr(page, "rect", None),
                page_no=page_no,
                source=source,
                rotation=page.rotation,
                debug=debug,
            )
        if inferred:
            if debug:
                _LOGGER.debug(
                    "Text-grid fallback created %s table(s): source=%s page=%s",
                    len(inferred),
                    source,
                    page_no,
                )
            return inferred

        if debug:
            _LOGGER.info(
                "No tables found on page after strategies: source=%s page=%s",
                source,
                page_no,
            )
        return []

    if _LOGGER.isEnabledFor(logging.INFO):
        _LOGGER.info(
            "Detected tables on page: source=%s page=%s count=%s",
            source,
            page_no,
            len(selected.tables),
        )
    elif debug:
        _LOGGER.debug(
            "Using tables strategy %s on page=%s table_count=%s",
            selected_label,
            page_no,
            len(selected.tables),
        )

    table_items = []
    for table_index, table in enumerate(selected.tables, start=1):
        bbox = getattr(table, "bbox", None)
        if not bbox:
            _LOGGER.warning(
                "Table %s on page %s has no bbox and was skipped: source=%s",
                table_index,
                page_no,
                source,
            )
            continue

        if debug:
            _LOGGER.debug(
                "Table parse: source=%s page=%s table=%s strategy=%s",
                source,
                page_no,
                table_index,
                selected_label,
            )

        rows = []
        cells = getattr(table, "cells", None) or []

        try:
            rows = table.extract()
        except Exception:
            if debug:
                _LOGGER.warning(
                    "table.extract() failed, falling back to cell text: source=%s page=%s table=%s",
                    source,
                    page_no,
                    table_index,
                    exc_info=True,
                )
                rows = []
            for cell in cells:
                row = getattr(cell, "row", None)
                col = getattr(cell, "col", None)
                if row is None or col is None:
                    continue
                while len(rows) <= row:
                    rows.append([])
                row_cells = rows[row]
                while len(row_cells) <= col:
                    row_cells.append("")
                row_cells[col] = _sanitize_text(
                    getattr(cell, "text", "") or getattr(cell, "content", "") or ""
                )

        if not rows and cells:
            rows = _rows_from_cells(cells)

        if not rows:
            # no table content recovered from detector
            continue

        table_rows = len(rows)
        table_cols = 0
        for row in rows:
            if isinstance(row, (list, tuple)):
                table_cols = max(table_cols, len(row))

        components = None
        re_rows, components = _reextract_table_rows(
            page,
            table,
            row_count=table_rows,
            col_count=table_cols,
        )
        if re_rows:
            rows = re_rows
        else:
            components = None

        row_count = 0
        col_count = 0
        row_texts = []
        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue
            row_count += 1
            col_count = max(col_count, len(row))
            row_texts.append(" | ".join(_sanitize_text(cell or "") for cell in row))
        if row_count == 0:
            if debug:
                _LOGGER.debug(
                    "No readable rows after table re-extraction: source=%s page=%s table=%s",
                    source,
                    page_no,
                    table_index,
                )
            continue

        if components is None:
            components = _infer_table_components(table, row_count=row_count, col_count=col_count)
        elif row_count != table_rows or col_count != table_cols:
            components = _infer_table_components(table, row_count=row_count, col_count=col_count)
        if debug:
            _LOGGER.debug(
                "Table %s on page %s row_count=%s col_count=%s",
                table_index,
                page_no,
                row_count,
                col_count,
            )
            _LOGGER.debug(
                "Table %s geometry: source=%s page=%s row_lines=%s col_lines=%s",
                table_index,
                source,
                page_no,
                len(components.get("row_lines", [])),
                len(components.get("vertical_lines", [])),
            )

        x0, y0, x1, y1 = [float(v) for v in bbox]
        table_items.append(
            {
                "page": page_no,
                "start_page": page_no,
                "table_no": table_index,
                "bbox": [x0, y0, x1, y1],
                "row_count": row_count,
                "col_count": col_count,
                "rotation": int(page.rotation or 0),
                "rows_text": row_texts,
                "x": _round_float((x0 + x1) / 2.0),
                "y": _round_float((y0 + y1) / 2.0),
                "page_width": float(page.rect.width),
                "page_height": float(page.rect.height),
                "text": "\n".join(row_texts),
                "row_lines": list(components.get("row_lines", [])),
                "vertical_lines": list(components.get("vertical_lines", [])),
                    "components": {
                        "bbox": components.get("bbox"),
                        "row_lines": list(components.get("row_lines", [])),
                        "vertical_lines": list(components.get("vertical_lines", [])),
                    },
                    "infer_method": "find-tables",
                    "source": source,
                }
            )

    return table_items


def _table_row_signature(row):
    return re.sub(r"\s+", " ", (row or "").strip().lower())


def _bbox_overlap_ratio(a_bbox, b_bbox):
    ax0, ay0, ax1, ay1 = a_bbox
    bx0, by0, bx1, by1 = b_bbox

    x_overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    y_overlap = max(0.0, min(ay1, by1) - max(ay0, by0))
    if x_overlap <= 0 or y_overlap <= 0:
        return 0.0, 0.0

    a_area = max(0.0, (ax1 - ax0) * (ay1 - ay0))
    b_area = max(0.0, (bx1 - bx0) * (by1 - by0))
    if a_area <= 0 or b_area <= 0:
        return 0.0, 0.0

    overlap_area = x_overlap * y_overlap
    return overlap_area / min(a_area, b_area), overlap_area / (a_area + b_area - overlap_area)


def _is_cross_page_split(prev_table, next_table):
    prev_height = prev_table.get("page_height") or 0.0
    next_height = next_table.get("page_height") or 0.0
    if prev_height <= 0.0 or next_height <= 0.0:
        return False

    prev_ratio_from_bottom = 1.0 - (prev_table.get("bbox", [0, 0, 0, 0])[3] / prev_height)
    next_ratio_from_top = next_table.get("bbox", [0, 0, 0, 0])[1] / next_height

    # table continues near page boundary (bottom of prev and top of next)
    return prev_ratio_from_bottom <= 0.35 and next_ratio_from_top <= 0.35


def _merge_cross_page_tables(tables):
    if len(tables) <= 1:
        return tables

    sorted_tables = sorted(
        tables,
        key=lambda item: (item.get("page", 0), item.get("bbox", [0, 0, 0, 0])[1], item.get("table_no", 0)),
    )
    merged = []
    current = None

    for table in sorted_tables:
        if current is None:
            current = dict(table)
            current["current_end_page"] = current.get("page")
            current["pages"] = [current.get("start_page", current.get("page"))]
            continue

        same_col = current.get("col_count") == table.get("col_count")
        same_rotation = current.get("rotation") == table.get("rotation")
        current_end_page = current.get("current_end_page", current.get("page", 0))
        adjacent_page = table.get("page", 0) == current_end_page + 1
        if not (same_col and same_rotation and adjacent_page):
            merged.append(current)
            current = dict(table)
            current["current_end_page"] = current.get("page")
            current["pages"] = [current.get("start_page", current.get("page"))]
            continue

        overlap_ratio, _ = _bbox_overlap_ratio(current.get("bbox", [0, 0, 0, 0]), table.get("bbox", [0, 0, 0, 0]))
        if overlap_ratio < 0.1 or not _is_cross_page_split(current, table):
            merged.append(current)
            current = dict(table)
            current["pages"] = [current.get("start_page", current.get("page"))]
            continue

        current_rows = list(current.get("rows_text", []))
        next_rows = list(table.get("rows_text", []))
        if current_rows and next_rows:
            header_signature = _table_row_signature(current_rows[0])
            next_first = _table_row_signature(next_rows[0])
            if header_signature and next_first and header_signature == next_first:
                next_rows = next_rows[1:]

        current_rows.extend(next_rows)
        current["rows_text"] = current_rows
        current["text"] = "\n".join(current_rows)
        current["row_count"] = len(current_rows)
        current["row_lines"] = list(current.get("row_lines", [])) + list(table.get("row_lines", []))
        current["vertical_lines"] = list(current.get("vertical_lines", [])) + list(table.get("vertical_lines", []))
        current["components"] = {
            "bbox": current.get("components", {}).get("bbox"),
            "row_lines": list(current.get("row_lines", [])),
            "vertical_lines": list(current.get("vertical_lines", [])),
        }
        current["page_end"] = table.get("page")
        current["current_end_page"] = table.get("page")
        if "pages" not in current:
            current["pages"] = [current.get("start_page", current.get("page", 0))]
        current["pages"].append(table.get("page"))
        current["pages"] = sorted(set(current["pages"]))

        continue

    if current is not None:
        merged.append(current)

    # keep page field as the first page
    for item in merged:
        item["page"] = item.get("start_page", item.get("page"))
        if "current_end_page" in item:
            item["page_end"] = max(item.get("start_page", item.get("page", 0)), item.get("current_end_page", item.get("page", 0)))
            del item["current_end_page"]
    return merged


def _extract_page_lines(
    page,
    page_no,
    source,
    header_ratio,
    footer_ratio,
    preserve_newlines=False,
    strip_markdown_lines=False,
    debug=False,
):
    page_data = page.get_text("dict", sort=True)
    page_words = page.get_text("words")
    page_rect = page.rect
    rotation = int(page.rotation or 0)
    axis = _rotation_axis(rotation)
    lines = []
    line_no = 0

    for block in page_data.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            spans = line.get("spans", []) or []
            if not spans:
                continue

            line_no += 1
            line_bbox = _line_bbox(line, spans)
            location = _classify_region(page_rect, line_bbox, rotation, header_ratio, footer_ratio)
            x, y = _line_center(line_bbox)
            x_ratio = x / float(page_rect.width) if page_rect and float(page_rect.width) > 0 else None
            y_ratio = y / float(page_rect.height) if page_rect and float(page_rect.height) > 0 else None

            raw_parts = []
            markdown_parts = []
            max_size = 0.0
            font_counts = Counter()
            colors = []
            span_count = 0
            span_items = []

            for span_idx, span in enumerate(spans, start=1):
                raw = (span.get("text") or "").replace("\xa0", " ")
                raw = _sanitize_text(
                    raw,
                    context={
                        "source": source,
                        "page": page_no,
                        "line": line_no,
                        "span": span_idx,
                    },
                )
                if not raw.strip():
                    continue

                styled = _span_to_markdown(raw, span)
                if preserve_newlines:
                    raw_parts.append(raw)
                else:
                    _append_span(raw_parts, raw)
                _append_span(markdown_parts, styled)
                max_size = max(max_size, float(span.get("size") or 0.0))
                span_count += 1

                font = (span.get("font") or "").strip()
                if font:
                    font_counts[font] += 1

                color = span.get("color")
                if color is not None:
                    colors.append(color)
                span_bbox = span.get("bbox")
                if isinstance(span_bbox, (list, tuple)) and len(span_bbox) == 4:
                    try:
                        x0, y0, x1, y1 = [float(v) for v in span_bbox]
                        span_items.append(
                            {
                                "text": raw,
                                "bbox": [x0, y0, x1, y1],
                                "size": _round_float(span.get("size", 0.0)),
                                "font": span.get("font"),
                                "color": color,
                            }
                        )
                    except (TypeError, ValueError):
                        pass

            raw_text = "".join(raw_parts)
            if not preserve_newlines:
                raw_text = _normalize_line(raw_text)
            if not raw_text:
                continue
            if strip_markdown_lines and _is_markdown_like(raw_text):
                if debug:
                    _LOGGER.debug(
                        "Skipped markdown-like line: source=%s page=%s line=%s text=%r",
                        source,
                        page_no,
                        line_no,
                        raw_text[:120],
                    )
                continue

            baseline_axis = axis
            baseline_value = _line_baseline(spans, baseline_axis)
            page_axis_size = float(page_rect.width) if baseline_axis == "x" else float(page_rect.height)
            baseline_ratio = baseline_value / page_axis_size if page_axis_size > 0 else None
            line_rotation = (int(rotation) + round(_line_tilt_angle(line, spans))) % 360

            if debug:
                text_snippet = raw_text if len(raw_text) <= 120 else raw_text[:117] + "..."
                _LOGGER.debug(
                    "raw line: source=%s page=%s line=%s x=%s y=%s rotation=%s text=%r",
                    source,
                    page_no,
                    line_no,
                    _round_float(x),
                    _round_float(y),
                    line_rotation,
                    text_snippet,
                )

            dominant_font = font_counts.most_common(1)[0][0] if font_counts else None
            if span_count <= 1:
                for word in page_words or []:
                    if len(word) < 5:
                        continue
                    wx0, wy0, wx1, wy1, wtext = word[:5]
                    wtext = (wtext or "").strip()
                    if not wtext:
                        continue
                    if wx1 <= line_bbox[0] or wx0 >= line_bbox[2] or wy1 <= line_bbox[1] or wy0 >= line_bbox[3]:
                        continue
                    span_items.append(
                        {
                            "text": _normalize_line(wtext),
                            "bbox": [float(wx0), float(wy0), float(wx1), float(wy1)],
                            "size": _round_float(max_size),
                            "font": None,
                            "color": colors[0] if colors else None,
                        }
                    )

            lines.append(
                {
                    "raw": raw_text,
                    "markdown": _normalize_line("".join(markdown_parts)) or raw_text,
                    "size": _round_float(max_size),
                    "bbox": line_bbox,
                    "location": location,
                    "rotation": line_rotation,
                    "is_watermark_rotation": _is_watermark_rotation(line_rotation),
                    "rotation_axis": baseline_axis,
                    "page": page_no,
                    "line": line_no,
                    "span_count": span_count,
                    "font_family": dominant_font,
                    "font_family_map": dict(font_counts),
                    "color": colors[0] if colors else None,
                    "baseline": {
                        "axis": baseline_axis,
                        "value": baseline_value,
                        "ratio": baseline_ratio,
                    },
                    "source": source,
                    "position": {
                        "baseline": {
                            "axis": baseline_axis,
                            "value": baseline_value,
                            "ratio": baseline_ratio,
                        },
                        "x": x,
                        "y": y,
                        "x_ratio": x_ratio,
                        "y_ratio": y_ratio,
                    },
                    "spans": span_items,
                    "row_no": 0,
                }
            )

    _assign_row_ids(lines)
    return lines


def _estimate_body_font_size(lines):
    sizes = [line.get("size", 0.0) for line in lines if line.get("size")]
    if not sizes:
        return 12.0
    return float(Counter(sizes).most_common(1)[0][0])


def _heading_level(size, body_size):
    if body_size <= 0:
        return 0

    ratio = size / body_size
    if ratio >= 1.7:
        return 1
    if ratio >= 1.45:
        return 2
    if ratio >= 1.25:
        return 3
    return 0


def _line_to_markdown(line_text, line_size, body_size):
    text = _normalize_line(line_text)
    if not text:
        return ""

    cleaned = re.sub(r"^\s*([•●▪◦·]|\*|-)\s+", "- ", text)
    heading = _heading_level(line_size, body_size)
    if heading:
        return f"{'#' * heading} {cleaned}"
    return cleaned


def _line_to_payload(line, text, target_region, removed_reason, removed):
    position = line.get("position", {})
    return {
        "page": line["page"],
        "line_no": line["line"],
        "region": target_region,
        "text": text,
        "rotation": line["rotation"],
        "x": position.get("x"),
        "y": position.get("y"),
        "removed": removed,
        "removed_reason": removed_reason,
    }


def _span_tilt_angle(span):
    direction = span.get("dir")
    if direction and len(direction) == 2:
        dx, dy = direction
        if dx or dy:
            return float(math.degrees(math.atan2(-dy, dx)))
    return 0.0


def _span_insert_point(span):
    origin = _to_point(span.get("origin"))
    if origin is not None:
        return origin

    bbox = span.get("bbox")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    return (x0, y1)


def _draw_raw_spans_on_page(out_page, spans, source, page_no, debug=False):
    for span in spans or []:
        if not isinstance(span, dict):
            continue

        text = (span.get("text") or "").replace("\xa0", " ")
        if not text:
            continue

        point = _span_insert_point(span)
        if point is None:
            continue

        size = _coerce_number(span.get("size"), 0.0)
        if not size or size <= 0:
            size = 11.0
        color = _to_rgb_color(span.get("color"), default=(0.0, 0.0, 0.0))
        rotation = _span_tilt_angle(span)
        font = (span.get("font") or "helv").strip() or "helv"

        inserted = False
        for font_name in (font, "helv", "sans-serif"):
            for rotate_value in (rotation, 0.0):
                try:
                    out_page.insert_text(
                        point,
                        text,
                        fontsize=size,
                        fontname=font_name,
                        color=color,
                        rotate=rotate_value,
                    )
                    inserted = True
                    break
                except Exception:
                    continue
            if inserted:
                break

        if not inserted and debug:
            _LOGGER.debug(
                "Preview text span insert failed: source=%s page=%s font=%r size=%s rotation=%s text=%r",
                source,
                page_no,
                font,
                size,
                _round_float(rotation),
                text[:120],
            )


def _is_preview_watermark_line(
    line,
    watermark_angle,
    watermark_tolerance,
    strip_body_rotation=False,
    remove_markdown_lines=False,
):
    raw_text = _normalize_line(line.get("raw") or "")
    if not raw_text:
        return False, "empty-text"

    rotation = line.get("rotation")
    location = line.get("location", "body")
    if _is_rotation_match(rotation, watermark_angle, watermark_tolerance) and (
        location == "body" or strip_body_rotation
    ):
        return True, "watermark-rotation"

    if remove_markdown_lines and _is_markdown_like(raw_text):
        return True, "markdown"

    return False, None


def _summarize_items(items):
    if not items:
        return {
            "count": 0,
            "kept_count": 0,
            "removed_count": 0,
            "size": {
                "min": None,
                "max": None,
                "avg": None,
            },
            "baseline": {
                "min": None,
                "max": None,
                "avg": None,
                "ratio": {
                    "min": None,
                    "max": None,
                    "avg": None,
                },
                "dominant_axis": None,
            },
            "rotation": {
                "dominant_axis": None,
                "axis_counts": {},
                "counts": {},
            },
            "bbox_union": None,
            "top_fonts": [],
        }

    kept = [i for i in items if not i["removed"]]
    removed = [i for i in items if i["removed"]]
    sizes = [float(item.get("size", 0.0)) for item in items if item.get("size")]
    baselines = []
    baseline_ratios = []
    rotation_counts = Counter()
    axis_counts = Counter()

    for item in items:
        baseline = item.get("position", {}).get("baseline", {})
        baseline_value = baseline.get("value")
        baseline_ratio = baseline.get("ratio")

        if baseline_value is not None:
            baselines.append(float(baseline_value))

        if baseline_ratio is not None:
            baseline_ratios.append(float(baseline_ratio))

        rotation = item.get("rotation")
        if rotation is not None:
            rotation_counts[int(rotation)] += 1

        axis = baseline.get("axis")
        if axis:
            axis_counts[str(axis)] += 1

    rotation_dominant_axis = axis_counts.most_common(1)[0][0] if axis_counts else None

    bbox_union = None
    for item in items:
        bbox = item.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        if len(bbox) != 4:
            continue
        if bbox_union is None:
            bbox_union = [float(v) for v in bbox]
        else:
            bbox_union = [
                min(bbox_union[0], float(bbox[0])),
                min(bbox_union[1], float(bbox[1])),
                max(bbox_union[2], float(bbox[2])),
                max(bbox_union[3], float(bbox[3])),
            ]

    font_counter = Counter(item.get("font_family") for item in items if item.get("font_family"))

    return {
        "count": len(items),
        "kept_count": len(kept),
        "removed_count": len(removed),
        "size": {
            "min": min(sizes) if sizes else None,
            "max": max(sizes) if sizes else None,
            "avg": sum(sizes) / len(sizes) if sizes else None,
        },
        "baseline": {
            "min": min(baselines) if baselines else None,
            "max": max(baselines) if baselines else None,
            "avg": sum(baselines) / len(baselines) if baselines else None,
            "ratio": {
                "min": min(baseline_ratios) if baseline_ratios else None,
                "max": max(baseline_ratios) if baseline_ratios else None,
                "avg": sum(baseline_ratios) / len(baseline_ratios) if baseline_ratios else None,
            },
            "dominant_axis": rotation_dominant_axis,
        },
        "rotation": {
            "dominant_axis": rotation_dominant_axis,
            "axis_counts": dict(axis_counts),
            "counts": {str(k): v for k, v in rotation_counts.items()},
        },
        "bbox_union": bbox_union,
        "top_fonts": font_counter.most_common(3),
    }


def _collect_anomalies(lines, sections, region_summary, removed, body_size, strip_headers, strip_footers, header_ratio, footer_ratio):
    anomalies = []

    if not lines:
        return [{"type": "empty-page", "message": "No text lines were extracted."}]

    total = removed.get("total", len(lines))
    rotation = lines[0].get("rotation", 0) if lines else 0
    if rotation not in _SUPPORTED_ROTATIONS:
        anomalies.append({
            "type": "rotation-unsupported",
            "message": "Page rotation is not 0/90/180/270. Region split may be unreliable.",
            "rotation": rotation,
        })

    location_count = {
        "header": sum(1 for line in lines if line.get("location") == "header"),
        "footer": sum(1 for line in lines if line.get("location") == "footer"),
        "body": sum(1 for line in lines if line.get("location") == "body"),
    }

    if strip_headers and location_count["header"] == 0 and total >= 2:
        anomalies.append(
            {
                "type": "no-header-detected",
                "message": f"No header location match with header_ratio={header_ratio}.",
                "ratio": header_ratio,
            }
        )

    if strip_footers and location_count["footer"] == 0 and total >= 2:
        anomalies.append(
            {
                "type": "no-footer-detected",
                "message": f"No footer location match with footer_ratio={footer_ratio}.",
                "ratio": footer_ratio,
            }
        )

    def _axis_counts(region_name):
        rotation = region_summary.get(region_name, {}).get("rotation", {})
        axes = rotation.get("axis_counts", {}) if isinstance(rotation, dict) else {}
        return axes

    def _baseline_summary(region_name):
        return region_summary.get(region_name, {}).get("baseline", {})

    def _rotation_axis_outlier(region_name, expected_axis):
        axes = _axis_counts(region_name)
        if not axes:
            return None
        if all(k == expected_axis for k in axes.keys()):
            return None

        sample = sections.get(region_name, {}).get("items", [])
        sample_item = sample[0] if sample else {}

        return {
            "type": f"{region_name}-rotation-axis-mixed",
            "message": f"{region_name.title()} lines use mixed rotation axes.",
            "axes": list(axes.keys()),
            "counts": axes,
            "expected_axis": expected_axis,
            "sample": {
                "line_no": sample_item.get("line_no"),
                "rotation": sample_item.get("rotation"),
                "rotation_axis": sample_item.get("rotation_axis"),
                "region": sample_item.get("region"),
                "snippet": _surrounding_snippet(sample_item.get("text", ""), max(0, len(sample_item.get("text", "")) // 2)),
            },
        }

    expected_axis = _rotation_axis(lines[0].get("rotation", 0)) if lines else "y"
    for region_name in ("header", "footer", "watermark"):
        outlier = _rotation_axis_outlier(region_name, expected_axis)
        if outlier:
            anomalies.append(outlier)

    for region_name, direction in (
        ("header", "top"),
        ("footer", "bottom"),
    ):
        baseline = _baseline_summary(region_name)
        ratios = baseline.get("ratio", {})
        if ratios:
            ratio_min = ratios.get("min")
            ratio_max = ratios.get("max")
        else:
            ratio_min = None
            ratio_max = None

        if region_name == "header":
            if ratio_min is None and location_count[region_name] > 0:
                anomalies.append(
                    {
                        "type": f"{region_name}-baseline-missing",
                        "message": f"Header baseline ratio missing for {location_count['header']} line(s).",
                        "region": region_name,
                        "count": location_count[region_name],
                    }
                )
            elif ratio_max is not None and ratio_max > header_ratio * 2:
                anomalies.append(
                    {
                        "type": f"{region_name}-unexpected-baseline",
                        "message": f"Header baseline is farther from top than expected.",
                        "region": region_name,
                        "baseline_ratio_max": round(ratio_max, 4),
                        "expected_max_ratio": round(header_ratio * 2, 4),
                        "direction": direction,
                    }
                )
        else:
            if ratio_min is None and location_count[region_name] > 0:
                anomalies.append(
                    {
                        "type": f"{region_name}-baseline-missing",
                        "message": f"Footer baseline ratio missing for {location_count['footer']} line(s).",
                        "region": region_name,
                        "count": location_count[region_name],
                    }
                )
            elif ratio_min is not None and ratio_min < 1 - footer_ratio * 2:
                anomalies.append(
                    {
                        "type": f"{region_name}-unexpected-baseline",
                        "message": f"Footer baseline is farther from bottom than expected.",
                        "region": region_name,
                        "baseline_ratio_min": round(ratio_min, 4),
                        "expected_min_ratio": round(1 - footer_ratio * 2, 4),
                        "direction": direction,
                    }
                )

    if total >= 4 and removed.get("watermark", 0) >= max(1, math.ceil(total * 0.5)):
        anomalies.append(
            {
                "type": "aggressive-watermark-filter",
                "message": "More than 50% of lines were removed as watermark.",
                "count": removed.get("watermark", 0),
                "total": total,
            }
        )

    if body_size > 0:
        for line in lines:
            line_size = float(line.get("size", 0.0) or 0.0)
            if not line_size:
                continue

            ratio = line_size / body_size if body_size > 0 else 0.0
            location = line.get("location")

            if location == "body" and ratio >= 2.4:
                raw = line.get("raw") or ""
                anomalies.append(
                    {
                        "type": "body-font-outlier",
                        "message": "Body line has very large font compared with body median.",
                        "line": line.get("line"),
                        "size_ratio": round(ratio, 2),
                        "line_no": line.get("line"),
                        "snippet": _surrounding_snippet(raw, max(0, len(raw) // 2)),
                    }
                )
                break

    # detect exceptional watermark-like body lines
    for section_name in ("watermark", "header", "footer"):
        entries = sections.get(section_name, {}).get("items", [])
        if any(item.get("removed") and item.get("removed_reason") == "watermark-repeat" for item in entries):
            anomalies.append(
                {
                    "type": f"{section_name}-pattern-removal",
                    "message": f"One or more lines in '{section_name}' were removed by watermark pattern or repetition rules.",
                }
            )
            break

    return anomalies


def _build_sections(lines, body_size, repeated_watermarks, compiled_patterns, strip_headers, strip_footers, strip_watermarks):
    sections = {
        "header": {"items": [], "text": ""},
        "footer": {"items": [], "text": ""},
        "watermark": {"items": [], "text": ""},
        "body": {"items": [], "text": ""},
    }

    removed = {
        "header": 0,
        "footer": 0,
        "watermark": 0,
        "pattern": 0,
        "kept": 0,
        "total": len(lines),
    }

    kept_text_lines = []
    for line in lines:
        raw = line["raw"]
        markdown = _line_to_markdown(raw, line["size"], body_size)
        if not markdown:
            continue

        target_region = line["location"]
        removed_reason = None
        is_removed = False

        if compiled_patterns:
            _, match_pos = _first_pattern_hit(raw, compiled_patterns)
            if match_pos is not None:
                target_region = "watermark"
                removed_reason = "watermark-pattern"
                is_removed = True
                removed["watermark"] += 1
                removed["pattern"] += 1
                _LOGGER.warning(
                    "Removed by watermark-pattern: source=%s page=%s line=%s rotation=%s location=%s snippet=%s",
                    line.get("source"),
                    line.get("page"),
                    line.get("line"),
                    line.get("rotation"),
                    line.get("location"),
                    _surrounding_snippet(raw, match_pos),
                )
                payload = _line_to_payload(line, markdown, target_region, removed_reason, True)
                sections[target_region]["items"].append(payload)
                continue

        if strip_watermarks and line.get("is_watermark_rotation"):
            target_region = "watermark"
            removed_reason = "watermark-rotation"
            is_removed = True
            removed["watermark"] += 1
            _LOGGER.warning(
                "Removed by watermark-rotation: source=%s page=%s line=%s rotation=%s location=%s snippet=%s",
                line.get("source"),
                line.get("page"),
                line.get("line"),
                line.get("rotation"),
                line.get("location"),
                _surrounding_snippet(raw, max(0, len(raw) // 2)),
            )
            payload = _line_to_payload(line, markdown, target_region, removed_reason, True)
            sections[target_region]["items"].append(payload)
            continue

        repeated_key = _normalize_line(raw).casefold()
        if strip_watermarks and repeated_key in repeated_watermarks:
            target_region = "watermark"
            removed_reason = "watermark-repeat"
            is_removed = True
            removed["watermark"] += 1
            _LOGGER.warning(
                "Removed by watermark-detection: source=%s page=%s line=%s rotation=%s location=%s snippet=%s",
                line.get("source"),
                line.get("page"),
                line.get("line"),
                line.get("rotation"),
                line.get("location"),
                _surrounding_snippet(raw, max(0, len(raw) // 2)),
            )
            payload = _line_to_payload(line, markdown, target_region, removed_reason, True)
            sections[target_region]["items"].append(payload)
            continue

        if line["location"] == "header" and strip_headers:
            target_region = "header"
            removed_reason = "header"
            is_removed = True
            removed["header"] += 1
        elif line["location"] == "footer" and strip_footers:
            target_region = "footer"
            removed_reason = "footer"
            is_removed = True
            removed["footer"] += 1

        payload = _line_to_payload(line, markdown, target_region, removed_reason, is_removed)
        sections[target_region]["items"].append(payload)

        if is_removed:
            sections[target_region]["text"] = sections[target_region]["text"] + markdown + "\n"
        else:
            kept_text_lines.append((line["line"], markdown))
            removed["kept"] += 1

    kept_lines = "\n".join(
        markdown for _, markdown in sorted(kept_text_lines, key=lambda pair: pair[0])
    )

    summary = {
        region: _summarize_items(section["items"]) for region, section in sections.items()
    }
    removed["total"] = len(lines)
    return sections, summary, removed, kept_lines


def _extract_pages(
    doc,
    source,
    header_ratio,
    footer_ratio,
    max_pages=None,
    pages=None,
    preserve_newlines=False,
    strip_markdown_lines=False,
    extract_tables=False,
    debug=False,
    table_debug=False,
    table_mode="auto",
    capture_raw_pages=False,
):
    extracted = []
    raw_pages = []
    page_numbers = _coerce_page_numbers(doc, pages, max_pages)
    for page_no in page_numbers:
        if page_no < 1 or page_no > doc.page_count:
            continue
        page = doc[page_no - 1]
        lines = _extract_page_lines(
            page,
            page_no,
            source,
            header_ratio,
            footer_ratio,
            preserve_newlines=preserve_newlines,
            strip_markdown_lines=strip_markdown_lines,
            debug=debug,
        )
        for line in lines:
            line["source"] = source
        table_lines = [line for line in lines if not line.get("is_watermark_rotation")]
        shape_lines = _extract_page_drawings(
            page,
            page_no,
            source,
            debug=table_debug or debug,
            header_ratio=header_ratio,
            footer_ratio=footer_ratio,
        )
        tables = []
        if extract_tables:
            tables = _extract_page_tables(
                page,
                page_no,
                source,
                lines=table_lines,
                shape_lines=shape_lines,
                debug=table_debug,
                table_mode=table_mode,
            )
        if capture_raw_pages:
            raw_pages.append(
                _extract_page_raw_payload(
                    page=page,
                    page_no=page_no,
                    source=source,
                    debug=debug,
                )
            )
        extracted.append((page_no, lines, tables, shape_lines))

    if capture_raw_pages:
        return extracted, raw_pages
    return extracted


def _to_json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, tuple, set)):
        return [_to_json_value(item) for item in value]

    if isinstance(value, dict):
        return {str(key): _to_json_value(val) for key, val in value.items()}

    if hasattr(value, "x0") and hasattr(value, "y0") and hasattr(value, "x1") and hasattr(value, "y1"):
        try:
            return [
                _round_float(float(value.x0)),
                _round_float(float(value.y0)),
                _round_float(float(value.x1)),
                _round_float(float(value.y1)),
            ]
        except (TypeError, ValueError, AttributeError):
            pass

    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
        try:
            return [_to_json_value(item) for item in value]
        except Exception:
            pass

    return str(value)


def _extract_page_raw_payload(page, page_no, source, debug=False):
    payload = {
        "type": "raw-page",
        "page": page_no,
        "source": source,
    }

    try:
        page_rect = page.rect
        payload["rect"] = _to_json_value(page_rect)
        payload["rotation"] = int(page.rotation or 0)
        payload["media_box"] = _to_json_value(getattr(page, "mediabox", page_rect))
    except Exception:
        if debug:
            _LOGGER.debug(
                "Failed page geometry extraction: source=%s page=%s",
                source,
                page_no,
                exc_info=True,
            )
        payload["rect"] = None
        payload["rotation"] = 0
        payload["media_box"] = None

    text_modes = (
        ("text", "text"),
        ("text_dict", "dict"),
        ("text_rawdict", "rawdict"),
        ("text_words", "words"),
    )
    for key, mode in text_modes:
        try:
            payload[key] = _to_json_value(page.get_text(mode))
        except Exception as exc:
            if debug:
                _LOGGER.debug(
                    "Failed page.get_text('%s'): source=%s page=%s error=%s",
                    mode,
                    source,
                    page_no,
                    exc,
                )
            payload[key] = {"error": str(exc)}

    for key, getter in (
        ("links", lambda: page.get_links()),
        ("drawings", lambda: page.get_drawings()),
        ("images", lambda: page.get_images(full=True)),
        ("blocks", lambda: page.get_text("blocks")),
    ):
        try:
            payload[key] = _to_json_value(getter())
        except Exception as exc:
            if debug:
                _LOGGER.debug(
                    "Failed to extract page field=%s: source=%s page=%s error=%s",
                    key,
                    source,
                    page_no,
                    exc,
                )
            payload[key] = {"error": str(exc)}

    return payload


def _write_image_only_page_pdf(
    source_path,
    page_no,
    output_path,
    strip_markdown_lines=False,
    header_ratio=0.08,
    footer_ratio=0.08,
):
    page_no = int(page_no)
    if page_no < 1:
        raise ValueError(f"Page number must be >=1: {page_no}")

    with pymupdf.open(source_path) as doc:
        if page_no > doc.page_count:
            raise ValueError(
                f"Page number out of range: {page_no} (total: {doc.page_count})"
            )

        page = doc[page_no - 1]
        markdown_rects = []
        if strip_markdown_lines:
            lines = _extract_page_lines(
                page,
                page_no,
                str(source_path),
                header_ratio,
                footer_ratio,
                preserve_newlines=False,
                strip_markdown_lines=False,
                debug=False,
            )
            for line in lines:
                raw_text = _normalize_line(line.get("raw") or "")
                if not raw_text or not _is_markdown_like(raw_text):
                    continue
                bbox = line.get("bbox")
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue
                try:
                    x0, y0, x1, y1 = [float(v) for v in bbox]
                except (TypeError, ValueError):
                    continue
                if x1 <= x0 or y1 <= y0:
                    continue
                markdown_rects.append(pymupdf.Rect(x0, y0, x1, y1))
        page_images = page.get_images(full=True)
        png_xref_to_stream = {}
        for item in page_images:
            if not item:
                continue
            xref = item[0]
            if xref in png_xref_to_stream:
                continue

            try:
                image_info = doc.extract_image(int(xref))
            except Exception:
                continue

            if (image_info.get("ext") or "").lower() != "png":
                continue

            image_bytes = image_info.get("image")
            if not isinstance(image_bytes, (bytes, bytearray)):
                continue
            png_xref_to_stream[xref] = bytes(image_bytes)

        rendered_pdf = pymupdf.open()
        out_page = rendered_pdf.new_page(width=page.rect.width, height=page.rect.height)
        for xref, image_bytes in png_xref_to_stream.items():
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                continue

            if not rects:
                continue
            for rect in rects:
                if rect is None:
                    continue
                out_page.insert_image(rect, stream=image_bytes)

        for rect in markdown_rects:
            try:
                out_page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
            except Exception:
                continue

        rendered_pdf.save(output_path, deflate=True, garbage=4)
        rendered_pdf.close()

    return str(Path(output_path))


def _write_preview_cleaned_page_pdf(
    source_path,
    page_no,
    output_path,
    watermark_angle=_WATERMARK_ROTATION_DEGREE,
    watermark_tolerance=_WATERMARK_ROTATION_TOLERANCE,
    header_ratio=0.08,
    footer_ratio=0.08,
    remove_markdown_lines=False,
    strip_body_rotation=False,
    debug=False,
):
    page_no = int(page_no)
    if page_no < 1:
        raise ValueError(f"Page number must be >=1: {page_no}")

    with pymupdf.open(source_path) as doc:
        if page_no > doc.page_count:
            raise ValueError(
                f"Page number out of range: {page_no} (total: {doc.page_count})"
            )

        page = doc[page_no - 1]
        source_text = str(source_path)
        raw_lines = _extract_page_lines(
            page=page,
            page_no=page_no,
            source=source_text,
            header_ratio=header_ratio,
            footer_ratio=footer_ratio,
            preserve_newlines=False,
            strip_markdown_lines=False,
            debug=debug,
        )

        removed_lines = []
        removed_line_numbers = set()
        removed_count = 0
        for line in raw_lines:
            should_remove, removed_reason = _is_preview_watermark_line(
                line=line,
                watermark_angle=watermark_angle,
                watermark_tolerance=watermark_tolerance,
                strip_body_rotation=strip_body_rotation,
                remove_markdown_lines=remove_markdown_lines,
            )
            if should_remove and debug:
                _LOGGER.debug(
                    "Preview removed element: source=%s page=%s line=%s location=%s reason=%s rotation=%s text=%r",
                    source_text,
                    page_no,
                    line.get("line"),
                    line.get("location"),
                    removed_reason,
                    line.get("rotation"),
                    _normalize_line(line.get("raw") or "")[:120],
                )
            if should_remove:
                removed_count += 1
                removed_line_numbers.add(line.get("line"))
                removed_lines.append(
                    {
                        "line_no": line.get("line"),
                        "rotation": line.get("rotation"),
                        "location": line.get("location"),
                        "bbox": line.get("bbox"),
                        "reason": removed_reason,
                        "text": line.get("raw"),
                    }
                )
                continue

        if debug:
            _LOGGER.debug(
                "Preview rebuild summary: source=%s page=%s kept_lines=%s removed_lines=%s",
                source_text,
                page_no,
                len(raw_lines) - removed_count,
                removed_count,
            )

        rendered_pdf = pymupdf.open()
        out_page = rendered_pdf.new_page(width=page.rect.width, height=page.rect.height)
        for item in page.get_images(full=True):
            if not item:
                continue
            xref = item[0]
            try:
                image_info = doc.extract_image(int(xref))
            except Exception:
                continue

            image_bytes = image_info.get("image")
            if not isinstance(image_bytes, (bytes, bytearray)):
                continue

            try:
                rects = page.get_image_rects(xref)
            except Exception:
                continue
            if not rects:
                continue

            for rect in rects:
                if rect is None:
                    continue
                try:
                    out_page.insert_image(rect, stream=image_bytes)
                except Exception:
                    continue

        for drawing in page.get_drawings():
            for x0, y0, x1, y1, linewidth, color in _extract_shape_lines_from_drawing(
                drawing
            ):
                if x1 <= x0 or y1 <= y0:
                    continue
                line_width = linewidth if linewidth and linewidth > 0 else 0.5
                try:
                    out_page.draw_line(
                        pymupdf.Point(x0, y0),
                        pymupdf.Point(x1, y1),
                        color=_to_rgb_color(color, default=(0.0, 0.0, 0.0)),
                        width=float(line_width),
                    )
                except Exception:
                    continue

        raw_payload = page.get_text("dict", sort=True)
        raw_blocks = raw_payload.get("blocks") if isinstance(raw_payload, dict) else []
        raw_line_no = 0
        for block in raw_blocks:
            if not isinstance(block, dict) or block.get("type") != 0:
                continue
            for line in block.get("lines", []) or []:
                if not isinstance(line, dict):
                    continue
                raw_line_no += 1
                if raw_line_no in removed_line_numbers:
                    continue
                _draw_raw_spans_on_page(
                    out_page,
                    line.get("spans", []),
                    source_text,
                    page_no,
                    debug=debug,
                )

        if removed_count:
            if removed_lines and debug:
                _LOGGER.debug(
                    "Preview removed line numbers: source=%s page=%s lines=%s",
                    source_text,
                    page_no,
                    ", ".join(str(item.get("line_no")) for item in removed_lines),
                )

        rendered_pdf.save(output_path, deflate=True, garbage=4)
        rendered_pdf.close()

    return str(Path(output_path))


def _write_reconstructed_page_pdf(
    source_path,
    page_no,
    output_path,
    debug=False,
):
    page_no = int(page_no)
    if page_no < 1:
        raise ValueError(f"Page number must be >=1: {page_no}")

    with pymupdf.open(source_path) as doc:
        if page_no > doc.page_count:
            raise ValueError(
                f"Page number out of range: {page_no} (total: {doc.page_count})"
            )

        page = doc[page_no - 1]
        source_text = str(source_path)

        def _normalize_color(color_value, default=(0.0, 0.0, 0.0)):
            if color_value is None:
                return default

            if isinstance(color_value, (list, tuple)):
                if len(color_value) >= 3:
                    try:
                        return tuple(max(0.0, min(1.0, float(v))) for v in color_value[:3])
                    except (TypeError, ValueError):
                        return default
                return default

            try:
                value = float(color_value)
            except (TypeError, ValueError):
                return default

            if 0.0 <= value <= 1.0:
                return (value, value, value)
            if value <= 0:
                return default

            value = int(value)
            if value <= 0xFFFFFF:
                return (
                    ((value >> 16) & 0xFF) / 255.0,
                    ((value >> 8) & 0xFF) / 255.0,
                    (value & 0xFF) / 255.0,
                )
            return default

        def _to_point(value):
            if value is None:
                return None
            if not isinstance(value, (list, tuple)):
                return None
            if len(value) != 2:
                return None
            try:
                return float(value[0]), float(value[1])
            except (TypeError, ValueError):
                return None

        def _to_point_from_args(args):
            if not args:
                return None
            if len(args) == 1:
                return _to_point(args[0])
            try:
                return float(args[-2]), float(args[-1])
            except (TypeError, ValueError):
                return None

        def _to_rect(value):
            if not isinstance(value, (list, tuple)):
                return None
            if len(value) != 4:
                return None
            try:
                return float(value[0]), float(value[1]), float(value[2]), float(value[3])
            except (TypeError, ValueError):
                return None

        def _to_rotation(direction):
            if not direction or len(direction) != 2:
                return 0.0
            try:
                dx, dy = direction
                if not dx and not dy:
                    return 0.0
                return float(math.degrees(math.atan2(-float(dy), float(dx))))
            except (TypeError, ValueError):
                return 0.0

        def _render_drawing(drawing, drawing_index=None):
            if not isinstance(drawing, dict):
                if debug:
                    _LOGGER.debug(
                        "Drawing[%s] skip: not dict type=%s page=%s",
                        drawing_index,
                        type(drawing),
                        page_no,
                    )
                return

            items = drawing.get("items")
            if not isinstance(items, list):
                if debug:
                    _LOGGER.debug("Drawing[%s] skip: items missing page=%s", drawing_index, page_no)
                return

            fill_key_present = "fill" in drawing
            fill_color_raw = drawing.get("fill")
            fill_color = _normalize_color(fill_color_raw) if fill_color_raw is not None else None
            stroke_color = _normalize_color(drawing.get("color"), default=(0.0, 0.0, 0.0))
            line_width = drawing.get("linewidth", 1.0)
            try:
                line_width = float(line_width) if float(line_width) > 0 else 0.5
            except (TypeError, ValueError):
                line_width = 0.5
            stroke_opacity = drawing.get("stroke_opacity", 1.0)
            fill_opacity = drawing.get("fill_opacity", 1.0)
            try:
                stroke_opacity = float(stroke_opacity)
            except (TypeError, ValueError):
                stroke_opacity = 1.0
            try:
                fill_opacity = float(fill_opacity)
            except (TypeError, ValueError):
                fill_opacity = 1.0

            fill_ops = ("f", "F", "f*", "b", "B", "b*")
            op_counts = Counter()
            has_fill_op = False
            for item in items:
                if not item or not isinstance(item, (list, tuple)) or not item:
                    continue
                if item[0] in fill_ops:
                    has_fill_op = True
                    if isinstance(item[0], str):
                        op_counts[item[0]] += 1

            if debug:
                _LOGGER.debug(
                    "Drawing[%s] start page=%s fill_key=%s raw_fill=%r norm_fill=%r stroke=%r linewidth=%s fill_op=%s ops=%s closePath=%s",
                    drawing_index,
                    page_no,
                    fill_key_present,
                    fill_color_raw,
                    fill_color,
                    stroke_color,
                    _round_float(line_width),
                    fill_opacity,
                    " ".join(f"{k}:{op_counts[k]}" for k in sorted(op_counts.keys())),
                    drawing.get("closePath"),
                )
                if fill_key_present and fill_color is None:
                    _LOGGER.debug(
                        "Drawing[%s] fill value normalization failed: raw_fill=%r page=%s",
                        drawing_index,
                        fill_color_raw,
                        page_no,
                    )

            use_shape = hasattr(out_page, "new_shape")
            shape_disable_reason = None
            shape = None
            if use_shape:
                try:
                    shape = out_page.new_shape()
                except Exception as exc:
                    use_shape = False
                    shape_disable_reason = f"new_shape failed: {type(exc).__name__}: {exc}"
            else:
                shape_disable_reason = "new_shape unavailable"

            if not use_shape and debug:
                _LOGGER.debug(
                    "Drawing[%s] Shape API disabled: %s page=%s",
                    drawing_index,
                    shape_disable_reason,
                    page_no,
                )

            if use_shape:
                if debug:
                    _LOGGER.debug("Drawing[%s] use Shape API page=%s", drawing_index, page_no)

                current = None
                has_geom = False
                fill_op_executed = False
                for item in items:
                    if not item:
                        continue
                    op = item[0]
                    args = item[1:]

                    if op == "m":
                        point = _to_point_from_args(args)
                        if point is not None:
                            shape.move_to(point[0], point[1])
                            current = point
                            has_geom = True
                        continue

                    if op == "l":
                        point = _to_point_from_args(args)
                        if current is not None and point is not None:
                            shape.line_to(point[0], point[1])
                            current = point
                            has_geom = True
                        continue

                    if op == "c":
                        p3 = _to_point_from_args(args[-2:]) if len(args) >= 2 else None
                        if p3 is None:
                            continue
                        p1 = _to_point_from_args(args[0:2]) if len(args) >= 2 else current
                        p2 = _to_point_from_args(args[2:4]) if len(args) >= 4 else current
                        if current is not None and p1 is not None and p2 is not None:
                            shape.curve_to(
                                p1[0], p1[1],
                                p2[0], p2[1],
                                p3[0], p3[1],
                            )
                        else:
                            shape.line_to(p3[0], p3[1])
                        current = p3
                        has_geom = True
                        continue

                    if op in ("v", "y"):
                        point = _to_point_from_args(args)
                        if point is not None:
                            if current is not None:
                                shape.line_to(point[0], point[1])
                            current = point
                            has_geom = True
                        continue

                    if op == "h":
                        try:
                            shape.close_path()
                        except Exception as exc:
                            if debug:
                                _LOGGER.debug(
                                    "Drawing[%s] shape.close_path failed: %s page=%s",
                                    drawing_index,
                                    exc,
                                    page_no,
                                )
                        current = None
                        continue

                        if op == "re":
                            rect_values = _to_rect(args)
                            if rect_values is None:
                                if debug:
                                    _LOGGER.debug(
                                    "Drawing[%s] skip invalid rect args=%r page=%s",
                                    drawing_index,
                                    args,
                                    page_no,
                                )
                            continue
                        x0, y0, x1, y1 = rect_values
                        if x1 == x0 or y1 == y0:
                            if debug:
                                _LOGGER.debug(
                                    "Drawing[%s] skip degenerate rect page=%s rect=%r",
                                    drawing_index,
                                    page_no,
                                    rect_values,
                                )
                            continue
                            try:
                                shape.draw_rect(pymupdf.Rect(x0, y0, x1, y1))
                                has_geom = True
                            except Exception as exc:
                                if debug:
                                    _LOGGER.debug(
                                        "Drawing[%s] shape.draw_rect failed: %s page=%s",
                                        drawing_index,
                                        exc,
                                        page_no,
                                    )
                            continue

                    if op in fill_ops:
                        fill_op_executed = True
                        if debug:
                            _LOGGER.debug(
                                "Drawing[%s] fill operator encountered: op=%s page=%s",
                                drawing_index,
                                op,
                                page_no,
                            )
                        try:
                            shape.finish(
                                color=stroke_color,
                                fill=fill_color,
                                width=line_width,
                                closePath=drawing.get("closePath", True),
                                fill_opacity=fill_opacity,
                                stroke_opacity=stroke_opacity,
                                dashes=drawing.get("dashes"),
                                lineCap=drawing.get("lineCap"),
                                lineJoin=drawing.get("lineJoin"),
                            )
                            shape.commit()
                            if debug:
                                _LOGGER.debug(
                                    "Drawing[%s] shape.finish+commit success on op=%s",
                                    drawing_index,
                                    op,
                                )
                        except TypeError as exc:
                            if debug:
                                _LOGGER.debug(
                                    "Drawing[%s] shape.finish TypeError on op=%s: %s",
                                    drawing_index,
                                    op,
                                    exc,
                                )
                            try:
                                shape.finish(color=stroke_color, fill=fill_color, width=line_width)
                                shape.commit()
                                if debug:
                                    _LOGGER.debug(
                                        "Drawing[%s] shape.finish TypeError fallback args success on op=%s",
                                        drawing_index,
                                        op,
                                    )
                            except Exception as fallback_exc:
                                use_shape = False
                                shape_disable_reason = (
                                    f"shape.finish fallback failed on op {op}: "
                                    f"{type(fallback_exc).__name__}: {fallback_exc}"
                                )
                                if debug:
                                    _LOGGER.debug(
                                        "Drawing[%s] %s",
                                        drawing_index,
                                        shape_disable_reason,
                                    )
                        except Exception as exc:
                            use_shape = False
                            shape_disable_reason = (
                                f"shape.finish/commit failed on op {op}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            if debug:
                                _LOGGER.debug(
                                    "Drawing[%s] %s",
                                    drawing_index,
                                    shape_disable_reason,
                                )
                        break

                    if debug:
                        _LOGGER.debug(
                            "Drawing[%s] unsupported Shape op=%r args=%r page=%s",
                            drawing_index,
                            op,
                            args,
                            page_no,
                        )

                if use_shape:
                    if fill_key_present and fill_color is not None and not fill_op_executed:
                        if debug:
                            _LOGGER.debug(
                                "Drawing[%s] fill key present but no fill op; trying fallback fill commit page=%s",
                                drawing_index,
                                page_no,
                            )
                        try:
                            shape.finish(
                                color=stroke_color,
                                fill=fill_color,
                                width=line_width,
                                closePath=drawing.get("closePath", True),
                                fill_opacity=fill_opacity,
                                stroke_opacity=stroke_opacity,
                                dashes=drawing.get("dashes"),
                                lineCap=drawing.get("lineCap"),
                                lineJoin=drawing.get("lineJoin"),
                            )
                            shape.commit()
                            if debug:
                                _LOGGER.debug("Drawing[%s] fill-on-commit success", drawing_index)
                        except Exception as exc:
                            use_shape = False
                            shape_disable_reason = (
                                f"fill-on-commit failed: {type(exc).__name__}: {exc}"
                            )
                            if debug:
                                _LOGGER.debug("Drawing[%s] %s", drawing_index, shape_disable_reason)

                    if use_shape:
                        try:
                            if hasattr(shape, "close"):
                                shape.close()
                            else:
                                shape.commit()
                        except Exception as exc:
                            if debug:
                                _LOGGER.debug(
                                    "Drawing[%s] shape final close/commit failed: %s page=%s",
                                    drawing_index,
                                    exc,
                                    page_no,
                                )
                        if debug:
                            _LOGGER.debug(
                                "Drawing[%s] shape renderer done has_fill_op=%s has_geom=%s has_fill_key=%s",
                                drawing_index,
                                fill_op_executed,
                                has_geom,
                                fill_key_present,
                            )
                        return

            if debug:
                _LOGGER.debug(
                    "Drawing[%s] fallback renderer start: reason=%s page=%s",
                    drawing_index,
                    shape_disable_reason or "shape branch returned with fallback",
                    page_no,
                )

            current = None
            fallback_ops = Counter()
            for item in items:
                if not item:
                    continue
                op = item[0]
                args = item[1:]
                if isinstance(op, str):
                    fallback_ops[op] += 1

                if op == "m":
                    current = _to_point_from_args(args)
                    continue

                if op == "l":
                    target = _to_point_from_args(args)
                    if current is not None and target is not None:
                        try:
                            if target[0] != current[0] or target[1] != current[1]:
                                out_page.draw_line(
                                    pymupdf.Point(current[0], current[1]),
                                    pymupdf.Point(target[0], target[1]),
                                    color=stroke_color,
                                    width=line_width,
                                )
                        except Exception as exc:
                            if debug:
                                _LOGGER.debug(
                                    "Drawing[%s] fallback draw_line failed: %s page=%s",
                                    drawing_index,
                                    exc,
                                    page_no,
                                )
                    current = target
                    continue

                if op in ("v", "y"):
                    current = _to_point_from_args(args)
                    continue

                if op == "h":
                    current = None
                    continue

                if op == "c":
                    target = _to_point_from_args(args)
                    if current is not None and target is not None:
                        try:
                            out_page.draw_line(
                                pymupdf.Point(current[0], current[1]),
                                pymupdf.Point(target[0], target[1]),
                                color=stroke_color,
                                width=line_width,
                            )
                        except Exception as exc:
                            if debug:
                                _LOGGER.debug(
                                    "Drawing[%s] fallback draw_line(curve fallback) failed: %s page=%s",
                                    drawing_index,
                                    exc,
                                    page_no,
                                )
                    current = target
                    continue

                if op == "re":
                    rect_values = _to_rect(args)
                    if not rect_values:
                        if debug:
                            _LOGGER.debug(
                                "Drawing[%s] fallback invalid rect args=%r page=%s",
                                drawing_index,
                                args,
                                page_no,
                            )
                        continue
                    x0, y0, x1, y1 = rect_values
                    if x1 == x0 or y1 == y0:
                        if debug:
                            _LOGGER.debug(
                                "Drawing[%s] fallback skip degenerate rect page=%s rect=%r",
                                drawing_index,
                                page_no,
                                rect_values,
                            )
                        continue
                    try:
                        out_page.draw_rect(
                            pymupdf.Rect(x0, y0, x1, y1),
                            color=stroke_color,
                            width=line_width,
                            fill=fill_color,
                            fill_opacity=fill_opacity if fill_opacity else 1.0,
                        )
                    except TypeError:
                        try:
                            out_page.draw_rect(
                                pymupdf.Rect(x0, y0, x1, y1),
                                color=stroke_color,
                                width=line_width,
                                fill=fill_color,
                            )
                        except Exception as exc:
                            if debug:
                                _LOGGER.debug(
                                    "Drawing[%s] fallback draw_rect failed: %s page=%s",
                                    drawing_index,
                                    exc,
                                    page_no,
                                )
                    except Exception as exc:
                        if debug:
                            _LOGGER.debug(
                                "Drawing[%s] fallback draw_rect failed: %s page=%s",
                                drawing_index,
                                exc,
                                page_no,
                            )
                    continue

                if op in fill_ops:
                    if debug:
                        _LOGGER.debug(
                            "Drawing[%s] fallback encountered fill op=%s but fallback path does not apply fill",
                            drawing_index,
                            op,
                        )
                    continue

                if debug:
                    _LOGGER.debug(
                        "Drawing[%s] fallback unsupported op=%r args=%r page=%s",
                        drawing_index,
                        op,
                        args,
                        page_no,
                    )

            if debug:
                _LOGGER.debug(
                    "Drawing[%s] fallback completed page=%s ops=%s",
                    drawing_index,
                    page_no,
                    " ".join(f"{k}:{fallback_ops[k]}" for k in sorted(fallback_ops.keys())),
                )

        raw_text = page.get_text("dict", sort=True)
        drawings = page.get_drawings()
        images = page.get_images(full=True)

        if debug:
            fill_with_color = 0
            fill_op_missing = 0
            draw_fill_ops = Counter()
            for drawing in drawings:
                if not isinstance(drawing, dict):
                    continue
                drawing_fill = (
                    _normalize_color(drawing.get("fill"), default=None)
                    if drawing.get("fill") is not None
                    else None
                )
                if drawing_fill is not None:
                    fill_with_color += 1
                    items = drawing.get("items") or []
                    if not any(
                        isinstance(item, (list, tuple))
                        and item
                        and item[0] in ("f", "F", "f*", "b", "B", "b*")
                        for item in items
                    ):
                        fill_op_missing += 1
                    for item in items:
                        if not item or not isinstance(item, (list, tuple)):
                            continue
                        if isinstance(item[0], str):
                            draw_fill_ops[item[0]] += 1

            _LOGGER.debug(
                "Reconstruct source=%s page=%s extracted_text_blocks=%s drawings=%s images=%s draw_fill=%s fill_op_missing=%s fill_ops=%s",
                source_text,
                page_no,
                len((raw_text or {}).get("blocks", [])) if isinstance(raw_text, dict) else "n/a",
                len(drawings),
                len(images),
                fill_with_color,
                fill_op_missing,
                " ".join(f"{k}:{draw_fill_ops[k]}" for k in sorted(draw_fill_ops.keys())),
            )

        rendered_pdf = pymupdf.open()
        out_page = rendered_pdf.new_page(width=page.rect.width, height=page.rect.height)
        out_page.draw_rect(
            out_page.rect,
            fill=(1, 1, 1),
            color=(1, 1, 1),
            width=0,
        )

        fontfile_for_korean = _get_reconstruct_fontfile()
        korean_fontname = None
        if not fontfile_for_korean:
            if debug:
                _LOGGER.debug(
                    "No Korean fallback fontfile found; text may use Latin-only fonts: source=%s page=%s",
                    source_path,
                    page_no,
                )
        else:
            if hasattr(rendered_pdf, "insert_font"):
                try:
                    korean_fontname = rendered_pdf.insert_font(
                        fontname="reconstruct_korean",
                        fontfile=fontfile_for_korean,
                    )
                    if debug:
                        _LOGGER.debug(
                            "Registered Korean fallback fontname=%s fontfile=%s source=%s page=%s",
                            korean_fontname,
                            fontfile_for_korean,
                            source_path,
                            page_no,
                        )
                except TypeError:
                    try:
                        with open(fontfile_for_korean, "rb") as font_handle:
                            font_bytes = font_handle.read()
                        korean_fontname = rendered_pdf.insert_font(
                            fontname="reconstruct_korean",
                            fontbuffer=font_bytes,
                        )
                    except Exception as exc:
                        if debug:
                            _LOGGER.debug(
                                "Failed to register fallback Korean font via buffer path=%s source=%s page=%s error=%s",
                                fontfile_for_korean,
                                source_path,
                                page_no,
                                exc,
                            )
                except Exception as exc:
                    if debug:
                        _LOGGER.debug(
                            "Failed to register fallback Korean fontfile=%s source=%s page=%s error=%s",
                            fontfile_for_korean,
                            source_path,
                            page_no,
                            exc,
                        )
            elif debug:
                _LOGGER.debug(
                    "rendered_pdf.insert_font missing; cannot register fallback Korean fontfile=%s source=%s page=%s",
                    fontfile_for_korean,
                    source_path,
                    page_no,
                )

        image_xref_to_stream = {}
        for item in images:
            if not item:
                continue
            xref = item[0]
            if xref in image_xref_to_stream:
                continue

            try:
                image_info = doc.extract_image(int(xref))
            except Exception:
                continue

            image_bytes = image_info.get("image")
            if not isinstance(image_bytes, (bytes, bytearray)):
                continue
            image_xref_to_stream[int(xref)] = bytes(image_bytes)

        for item in images:
            if not item:
                continue
            xref = int(item[0])
            image_bytes = image_xref_to_stream.get(xref)
            if not image_bytes:
                continue
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                continue
            if not rects:
                continue
            for rect in rects:
                if rect is None:
                    continue
                try:
                    out_page.insert_image(rect, stream=image_bytes)
                except Exception:
                    continue

        for drawing_index, drawing in enumerate(drawings):
            _render_drawing(drawing, drawing_index=drawing_index)

        raw_blocks = raw_text.get("blocks") if isinstance(raw_text, dict) else []
        if debug and fontfile_for_korean:
            _LOGGER.debug(
                "Reconstruct with Korean fallback fontfile=%s for page=%s (fontname=%r)",
                fontfile_for_korean,
                page_no,
                korean_fontname,
            )

        for block in raw_blocks:
            if not isinstance(block, dict) or block.get("type") != 0:
                continue
            for line in block.get("lines", []) or []:
                if not isinstance(line, dict):
                    continue
                for span in line.get("spans", []) or []:
                    if not isinstance(span, dict):
                        continue

                    text = (span.get("text") or "").replace("\xa0", " ")
                    if not text:
                        continue

                    origin = span.get("origin")
                    if origin is None:
                        bbox = span.get("bbox")
                        if not (
                            isinstance(bbox, (list, tuple))
                            and len(bbox) == 4
                        ):
                            continue
                        try:
                            x0, y0, x1, y1 = [float(v) for v in bbox]
                            origin = (x0, y1)
                        except (TypeError, ValueError):
                            continue

                    point = _to_point(origin)
                    if point is None:
                        continue
                    size = span.get("size")
                    try:
                        size = float(size)
                    except (TypeError, ValueError):
                        size = 11.0
                    if size <= 0:
                        size = 11.0

                    color = _normalize_color(span.get("color"), default=(0.0, 0.0, 0.0))
                    font = (span.get("font") or "helv").strip() or "helv"
                    rotate = _to_rotation(span.get("dir"))
                    inserted = False
                    use_korean_font = korean_fontname is not None and _contains_korean(text)

                    font_candidates = []
                    if use_korean_font:
                        font_candidates.append(korean_fontname)
                    font_candidates.extend([font, "helv"])

                    for font_name in font_candidates:
                        for angle in (rotate, 0.0):
                            try:
                                out_page.insert_text(
                                    point,
                                    text,
                                    fontname=font_name,
                                    fontsize=size,
                                    color=color,
                                    rotate=angle,
                                )
                                inserted = True
                                break
                            except Exception:
                                continue
                        if inserted:
                            break

                    if not inserted and debug:
                        _LOGGER.debug(
                            "Reconstruct text span insert failed: source=%s page=%s font=%r size=%s rotate=%s text=%r",
                            source_text,
                            page_no,
                            font,
                            _round_float(size),
                            _round_float(rotate),
                            text[:120],
                        )

        rendered_pdf.save(output_path, deflate=True, garbage=4)
        rendered_pdf.close()

    return str(Path(output_path))


def _image_only_output_path(source_path, page_no):
    prefix = f"{Path(source_path).stem}_page{page_no}_"
    tmp = tempfile.NamedTemporaryFile(prefix=prefix, suffix="_image_only.pdf", delete=False)
    tmp.close()
    return tmp.name


def _reconstruct_output_path(source_path, page_no):
    prefix = f"{Path(source_path).stem}_page{page_no}_"
    tmp = tempfile.NamedTemporaryFile(
        prefix=prefix, suffix="_reconstruct_from_extract.pdf", delete=False
    )
    tmp.close()
    return tmp.name


def _safe_text_list(values):
    return [_sanitize_text(value) for value in values if value is not None]


def _normalize_table_record(table):
    pages = table.get("pages", [table.get("page")])
    row_texts = table.get("rows_text") or []
    return {
        "page": table.get("page"),
        "start_page": table.get("start_page", table.get("page")),
        "end_page": table.get("page_end", table.get("page")),
        "pages": pages,
        "table_no": table.get("table_no"),
        "x": table.get("x"),
        "y": table.get("y"),
        "bbox": table.get("bbox"),
        "rotation": table.get("rotation"),
        "infer_method": table.get("infer_method"),
        "rows": table.get("row_count"),
        "cols": table.get("col_count"),
        "font_size": None,
        "text": _sanitize_text(table.get("text", "")),
        "rows_text": [_sanitize_text(row) for row in row_texts],
        "row_lines": table.get("row_lines", []),
        "vertical_lines": table.get("vertical_lines", []),
        "components": table.get("components", {}),
    }


def _split_table_row(row_text):
    if row_text is None:
        return []
    text = _sanitize_text(row_text)
    if not text:
        return []
    return [_sanitize_text(cell) for cell in str(text).split(" | ")]


def _markdown_cell(value):
    if value is None:
        return ""
    text = _sanitize_text(value)
    if text is None:
        return ""
    return str(text).replace("|", r"\|").replace("\n", "<br>")


def _collect_markdown_rows(table):
    rows = []
    for row_text in table.get("rows_text") or []:
        cells = _split_table_row(row_text)
        if not cells:
            continue
        rows.append([_markdown_cell(cell) for cell in cells])

    if not rows:
        text = _sanitize_text(table.get("text", ""))
        for line in str(text).splitlines() if text is not None else []:
            cells = _split_table_row(line)
            if cells:
                rows.append([_markdown_cell(cell) for cell in cells])

    return rows


def _table_to_markdown_block(index, table):
    rows = _collect_markdown_rows(table)
    page_no = table.get("page")
    start_page = table.get("start_page", page_no)
    end_page = table.get("page_end", table.get("end_page", page_no))
    pages = table.get("pages")
    if not pages:
        pages = [start_page]
    header_cells = []
    body_rows = rows
    if rows:
        max_cols = max(len(row) for row in rows)
        rows = [row + [""] * max(0, max_cols - len(row)) for row in rows]
        header_cells = rows[0]
        body_rows = rows[1:]

    lines = [f"## Table {index}"]
    location = f"page {start_page}" if start_page == end_page else f"pages {start_page}-{end_page}"
    lines.append(f"- page: {location}")
    lines.append(f"- table_no: {table.get('table_no')}")
    lines.append(f"- infer_method: {table.get('infer_method')}")
    lines.append(f"- rows: {table.get('row_count', len(rows))} cols: {table.get('col_count', len(header_cells))}")
    lines.append("")

    if not rows:
        lines.append("_(No rows detected)_")
        return lines

    lines.append("| " + " | ".join(_markdown_cell(cell) for cell in header_cells) + " |")
    lines.append("| " + " | ".join("---" for _ in header_cells) + " |")
    for row in body_rows:
        lines.append("| " + " | ".join(row) + " |")

    return lines


def _write_tables_markdown(records, output_path):
    table_blocks = []
    table_index = 1
    for record in records:
        for table in record.get("tables", []) or []:
            table_blocks.extend(_table_to_markdown_block(table_index, table))
            table_index += 1
            table_blocks.append("")

    if not table_blocks:
        table_blocks = ["# Tables", "", "No tables detected."]

    with open(output_path, "w", encoding="utf-8", errors="replace") as f:
        for line in table_blocks:
            f.write(f"{line}\n")

    return output_path


def _remove_watermark_rows_from_table(table, repeated_watermarks):
    if not repeated_watermarks:
        return table

    rows = table.get("rows_text") or []
    if not rows:
        return table

    cleaned_rows = []
    for row in rows:
        if row is None:
            continue

        raw_row = str(row)
        cells = [part.strip() for part in raw_row.split(" | ")]
        kept_cells = []
        for cell in cells:
            normalized_cell = _normalize_line(cell).casefold()
            if not normalized_cell:
                continue
            if normalized_cell in repeated_watermarks:
                continue
            if len(cells) == 1 and any(
                watermark in normalized_cell for watermark in repeated_watermarks
            ):
                continue
            kept_cells.append(cell.strip())

        if not kept_cells:
            continue

        cleaned_rows.append(" | ".join(kept_cells))

    if len(cleaned_rows) == len(rows):
        return table

    if not cleaned_rows:
        return None

    updated_table = dict(table)
    updated_table["rows_text"] = cleaned_rows
    updated_table["text"] = "\n".join(cleaned_rows)
    updated_table["row_count"] = len(cleaned_rows)
    updated_table["col_count"] = max(
        (len(row.split(" | ")) for row in cleaned_rows),
        default=0,
    )

    return updated_table


def read_pdf(
    path,
    strip_watermarks=True,
    strip_headers=True,
    strip_footers=True,
    patterns=None,
    ratio_threshold=0.6,
    header_ratio=0.08,
    footer_ratio=0.08,
    max_pages=100,
    pages=None,
    preserve_newlines=False,
    extract_tables=False,
    strip_markdown_lines=False,
    debug=False,
    table_debug=None,
    table_mode="auto",
    return_pages_with_lines=False,
    return_raw_pages=False,
):
    path = Path(path)

    with pymupdf.open(path) as doc:
        if return_pages_with_lines or return_raw_pages:
            pages_with_lines, raw_pages = _extract_pages(
                doc,
                str(path),
                header_ratio,
                footer_ratio,
                max_pages=max_pages,
                pages=pages,
                preserve_newlines=preserve_newlines,
                strip_markdown_lines=strip_markdown_lines,
                extract_tables=extract_tables,
                debug=debug,
                table_debug=table_debug if table_debug is not None else debug,
                table_mode=table_mode,
                capture_raw_pages=return_raw_pages,
            )
        else:
            pages_with_lines = _extract_pages(
                doc,
                str(path),
                header_ratio,
                footer_ratio,
                max_pages=max_pages,
                pages=pages,
                preserve_newlines=preserve_newlines,
                strip_markdown_lines=strip_markdown_lines,
                extract_tables=extract_tables,
                debug=debug,
                table_debug=table_debug if table_debug is not None else debug,
                table_mode=table_mode,
            )
            raw_pages = None

    tables = []
    for _, _, page_tables, _ in pages_with_lines:
        tables.extend(page_tables)

    if extract_tables:
        tables = _merge_cross_page_tables(tables)

    compiled_patterns = _compile_patterns(patterns or [])
    extracted_lines = [lines for _, lines, _, _ in pages_with_lines]
    repeated_watermarks = set()
    if strip_watermarks:
        repeated_watermarks = _collect_repeated_lines(
            extracted_lines,
            ratio_threshold=ratio_threshold,
        )

    if repeated_watermarks:
        filtered = []
        for table in tables:
            cleaned = _remove_watermark_rows_from_table(table, repeated_watermarks)
            if cleaned is not None:
                filtered.append(cleaned)
        tables = filtered

    table_by_page = {}
    for table in tables:
        page_key = table.get("page")
        table_by_page.setdefault(page_key, []).append(table)

    records = []
    for page_no, lines, _, shape_lines in pages_with_lines:
        body_size = _estimate_body_font_size(lines)
        sections, region_summary, removed, kept_text = _build_sections(
            lines,
            body_size,
            repeated_watermarks,
            compiled_patterns,
            strip_headers,
            strip_footers,
            strip_watermarks,
        )

        anomalies = _collect_anomalies(
            lines=lines,
            sections=sections,
            region_summary=region_summary,
            removed=removed,
            body_size=body_size,
            strip_headers=strip_headers,
            strip_footers=strip_footers,
            header_ratio=header_ratio,
            footer_ratio=footer_ratio,
        )

        records.append(
            {
                "page": page_no,
                "rotation": lines[0]["rotation"] if lines else 0,
                "text": kept_text,
                "header": _sanitize_text(
                    "\n".join(
                        _safe_text_list(
                            item["text"] for item in sections["header"]["items"] if item["removed"]
                        )
                    )
                ),
                "footer": _sanitize_text(
                    "\n".join(
                        _safe_text_list(
                            item["text"] for item in sections["footer"]["items"] if item["removed"]
                        )
                    )
                ),
                "watermark": _sanitize_text(
                    "\n".join(
                        _safe_text_list(
                            item["text"] for item in sections["watermark"]["items"] if item["removed"]
                        )
                    )
                ),
                "removed": removed,
                "regions": {
                    "header": sections["header"]["items"],
                    "body": sections["body"]["items"],
                    "footer": sections["footer"]["items"],
                    "watermark": sections["watermark"]["items"],
                },
                "region_summary": region_summary,
                "anomalies": anomalies,
                "tables": table_by_page.get(page_no, []),
                "table_count": len(table_by_page.get(page_no, [])),
                "shape_lines": shape_lines,
                "shape_line_count": len(shape_lines),
            }
        )

    if return_pages_with_lines:
        return records, pages_with_lines
    if return_raw_pages:
        return records, raw_pages

    return records


def write_jsonl(records, output_path):
    with open(output_path, "w", encoding="utf-8", errors="replace") as f:
        for record in records:
            ordered_items = []
            for region_name in ("header", "body", "footer", "watermark"):
                for item in record.get("regions", {}).get(region_name, []):
                    ordered_items.append(
                        (
                            item.get("line_no", 0),
                            {
                                "page": record.get("page"),
                                "font_size": _round_float(item.get("font_size", item.get("size"))),
                                "x": _round_float(item.get("x")),
                                "y": _round_float(item.get("y")),
                                "row_no": item.get("row_no"),
                                "rotation": item.get("rotation"),
                                "text": _sanitize_text(item.get("text", "")),
                            },
                        )
                    )
            ordered_items.sort(key=lambda entry: entry[0] if entry[0] is not None else 0)
            for _, payload in ordered_items:
                f.write(json.dumps(payload, ensure_ascii=False))
                f.write("\n")

            for table in record.get("tables", []):
                f.write(
                    json.dumps(
                        {
                    "type": "table",
                    "page": record.get("page"),
                    **_normalize_table_record(table),
                    },
                        ensure_ascii=False,
                    )
                )
                f.write("\n")


def write_jsonl_pages(records, output_path):
    with open(output_path, "w", encoding="utf-8", errors="replace") as f:
        for record in records:
            safe_record = {
                "page": record.get("page"),
                "rotation": record.get("rotation"),
                "text": _sanitize_text(record.get("text", "")),
                "header": _sanitize_text(record.get("header", "")),
                "footer": _sanitize_text(record.get("footer", "")),
                "watermark": _sanitize_text(record.get("watermark", "")),
                "removed": record.get("removed", {}),
                "regions": record.get("regions", {}),
                "region_summary": record.get("region_summary", {}),
                "anomalies": record.get("anomalies", []),
                "tables": [_normalize_table_record(table) for table in record.get("tables", [])],
                "table_count": record.get("table_count", 0),
            }
            f.write(json.dumps(safe_record, ensure_ascii=False))
            f.write("\n")


def write_raw_line_log(records, output_path):
    with open(output_path, "w", encoding="utf-8", errors="replace") as f:
        global_line_no = 1
        for record in records:
            page_no = record.get("page")

            for region_name in ("header", "body", "footer", "watermark"):
                for item in record.get("regions", {}).get(region_name, []):
                    payload = {
                        "type": "line",
                        "page": page_no,
                        "region": item.get("region") or region_name,
                        "line_no": item.get("line_no"),
                        "row_no": item.get("row_no"),
                        "global_line_no": global_line_no,
                        "removed": item.get("removed"),
                        "removed_reason": item.get("removed_reason"),
                        "rotation": item.get("rotation"),
                        "font_size": item.get("font_size"),
                        "x": item.get("x"),
                        "y": item.get("y"),
                        "x_ratio": item.get("x_ratio"),
                        "y_ratio": item.get("y_ratio"),
                        "text": item.get("text"),
                    }
                    f.write(json.dumps(payload, ensure_ascii=False))
                    f.write("\n")
                    global_line_no += 1

            payload = {
                "type": "table-detection-summary",
                "page": page_no,
                "region": "table-detection",
                "global_line_no": global_line_no,
                "shape_line_count": len(record.get("shape_lines") or []),
                "table_count": record.get("table_count", 0),
            }
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")
            global_line_no += 1

            for table in record.get("tables", []):
                normalized = _normalize_table_record(table)
                payload = {
                    "type": "table",
                    "page": page_no,
                    "table_no": normalized.get("table_no"),
                    "start_page": normalized.get("start_page"),
                    "end_page": normalized.get("end_page"),
                    "pages": normalized.get("pages"),
                    "region": "table",
                    "global_line_no": global_line_no,
                    "rotation": normalized.get("rotation"),
                    "x": normalized.get("x"),
                    "y": normalized.get("y"),
                    "bbox": normalized.get("bbox"),
                    "rows": normalized.get("rows"),
                    "cols": normalized.get("cols"),
                    "infer_method": normalized.get("infer_method"),
                    "text": normalized.get("text"),
                }
                f.write(json.dumps(payload, ensure_ascii=False))
                f.write("\n")
                global_line_no += 1

                for line in normalized.get("row_lines", []) or []:
                    f.write(
                        json.dumps(
                            {
                                "type": "table-line",
                                "page": page_no,
                                "table_no": normalized.get("table_no"),
                                "line_kind": "row",
                                "orientation": "horizontal",
                                "global_line_no": global_line_no,
                                "x0": line.get("x0"),
                                "y0": line.get("y0"),
                                "x1": line.get("x1"),
                                "y1": line.get("y1"),
                            },
                            ensure_ascii=False,
                        )
                    )
                    f.write("\n")
                    global_line_no += 1

                for line in normalized.get("vertical_lines", []) or []:
                    f.write(
                        json.dumps(
                            {
                                "type": "table-line",
                                "page": page_no,
                                "table_no": normalized.get("table_no"),
                                "line_kind": "column",
                                "orientation": "vertical",
                                "global_line_no": global_line_no,
                                "x0": line.get("x0"),
                                "y0": line.get("y0"),
                                "x1": line.get("x1"),
                                "y1": line.get("y1"),
                            },
                            ensure_ascii=False,
                        )
                    )
                    f.write("\n")
                    global_line_no += 1

            for line in record.get("shape_lines", []) or []:
                if line.get("type") != "shape-line":
                    continue
                payload = {
                    "type": "shape-line",
                    "page": page_no,
                    "region": line.get("region", "shape"),
                    "global_line_no": global_line_no,
                    "orientation": line.get("orientation"),
                    "x0": line.get("x0"),
                    "y0": line.get("y0"),
                    "x1": line.get("x1"),
                    "y1": line.get("y1"),
                    "x": line.get("x"),
                    "y": line.get("y"),
                    "length": line.get("length"),
                    "linewidth": line.get("linewidth"),
                    "color": line.get("color"),
                }
                f.write(json.dumps(payload, ensure_ascii=False))
                f.write("\n")
                global_line_no += 1


def write_raw_components(pages_with_lines, output_path):
    with open(output_path, "w", encoding="utf-8", errors="replace") as f:
        for page_no, lines, tables, shape_lines in pages_with_lines:
            for line in lines:
                payload = {
                    "type": "raw-line",
                    "page": page_no,
                    "line_no": line.get("line"),
                    "row_no": line.get("row_no"),
                    "region": line.get("location"),
                    "rotation": line.get("rotation"),
                    "rotation_axis": line.get("rotation_axis"),
                    "font_family": line.get("font_family"),
                    "font_size": line.get("size"),
                    "x": (line.get("position", {}) or {}).get("x"),
                    "y": (line.get("position", {}) or {}).get("y"),
                    "x_ratio": (line.get("position", {}) or {}).get("x_ratio"),
                    "y_ratio": (line.get("position", {}) or {}).get("y_ratio"),
                    "bbox": line.get("bbox"),
                    "text": line.get("raw"),
                    "markdown": line.get("markdown"),
                    "spans": line.get("spans", []),
                    "source": line.get("source"),
                }
                f.write(json.dumps(payload, ensure_ascii=False))
                f.write("\n")

            for table in tables:
                payload = {
                    "type": "raw-table",
                    "page": page_no,
                    "table_no": table.get("table_no"),
                    "source": table.get("source"),
                    "rotation": table.get("rotation"),
                    "bbox": table.get("bbox"),
                    "infer_method": table.get("infer_method"),
                    "rows": table.get("row_count"),
                    "cols": table.get("col_count"),
                    "text": table.get("text"),
                    "row_lines": table.get("row_lines", []),
                    "vertical_lines": table.get("vertical_lines", []),
                    "components": table.get("components", {}),
                }
                f.write(json.dumps(payload, ensure_ascii=False))
                f.write("\n")

            for line in shape_lines:
                if line.get("type") != "shape-line":
                    continue
                payload = {
                    "type": "raw-shape-line",
                    "page": page_no,
                    "rotation": lines[0].get("rotation", 0) if lines else 0,
                    "x0": line.get("x0"),
                    "y0": line.get("y0"),
                    "x1": line.get("x1"),
                    "y1": line.get("y1"),
                    "x": line.get("x"),
                    "y": line.get("y"),
                    "length": line.get("length"),
                    "orientation": line.get("orientation"),
                    "linewidth": line.get("linewidth"),
                    "color": line.get("color"),
                    "region": line.get("region", "shape"),
                }
                f.write(json.dumps(payload, ensure_ascii=False))
                f.write("\n")


def write_raw_pages(raw_pages, output_path):
    with open(output_path, "w", encoding="utf-8", errors="replace") as f:
        for payload in raw_pages:
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")


def write_rawdict_pages(raw_pages, output_path):
    with open(output_path, "w", encoding="utf-8", errors="replace") as f:
        for payload in raw_pages:
            if not isinstance(payload, dict):
                continue
            f.write(
                json.dumps(
                    {
                        "page": payload.get("page"),
                        "source": payload.get("source"),
                        "rotation": payload.get("rotation"),
                        "rect": payload.get("rect"),
                        "text_rawdict": payload.get("text_rawdict"),
                    },
                    ensure_ascii=False,
                )
            )
            f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read PDF pages with PyMuPDF, split/remove headers/footers/watermarks."
    )
    parser.add_argument("file", help="Path to the PDF file")
    parser.add_argument(
        "--output",
        help="Output path for page text JSONL. Defaults to <filename>_pages.jsonl",
    )
    parser.add_argument(
        "--watermark-patterns",
        nargs="*",
        default=[],
        help="Optional watermark fragments or regex patterns to strip (case-insensitive).",
    )
    parser.add_argument(
        "--strip-watermarks",
        action="store_true",
        default=True,
        help="Enable automatic repeated-line watermark removal (default).",
    )
    parser.add_argument(
        "--no-strip-watermarks",
        action="store_false",
        dest="strip_watermarks",
        help="Disable automatic repeated-line watermark removal.",
    )
    parser.add_argument(
        "--strip-headers",
        action="store_true",
        default=True,
        help="Enable automatic header removal (default).",
    )
    parser.add_argument(
        "--keep-headers",
        action="store_false",
        dest="strip_headers",
        help="Keep header lines in output JSONL instead of removing them.",
    )
    parser.add_argument(
        "--strip-footers",
        action="store_true",
        default=True,
        help="Enable automatic footer removal (default).",
    )
    parser.add_argument(
        "--keep-footers",
        action="store_false",
        dest="strip_footers",
        help="Keep footer lines in output JSONL instead of removing them.",
    )
    parser.add_argument(
        "--watermark-ratio",
        type=float,
        default=0.6,
        help="Ratio threshold to detect repeated watermark lines across pages.",
    )
    parser.add_argument(
        "--header-ratio",
        type=float,
        default=0.08,
        help="Top zone ratio (page fraction) considered header.",
    )
    parser.add_argument(
        "--footer-ratio",
        type=float,
        default=0.08,
        help="Bottom zone ratio (page fraction) considered footer.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Maximum number of pages to read from the PDF (default: 100).",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Specific pages or ranges to read (example: 1-5,10,20-30). Overrides --max-pages.",
    )
    parser.add_argument(
        "--preserve-newlines",
        action="store_true",
        help="Keep whitespace/newline characters from source line text instead of collapsing into single spaces.",
    )
    parser.add_argument(
        "--find-tables",
        action="store_true",
        help="Detect tables on each page with PyMuPDF table extractor.",
    )
    parser.add_argument(
        "--strip-markdown-lines",
        "--strip_markdown_lines",
        "--strip-markdown",
        dest="strip_markdown_lines",
        action="store_true",
        help="Skip markdown-like table/text-art lines before building sections and table structure.",
    )
    parser.add_argument(
        "--tables-markdown",
        nargs="?",
        const="",
        default=None,
        help=(
            "Write detected tables into markdown. Optional path can be provided."
        ),
    )
    parser.add_argument(
        "--image-only-page",
        type=int,
        default=None,
        help=(
            "Render one page as an image-only PDF (flattened page image). "
            "With --strip-markdown-lines, markdown-like text lines are removed before output."
        ),
    )
    parser.add_argument(
        "--image-only-output",
        default=None,
        help="Output path for --image-only-page. If omitted, a temporary file is created.",
    )
    parser.add_argument(
        "--image-only-dpi",
        type=int,
        default=180,
        help="(legacy) Kept for backward compatibility; currently ignored for image-only extraction.",
    )
    parser.add_argument(
        "--preview-page",
        type=int,
        default=None,
        help=(
            "Render one single page after removing watermark-rotation text and print/save the cleaned PDF "
            "(default rotation=55°)."
        ),
    )
    parser.add_argument(
        "--preview-output",
        default=None,
        help="Output path for --preview-page. If omitted, a temporary file is created.",
    )
    parser.add_argument(
        "--watermark-angle",
        type=float,
        default=_WATERMARK_ROTATION_DEGREE,
        help="Target text rotation (degrees) treated as watermark in --preview-page.",
    )
    parser.add_argument(
        "--watermark-angle-tolerance",
        type=float,
        default=_WATERMARK_ROTATION_TOLERANCE,
        help="Angle tolerance used for watermark matching in --preview-page.",
    )
    parser.add_argument(
        "--preview-strip-markdown",
        action="store_true",
        help="Also remove markdown-like lines in --preview-page.",
    )
    parser.add_argument(
        "--preview-strip-body-rotation",
        action="store_true",
        help=(
            "Also remove rotation-matching lines in all regions in --preview-page. "
            "Without this flag, only body-region rotation matches are removed."
        ),
    )
    parser.add_argument(
        "--reconstruct-page",
        type=int,
        default=None,
        help=(
            "Reconstruct one page from get_text/get_drawings/get_images and save as a PDF."
        ),
    )
    parser.add_argument(
        "--reconstruct-output",
        default=None,
        help="Output path for --reconstruct-page. If omitted, a temporary file is created.",
    )
    parser.add_argument(
        "--table-mode",
        default="auto",
        choices=("auto", "default", "lines", "text"),
        help="PyMuPDF table strategy to try.",
    )
    parser.add_argument(
        "--table-debug",
        action="store_true",
        help="Emit table detection diagnostics to log.",
    )
    parser.add_argument(
        "--raw-line-log",
        default=None,
        help="Write raw line extraction records to this file.",
    )
    parser.add_argument(
        "--raw-components",
        nargs="?",
        const="",
        default=None,
        help=(
            "Write full raw page components (lines + shape lines + tables) to JSONL."
            " Optional path can be provided."
        ),
    )
    parser.add_argument(
        "--raw-page",
        nargs="?",
        const="",
        default=None,
        help=(
            "Write pure PyMuPDF page objects (get_text/get_drawings/get_links) for selected pages."
            " Optional path can be provided. Pass `rawdict` to output only page.get_text('rawdict') content."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging, including raw line/table diagnostics.",
    )
    parser.add_argument(
        "--legacy-page-jsonl",
        action="store_true",
        help="Keep old page-based JSONL output instead of line-by-line JSONL output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug or args.table_debug else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.preview_page is not None:
        try:
            page_no = int(args.preview_page)
        except (TypeError, ValueError):
            raise SystemExit("--preview-page must be an integer.")

        if args.pages is not None:
            raise SystemExit("--preview-page cannot be combined with --pages.")

        if args.image_only_page is not None:
            raise SystemExit("--preview-page cannot be combined with --image-only-page.")

        if page_no < 1:
            raise SystemExit("--preview-page must be >= 1.")

        if args.watermark_angle is None:
            raise SystemExit("--watermark-angle is required.")
        if args.watermark_angle_tolerance is None:
            raise SystemExit("--watermark-angle-tolerance is required.")

        preview_output_path = args.preview_output
        if not preview_output_path:
            preview_output_path = _image_only_output_path(args.file, page_no)

        try:
            created_path = _write_preview_cleaned_page_pdf(
                args.file,
                page_no,
                preview_output_path,
                watermark_angle=args.watermark_angle,
                watermark_tolerance=args.watermark_angle_tolerance,
                header_ratio=args.header_ratio,
                footer_ratio=args.footer_ratio,
                remove_markdown_lines=args.preview_strip_markdown,
                strip_body_rotation=args.preview_strip_body_rotation,
                debug=args.debug or args.table_debug,
            )
        except Exception as exc:
            raise SystemExit(f"Failed to create preview cleaned page PDF: {exc}")

        print(f"Preview cleaned page PDF saved to: {created_path}")

        if args.find_tables:
            preview_records = read_pdf(
                created_path,
                strip_watermarks=False,
                strip_headers=False,
                strip_footers=False,
                patterns=[],
                ratio_threshold=args.watermark_ratio,
                header_ratio=args.header_ratio,
                footer_ratio=args.footer_ratio,
                max_pages=1,
                pages=[1],
                preserve_newlines=args.preserve_newlines,
                extract_tables=True,
                debug=args.debug,
                table_debug=args.debug or args.table_debug,
                table_mode=args.table_mode,
                return_pages_with_lines=False,
                return_raw_pages=False,
            )
            total_tables = sum(record.get("table_count", 0) for record in preview_records)
            print(f"Detected tables (preview): {total_tables}")

            if args.tables_markdown is not None:
                preview_tables_md = (
                    str(Path(created_path).with_name(f"{Path(created_path).stem}_tables.md"))
                    if args.tables_markdown == ""
                    else args.tables_markdown
                )
                if preview_tables_md:
                    _write_tables_markdown(preview_records, preview_tables_md)
                    print(f"Preview tables markdown -> {preview_tables_md}")

        return

    if args.reconstruct_page is not None:
        try:
            page_no = int(args.reconstruct_page)
        except (TypeError, ValueError):
            raise SystemExit("--reconstruct-page must be an integer.")

        if args.pages is not None:
            raise SystemExit("--reconstruct-page cannot be combined with --pages.")
        if args.image_only_page is not None:
            raise SystemExit("--reconstruct-page cannot be combined with --image-only-page.")
        if args.preview_page is not None:
            raise SystemExit("--reconstruct-page cannot be combined with --preview-page.")
        if page_no < 1:
            raise SystemExit("--reconstruct-page must be >= 1.")

        output_path = args.reconstruct_output
        if not output_path:
            output_path = _reconstruct_output_path(args.file, page_no)

        try:
            created_path = _write_reconstructed_page_pdf(
                args.file,
                page_no,
                output_path,
                debug=args.debug or args.table_debug,
            )
        except Exception as exc:
            raise SystemExit(f"Failed to create reconstructed page PDF: {exc}")

        print(f"Reconstructed page PDF saved to: {created_path}")
        return

    if args.image_only_page is not None:
        try:
            page_no = int(args.image_only_page)
        except (TypeError, ValueError):
            raise SystemExit("--image-only-page must be an integer.")

        if args.pages is not None:
            raise SystemExit("--image-only-page cannot be combined with --pages.")

        if page_no < 1:
            raise SystemExit("--image-only-page must be >= 1.")

        output_path = args.image_only_output
        if not output_path:
            output_path = _image_only_output_path(args.file, page_no)

        try:
            created_path = _write_image_only_page_pdf(
                args.file,
                page_no,
                output_path,
                strip_markdown_lines=args.strip_markdown_lines,
                header_ratio=args.header_ratio,
                footer_ratio=args.footer_ratio,
            )
        except Exception as exc:
            raise SystemExit(f"Failed to create image-only page PDF: {exc}")

        print(f"Image-only page PDF saved to: {created_path}")
        return

    try:
        requested_pages = _parse_pages(args.pages)
    except ValueError as exc:
        raise SystemExit(f"Invalid --pages argument: {exc}")

    output = args.output or f"{Path(args.file).stem}_pages.jsonl"
    request_tables_markdown = args.tables_markdown is not None
    tables_markdown_output = (
        str(Path(output).with_name(f"{Path(output).stem}_tables.md"))
        if request_tables_markdown and args.tables_markdown == ""
        else args.tables_markdown
    )
    request_raw_components = args.raw_components is not None
    raw_components_output = (
        str(
            Path(output).with_name(
                f"{Path(output).stem}_raw_components.jsonl"
            )
        )
        if request_raw_components and args.raw_components == ""
        else args.raw_components
    )
    request_raw_page = args.raw_page is not None
    request_rawdict_page = (
        request_raw_page
        and isinstance(args.raw_page, str)
        and args.raw_page.lower() == "rawdict"
    )
    raw_page_output = (
        "raw.jsonl"
        if request_rawdict_page
        else str(
            Path(output).with_name(f"{Path(output).stem}_raw_page.jsonl")
        )
        if request_raw_page and args.raw_page == ""
        else args.raw_page
    )
    result = read_pdf(
        args.file,
        strip_watermarks=args.strip_watermarks,
        strip_headers=args.strip_headers,
        strip_footers=args.strip_footers,
        patterns=args.watermark_patterns,
        ratio_threshold=args.watermark_ratio,
        header_ratio=args.header_ratio,
        footer_ratio=args.footer_ratio,
        max_pages=args.max_pages,
        pages=requested_pages,
        preserve_newlines=args.preserve_newlines,
        extract_tables=args.find_tables,
        strip_markdown_lines=args.strip_markdown_lines,
        debug=args.debug,
        table_debug=args.debug or args.table_debug,
        table_mode=args.table_mode,
        return_pages_with_lines=request_raw_components,
        return_raw_pages=request_raw_page,
    )
    if request_raw_components and request_raw_page:
        records, pages_with_lines, raw_pages = result
    elif request_raw_components:
        records, pages_with_lines = result
        raw_pages = None
    elif request_raw_page:
        records, raw_pages = result
    else:
        records = result
    if args.raw_line_log:
        write_raw_line_log(records, args.raw_line_log)
    if request_raw_components and raw_components_output:
        write_raw_components(pages_with_lines, raw_components_output)
    if request_raw_page and raw_page_output:
        if request_rawdict_page:
            write_rawdict_pages(raw_pages or [], raw_page_output)
        else:
            write_raw_pages(raw_pages or [], raw_page_output)
    if args.legacy_page_jsonl:
        write_jsonl_pages(records, output)
        if request_raw_components or request_raw_page:
            print(f"Extracted {len(records)} pages -> {output}")
            if request_raw_components:
                print(f"Raw components -> {raw_components_output}")
            if request_raw_page:
                if request_rawdict_page:
                    print(f"Raw page rawdict -> {raw_page_output}")
                else:
                    print(f"Raw page objects -> {raw_page_output}")
        else:
            print(f"Extracted {len(records)} pages -> {output}")
    else:
        write_jsonl(records, output)
        total_lines = sum(
            len(record.get("regions", {}).get(region, []))
            for record in records
            for region in ("header", "body", "footer", "watermark")
        )
        total_tables = sum(record.get("table_count", 0) for record in records)
        if request_tables_markdown and tables_markdown_output:
            _write_tables_markdown(records, tables_markdown_output)
        if request_raw_components or request_raw_page:
            print(
                f"Extracted {len(records)} pages, {total_lines} text lines, {total_tables} tables -> {output}"
            )
            if request_raw_components:
                print(f"Raw components -> {raw_components_output}")
            if request_raw_page:
                if request_rawdict_page:
                    print(f"Raw page rawdict -> {raw_page_output}")
                else:
                    print(f"Raw page objects -> {raw_page_output}")
            if request_tables_markdown:
                print(f"Tables markdown -> {tables_markdown_output}")
        else:
            print(f"Extracted {len(records)} pages, {total_lines} text lines, {total_tables} tables -> {output}")
            if request_tables_markdown:
                print(f"Tables markdown -> {tables_markdown_output}")


if __name__ == "__main__":
    main()
