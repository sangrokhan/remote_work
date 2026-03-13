import argparse
import os
import json
import logging
import math
import re
import sys
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
    "C:/Windows/Fonts/Malgun.ttf",
    "C:/Windows/Fonts/malgun.ttc",
    "C:/Windows/Fonts/malgungothic.ttf",
    "C:/Windows/Fonts/malgungothicb.ttf",
    "C:/Windows/Fonts/malgungothicbd.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgunb.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
    "C:/Windows/Fonts/NanumMyeongjo.ttf",
    "C:/Windows/Fonts/나눔고딕.ttf",
)
_RECONSTRUCTION_KOREAN_FONT = None


def _get_registry_korean_font_candidates(font_markers):
    if os.name != "nt":
        return tuple()

    try:
        import winreg  # type: ignore
    except Exception:
        return tuple()

    markers = tuple(_normalize_font_match_key(marker) for marker in font_markers)
    marker_set = tuple(m for m in markers if m)
    if not marker_set:
        return tuple()

    possible_roots = (
        os.environ.get("WINDIR"),
        os.environ.get("SYSTEMROOT"),
        os.environ.get("windir"),
        os.environ.get("systemroot"),
    )

    registry_root_candidates = (
        "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Fonts",
        "SOFTWARE\\WOW6432Node\\Microsoft\\Windows NT\\CurrentVersion\\Fonts",
    )
    root_dirs = []
    for env_root in possible_roots:
        if env_root:
            root_dirs.append(Path(env_root) / "Fonts")
    root_dirs.extend([Path("C:/Windows/Fonts"), Path("C:/WINNT/Fonts"), Path("/mnt/c/Windows/Fonts")])
    root_dirs = [path for path in root_dirs if isinstance(path, Path) and path.is_dir()]
    if not root_dirs:
        return tuple()

    found = []
    for key_path in registry_root_candidates:
        key = None
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
        except OSError:
            continue
        index = 0
        try:
            while True:
                try:
                    value_name, value_data, _ = winreg.EnumValue(key, index)
                except OSError:
                    break
                index += 1

                if not isinstance(value_data, str):
                    continue
                if not value_data:
                    continue

                target_value = _normalize_font_match_key(value_name) + _normalize_font_match_key(value_data)
                if not any(marker in target_value for marker in marker_set):
                    continue

                data_path = Path(value_data)
                candidate_paths = []
                if data_path.is_absolute():
                    candidate_paths.append(data_path)
                else:
                    for root_dir in root_dirs:
                        candidate_paths.append(root_dir / value_data)

                for candidate in candidate_paths:
                    try:
                        candidate_key = _normalize_font_match_key(candidate)
                        matched_markers = [
                            marker for marker in marker_set
                            if marker and marker in candidate_key
                        ]
                        if candidate.is_file():
                            if _LOGGER.isEnabledFor(logging.DEBUG):
                                _LOGGER.debug(
                                    "Registry font candidate match: value_name=%r markers=%s path=%s",
                                    value_name,
                                    sorted(matched_markers),
                                    candidate,
                                )
                            found.append(str(candidate))
                    except OSError:
                        continue
        finally:
            try:
                winreg.CloseKey(key)
            except Exception:
                pass

    return tuple(found)


def _normalize_font_match_key(value):
    if value is None:
        return ""

    normalized = str(value).lower()
    normalized = normalized.replace("-", "").replace("_", "")
    return "".join(ch for ch in normalized if not ch.isspace())


def _normalize_font_path_candidates(fontfile):
    if not fontfile:
        return []

    raw = str(fontfile).strip()
    if not raw:
        return []

    candidates = []
    seen = set()

    def add_candidate(value):
        if not value:
            return
        text = str(value).strip()
        if not text:
            return
        if text in seen:
            return
        seen.add(text)
        candidates.append(text)

    initial_variants = [
        raw,
        str(Path(raw)),
        os.fspath(Path(raw).expanduser()),
        os.path.normpath(raw),
        os.path.normpath(str(Path(raw).expanduser())),
    ]
    for variant in initial_variants:
        add_candidate(variant)

    expanded = list(candidates)
    for value in expanded:
        add_candidate(value.replace("/", "\\"))
        add_candidate(value.replace("\\", "/"))

    if os.name != "nt":
        drive_match = re.match(r"^([a-zA-Z]):[/\\\\](.*)$", raw)
        if drive_match:
            drive = drive_match.group(1).lower()
            relative = drive_match.group(2).replace("\\", "/").lstrip("/")
            add_candidate(f"/mnt/{drive}/{relative}")

    return candidates


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


def _get_reconstruct_fontfile(fontfile_override=None):
    global _RECONSTRUCTION_KOREAN_FONT

    if fontfile_override:
        override_path = Path(fontfile_override)
        override_exists = override_path.is_file()
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("Override fontfile check: path=%s exists=%s", fontfile_override, override_exists)
        if override_exists:
            return str(override_path)
        _LOGGER.warning(
            "Provided reconstruction fontfile does not exist or is not a file: %s",
            fontfile_override,
        )

    if _RECONSTRUCTION_KOREAN_FONT is not None:
        return _RECONSTRUCTION_KOREAN_FONT

    for font_path in _KOREAN_FONT_HINTS:
        try:
            exists = Path(font_path).is_file()
            if _LOGGER.isEnabledFor(logging.DEBUG):
                _LOGGER.debug("Korean font hint check: path=%s exists=%s", font_path, exists)
            if exists:
                _RECONSTRUCTION_KOREAN_FONT = str(font_path)
                return _RECONSTRUCTION_KOREAN_FONT
        except OSError:
            continue

    for font_path in _get_registry_korean_font_candidates(font_name_markers=(
        "noto",
        "nanum",
        "malgun",
        "malgungothic",
        "맑은고딕",
        "나눔",
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
        "yoon",
        "seoul",
        "gothic",
    )):
        try:
            is_file = Path(font_path).is_file()
            if _LOGGER.isEnabledFor(logging.DEBUG):
                _LOGGER.debug("Registry-provided Korean font check: path=%s exists=%s", font_path, is_file)
            if is_file:
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
        "C:/WINNT/Fonts",
        "/mnt/c/Windows/Fonts",
        "/mnt/c/WINNT/Fonts",
    )
    font_name_markers = tuple(
        _normalize_font_match_key(marker) for marker in (
        "noto",
        "nanum",
        "malgun",
        "malgungothic",
        "맑은고딕",
        "applegothic",
        "applegothicneoregular",
        "batang",
        "gulim",
        "dotum",
        "gulim",
        "msung",
        "msjh",
        "msyhl",
        "hei",
        "microsoftyi",
        "wqy",
        "sourcehans",
        "yoon",
        "seoul",
        "gothic",
        )
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
                    path_str = _normalize_font_match_key(path)
                except OSError:
                    continue
                matched_markers = [marker for marker in font_name_markers if marker in path_str]
                if matched_markers:
                    if _LOGGER.isEnabledFor(logging.DEBUG):
                        _LOGGER.debug(
                            "Korean font marker match in %s: markers=%s path=%s",
                            root_path,
                            sorted(matched_markers),
                            path,
                        )
                    try:
                        _RECONSTRUCTION_KOREAN_FONT = str(path)
                        return _RECONSTRUCTION_KOREAN_FONT
                    except OSError:
                        continue
            for path in root.rglob("*"):
                try:
                    if not path.is_file():
                        continue
                    if path.suffix.lower() not in {".ttf", ".ttc", ".otf"}:
                        continue
                except OSError:
                    continue
                try:
                    name = _normalize_font_match_key(path)
                except OSError:
                    continue
                matched_markers = [marker for marker in font_name_markers if marker in name]
                if matched_markers:
                    if _LOGGER.isEnabledFor(logging.DEBUG):
                        _LOGGER.debug(
                            "Korean font filename fallback match in %s: markers=%s path=%s",
                            root_path,
                            sorted(matched_markers),
                            path,
                        )
                    _RECONSTRUCTION_KOREAN_FONT = str(path)
                    return _RECONSTRUCTION_KOREAN_FONT

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


def _dedupe_pages(page_numbers):
    if not page_numbers:
        return []

    deduped = []
    seen = set()
    for raw_page_no in page_numbers:
        page_no = int(raw_page_no)
        if page_no in seen:
            continue
        seen.add(page_no)
        deduped.append(page_no)

    return deduped


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
    if isinstance(first, pymupdf.Rect):
        return [float(first.x0), float(first.y0), float(first.x1), float(first.y1)]

    if isinstance(first, (list, tuple)) and len(first) == 4:
        try:
            return [float(first[0]), float(first[1]), float(first[2]), float(first[3])]
        except (TypeError, ValueError):
            return None

    if len(args) == 2 and isinstance(args[1], (int, float)) and isinstance(
        first, (list, tuple)
    ) and len(first) == 4:
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




def _extract_page_tables(
    page,
    page_no,
    source,
    lines=None,
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
        if debug:
            _LOGGER.info(
                "No tables found with PyMuPDF find_tables: source=%s page=%s",
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
            rows = []

        if not rows:
            # no table content recovered from detector
            continue

        table_rows = len(rows)
        table_cols = 0
        for row in rows:
            if isinstance(row, (list, tuple)):
                table_cols = max(table_cols, len(row))

        x0, y0, x1, y1 = [float(v) for v in bbox]
        components = {
            "bbox": [x0, y0, x1, y1],
            "row_lines": [
                {"orientation": "horizontal", "x0": x0, "y0": y0, "x1": x1, "y1": y0},
                {"orientation": "horizontal", "x0": x0, "y0": y1, "x1": x1, "y1": y1},
            ],
            "vertical_lines": [
                {"orientation": "vertical", "x0": x0, "y0": y0, "x1": x0, "y1": y1},
                {"orientation": "vertical", "x0": x1, "y0": y0, "x1": x1, "y1": y1},
            ],
        }

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

        if row_count != table_rows or col_count != table_cols:
            components = {
                "bbox": [x0, y0, x1, y1],
                "row_lines": components.get("row_lines", []),
                "vertical_lines": components.get("vertical_lines", []),
            }
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


def _extract_page_lines(
    page,
    page_no,
    source,
    header_ratio,
    footer_ratio,
    preserve_newlines=False,
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


def _draw_raw_spans_on_page(
    out_page,
    spans,
    source,
    page_no,
    debug=False,
    korean_fontname=None,
):
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
        text_has_korean = _contains_korean(text)

        inserted = False
        font_candidates = [font]
        if text_has_korean and korean_fontname:
            font_candidates.append(korean_fontname)
        font_candidates.extend(("helv", "sans-serif"))

        seen = set()
        deduped = []
        for candidate in font_candidates:
            if not candidate:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)

        for font_name in deduped:
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
                "Text span insert failed: source=%s page=%s font=%r has_korean=%s size=%s rotation=%s text=%r",
                source,
                page_no,
                font,
                text_has_korean,
                size,
                _round_float(rotation),
                text[:120],
            )


def _iter_text_dict_lines(raw_text):
    raw_blocks = raw_text.get("blocks") if isinstance(raw_text, dict) else []
    line_no = 0
    for block in raw_blocks:
        if not isinstance(block, dict) or block.get("type") != 0:
            continue
        for line in block.get("lines", []) or []:
            if not isinstance(line, dict):
                continue
            line_no += 1
            yield line_no, line


def _ensure_page_font_resource(
    rendered_pdf,
    source_path,
    page_no,
    korean_fontfile=None,
    debug=False,
):
    fontfile_for_korean = _get_reconstruct_fontfile(fontfile_override=korean_fontfile)
    korean_fontname = None
    if not hasattr(rendered_pdf, "insert_font"):
        if debug:
            _LOGGER.debug(
                "Rendered target has no insert_font; cannot register fallback Korean fontfile=%s source=%s page=%s",
                fontfile_for_korean,
                source_path,
                page_no,
            )
        return None

    font_candidates = []
    seen_candidates = set()

    def add_font_candidate(value):
        for candidate in _normalize_font_path_candidates(value):
            try:
                if candidate not in seen_candidates:
                    font_candidates.append(candidate)
                    seen_candidates.add(candidate)
            except TypeError:
                continue

    if korean_fontfile:
        add_font_candidate(korean_fontfile)
    if fontfile_for_korean:
        add_font_candidate(fontfile_for_korean)

    # add explicit static hints and registry candidates as backup when override is missing or invalid
    for candidate in _KOREAN_FONT_HINTS:
        add_font_candidate(candidate)

    for candidate in _get_registry_korean_font_candidates(
        font_name_markers=(
            "noto",
            "nanum",
            "malgun",
            "malgungothic",
            "맑은고딕",
            "나눔",
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
            "yoon",
            "seoul",
            "gothic",
        )
    ):
        add_font_candidate(candidate)

    if _LOGGER.isEnabledFor(logging.DEBUG):
        _LOGGER.debug(
            "Resolved Korean fontfile candidates: source=%s page=%s count=%s candidates=%s",
            source_path,
            page_no,
            len(font_candidates),
            font_candidates,
        )

    if not font_candidates:
        if debug:
            _LOGGER.debug(
                "No Korean fallback fontfile found; text may use Latin-only fonts: source=%s page=%s",
                source_path,
                page_no,
            )
        return None

    font_name = "reconstruct_korean"
    last_failure = None
    for fontfile in font_candidates:
        if not fontfile:
            continue
        try:
            font_path = Path(fontfile)
            candidate_exists = font_path.is_file()
            if _LOGGER.isEnabledFor(logging.DEBUG):
                _LOGGER.debug(
                    "Korean font candidate check: path=%s exists=%s source=%s page=%s",
                    font_path,
                    candidate_exists,
                    source_path,
                    page_no,
                )
            if not candidate_exists:
                if debug:
                    _LOGGER.debug(
                        "Fallback font path is not a file: %s source=%s page=%s",
                        fontfile,
                        source_path,
                        page_no,
                    )
                continue
            font_bytes = None
            candidate_used = None

            # Try path-based insert first (faster path for normal installations)
            try:
                korean_fontname = rendered_pdf.insert_font(
                    fontname=font_name,
                    fontfile=str(font_path),
                )
                candidate_used = str(font_path)
                if _LOGGER.isEnabledFor(logging.DEBUG):
                    _LOGGER.debug(
                        "Font insert(fontfile) succeeded: candidate=%s returned=%r",
                        candidate_used,
                        korean_fontname,
                    )
            except TypeError:
                # older versions may not accept fontfile keyword
                if _LOGGER.isEnabledFor(logging.DEBUG):
                    _LOGGER.debug(
                        "Font insert(fontfile) unsupported signature, trying fontbuffer: candidate=%s",
                        font_path,
                    )
            except Exception as exc:
                last_failure = ("fontfile", str(font_path), str(exc))
                if debug:
                    _LOGGER.debug(
                        "Font insert with fontfile failed: candidate=%s source=%s page=%s error=%s",
                        font_path,
                        source_path,
                        page_no,
                        exc,
                    )

            if not korean_fontname:
                try:
                    if font_bytes is None:
                        with open(font_path, "rb") as font_handle:
                            font_bytes = font_handle.read()
                    korean_fontname = rendered_pdf.insert_font(
                        fontname=font_name,
                        fontbuffer=font_bytes,
                    )
                    candidate_used = str(font_path)
                except TypeError:
                    pass
                except Exception as exc:
                    if debug:
                        _LOGGER.debug(
                            "Font insert with fontbuffer failed: candidate=%s source=%s page=%s error=%s",
                            font_path,
                            source_path,
                            page_no,
                            exc,
                        )
                else:
                    if _LOGGER.isEnabledFor(logging.DEBUG):
                        _LOGGER.debug(
                            "Font insert(fontbuffer) succeeded: candidate=%s returned=%r",
                            font_path,
                            korean_fontname,
                        )

            if not korean_fontname:
                try:
                    if font_bytes is None:
                        with open(font_path, "rb") as font_handle:
                            font_bytes = font_handle.read()
                    korean_fontname = rendered_pdf.insert_font(
                        fontname=font_name,
                        fontbuffer=font_bytes,
                        set_simple=False,
                    )
                    candidate_used = str(font_path)
                except TypeError:
                    pass
                except Exception as exc:
                    if debug:
                        _LOGGER.debug(
                            "Font insert with fontbuffer(set_simple=False) failed: candidate=%s source=%s page=%s error=%s",
                            font_path,
                            source_path,
                            page_no,
                            exc,
                        )
                else:
                    if _LOGGER.isEnabledFor(logging.DEBUG):
                        _LOGGER.debug(
                            "Font insert(fontbuffer,set_simple=False) succeeded: candidate=%s returned=%r",
                            font_path,
                            korean_fontname,
                        )

            if korean_fontname:
                if debug:
                    _LOGGER.debug(
                        "Registered Korean fallback fontname=%s fontfile=%s source=%s page=%s",
                        korean_fontname,
                        candidate_used,
                        source_path,
                        page_no,
                    )
                return korean_fontname
            last_failure = ("not_registered", str(font_path), "all attempts returned empty fontname")
        except OSError:
            last_failure = ("os_error", str(fontfile), "path check/read error")
            continue
        except Exception as exc:
            last_failure = ("unknown", str(fontfile), str(exc))

    if debug:
        _LOGGER.debug(
            "Failed to register fallback Korean font: attempted candidates=%s source=%s page=%s",
            font_candidates,
            source_path,
            page_no,
        )
        if last_failure:
            _LOGGER.debug(
                "Last font registration failure detail: method=%s path=%s error=%s",
                last_failure[0],
                last_failure[1],
                last_failure[2],
            )
    if debug:
        _LOGGER.debug(
            "Fallback fontfile registration failed; using fallback fontname=%s for source=%s page=%s",
            "helv",
            source_path,
            page_no,
        )

    # Return a safe baseline font as final fallback. `None` is not returned here so
    # reconstruction drawing can proceed using a known latin font while still
    # attempting Korean fallback registration when possible.
    return "helv"



def _normalize_shape_point_from_args(args):
    if not args:
        return None
    if len(args) == 1:
        return _to_point(args[0])
    try:
        return float(args[-2]), float(args[-1])
    except (TypeError, ValueError):
        return None


def _normalize_shape_rect(values):
    if not isinstance(values, (list, tuple)):
        return None
    if len(values) == 1 and isinstance(values[0], pymupdf.Rect):
        return (
            float(values[0].x0),
            float(values[0].y0),
            float(values[0].x1),
            float(values[0].y1),
        )
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        inner = values[0]
        if len(inner) == 4:
            try:
                return (
                    float(inner[0]),
                    float(inner[1]),
                    float(inner[2]),
                    float(inner[3]),
                )
            except (TypeError, ValueError):
                return None
    if len(values) == 2 and isinstance(values[0], pymupdf.Rect):
        return (
            float(values[0].x0),
            float(values[0].y0),
            float(values[0].x1),
            float(values[0].y1),
        )
    if len(values) != 4:
        return None
    try:
        return float(values[0]), float(values[1]), float(values[2]), float(values[3])
    except (TypeError, ValueError):
        return None


def _normalize_shape_op(raw_op):
    if isinstance(raw_op, (bytes, bytearray)):
        try:
            return raw_op.decode("utf-8")
        except Exception:
            return str(raw_op)
    return raw_op


def _shorten_shape_args(values, max_items=8):
    if values is None:
        return "None"
    if not isinstance(values, (list, tuple)):
        return f"type={type(values).__name__} repr={values!r}"
    preview = list(values[:max_items])
    if len(values) > max_items:
        return f"len={len(values)} head={preview!r}..."
    return f"len={len(values)} repr={preview!r}"


def _shape_line_style(drawing):
    close_path_raw = drawing.get("closePath", True)
    if isinstance(close_path_raw, (int, float)):
        close_path = bool(close_path_raw)
    elif isinstance(close_path_raw, bool):
        close_path = close_path_raw
    elif close_path_raw is None:
        close_path = True
    else:
        close_path = bool(close_path_raw)

    line_cap_raw = drawing.get("lineCap")
    if isinstance(line_cap_raw, bool):
        line_cap = int(line_cap_raw)
    elif isinstance(line_cap_raw, (int, float)):
        line_cap = int(line_cap_raw)
    else:
        line_cap = None

    line_join_raw = drawing.get("lineJoin")
    if isinstance(line_join_raw, bool):
        line_join = int(line_join_raw)
    elif isinstance(line_join_raw, (int, float)):
        line_join = int(line_join_raw)
    else:
        line_join = None

    dashes = drawing.get("dashes")
    if dashes is not None:
        if isinstance(dashes, (list, tuple)):
            try:
                dashes = [float(value) for value in dashes]
            except (TypeError, ValueError):
                dashes = None
        else:
            dashes = None

    return {
        "closePath": close_path,
        "lineCap": line_cap,
        "lineJoin": line_join,
        "dashes": dashes,
    }


def _shape_finish_kwargs(
    fill_color,
    stroke_color,
    line_width,
    stroke_opacity,
    fill_opacity,
    shape_style,
    fill_only_op=False,
    include_fill_when_none=False,
):
    stroke_style = {
        "closePath": shape_style.get("closePath", True),
    }

    if fill_color is not None or include_fill_when_none:
        stroke_style["fill"] = fill_color
        stroke_style["fill_opacity"] = fill_opacity

    if not fill_only_op:
        stroke_style["color"] = stroke_color
        stroke_style["width"] = line_width
        stroke_style["stroke_opacity"] = stroke_opacity
        if shape_style.get("dashes") is not None:
            stroke_style["dashes"] = shape_style["dashes"]
        if shape_style.get("lineCap") is not None:
            stroke_style["lineCap"] = shape_style["lineCap"]
        if shape_style.get("lineJoin") is not None:
            stroke_style["lineJoin"] = shape_style["lineJoin"]
    else:
        stroke_style["width"] = 0

    return stroke_style


def _render_shape_drawing(out_page, drawing, page_no, drawing_index=None, debug=False):
    if not isinstance(drawing, dict):
        if debug:
            _LOGGER.debug(
                "Drawing[%s] skip: not dict type=%s page=%s",
                drawing_index,
                type(drawing).__name__,
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
    fill_color = _to_rgb_color(fill_color_raw) if fill_color_raw is not None else None
    stroke_color = _to_rgb_color(drawing.get("color"), default=(0.0, 0.0, 0.0))
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
    has_stroke_fill_op = False
    raw_op_types = Counter()
    non_fill_op_counts = Counter()
    for item_idx, item in enumerate(items):
        if not item or not isinstance(item, (list, tuple)) or not item:
            if debug and item is not None:
                _LOGGER.debug(
                    "Drawing[%s] item=%s empty-or-invalid at pre-scan page=%s",
                    drawing_index,
                    item_idx,
                    page_no,
                )
            continue

        raw_op = item[0]
        op = _normalize_shape_op(raw_op)
        raw_op_types[type(raw_op).__name__] += 1
        op_key = op if isinstance(op, str) else str(op)
        op_counts[op_key] += 1
        if op not in fill_ops:
            non_fill_op_counts[op_key] += 1

        if op in fill_ops:
            has_fill_op = True
            if op in ("b", "B", "b*"):
                has_stroke_fill_op = True

        if debug and item_idx < 80:
            _LOGGER.debug(
                "Drawing[%s] item=%s precheck raw_op=%r norm_op=%r arg=%s page=%s",
                drawing_index,
                item_idx,
                raw_op,
                op,
                _shorten_shape_args(item[1:]),
                page_no,
            )

    if debug:
        _LOGGER.debug(
            "Drawing[%s] start page=%s fill_key=%s raw_fill=%r norm_fill=%r stroke=%r linewidth=%s fill_op=%s fill_opacity=%r ops=%s non_fill_ops=%s raw_op_types=%s closePath=%s",
            drawing_index,
            page_no,
            fill_key_present,
            fill_color_raw,
            fill_color,
            stroke_color,
            _round_float(line_width),
            has_fill_op,
            fill_opacity,
            " ".join(f"{k}:{op_counts[k]}" for k in sorted(op_counts.keys())),
            " ".join(f"{k}:{non_fill_op_counts[k]}" for k in sorted(non_fill_op_counts.keys())),
            " ".join(f"{k}:{raw_op_types[k]}" for k in sorted(raw_op_types.keys())),
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

    def _render_debug_rect(values):
        if values is None:
            return "None"
        if not isinstance(values, (list, tuple)):
            return f"type={type(values).__name__} repr={values!r}"
        if len(values) == 1 and isinstance(values[0], (list, tuple)):
            return f"nested-len={len(values[0])} repr={list(values[0])!r}"
        return f"len={len(values)} repr={list(values)!r}"

    if use_shape:
        if debug:
            _LOGGER.debug("Drawing[%s] use Shape API page=%s", drawing_index, page_no)

        current = None
        has_geom = False
        fill_op_executed = False
        shape_cmds = 0
        for item_idx, item in enumerate(items):
            if not item:
                if debug:
                    _LOGGER.debug(
                        "Drawing[%s] shape item=%s empty skip page=%s",
                        drawing_index,
                        item_idx,
                        page_no,
                    )
                continue
            raw_op = item[0]
            op = _normalize_shape_op(raw_op)
            args = item[1:]
            if debug:
                _LOGGER.debug(
                    "Drawing[%s] shape item=%s raw_op=%r norm_op=%r arg=%s page=%s",
                    drawing_index,
                    item_idx,
                    raw_op,
                    op,
                    _shorten_shape_args(args),
                    page_no,
                )

            if op == "m":
                point = _normalize_shape_point_from_args(args)
                if point is not None:
                    shape.move_to(point[0], point[1])
                    current = point
                    has_geom = True
                    shape_cmds += 1
                else:
                    if debug:
                        _LOGGER.debug(
                            "Drawing[%s] skip move invalid point args=%s page=%s",
                            drawing_index,
                            _shorten_shape_args(args),
                            page_no,
                        )
                continue

            if op == "l":
                point = _normalize_shape_point_from_args(args)
                if current is not None and point is not None:
                    shape.line_to(point[0], point[1])
                    current = point
                    has_geom = True
                    shape_cmds += 1
                else:
                    if debug:
                        _LOGGER.debug(
                            "Drawing[%s] skip line current=%s point=%s page=%s",
                            drawing_index,
                            current,
                            point,
                            page_no,
                        )
                continue

            if op == "c":
                p3 = _normalize_shape_point_from_args(args[-2:]) if len(args) >= 2 else None
                if p3 is None:
                    if debug:
                        _LOGGER.debug(
                            "Drawing[%s] curve skip: invalid end point args=%s page=%s",
                            drawing_index,
                            _shorten_shape_args(args),
                            page_no,
                        )
                    continue
                p1 = _normalize_shape_point_from_args(args[0:2]) if len(args) >= 2 else current
                p2 = _normalize_shape_point_from_args(args[2:4]) if len(args) >= 4 else current
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
                shape_cmds += 1
                continue

            if op in ("v", "y"):
                point = _normalize_shape_point_from_args(args)
                if point is not None:
                    if current is not None:
                        shape.line_to(point[0], point[1])
                    current = point
                    has_geom = True
                    shape_cmds += 1
                else:
                    if debug:
                        _LOGGER.debug(
                            "Drawing[%s] curve fallback skip: invalid point args=%s page=%s",
                            drawing_index,
                            _shorten_shape_args(args),
                            page_no,
                        )
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
                else:
                    if debug:
                        _LOGGER.debug(
                            "Drawing[%s] shape.close_path success current=%s page=%s",
                            drawing_index,
                            current,
                            page_no,
                        )
                current = None
                continue

            if op == "re":
                rect_values = _normalize_shape_rect(args)
                if rect_values is None:
                    if debug:
                        _LOGGER.debug(
                            "Drawing[%s] skip invalid rect args=%r page=%s",
                            drawing_index,
                            args,
                            page_no,
                        )
                        _LOGGER.debug(
                            "Drawing[%s] rect decode=%s normalized_try=%r page=%s",
                            drawing_index,
                            _render_debug_rect(args),
                            _normalize_shape_rect(args),
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
                    shape_cmds += 1
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
                fill_only_op = op in ("f", "F", "f*")
                shape_style = _shape_line_style(drawing)
                shape_finish_kwargs = _shape_finish_kwargs(
                    fill_color=fill_color,
                    stroke_color=stroke_color,
                    line_width=line_width,
                    stroke_opacity=stroke_opacity,
                    fill_opacity=fill_opacity,
                    shape_style=shape_style,
                    fill_only_op=fill_only_op,
                )

                try:
                    if debug:
                        _LOGGER.debug(
                            "Drawing[%s] shape.finish kwargs color=%r fill=%r width=%r closePath=%r fill_opacity=%r stroke_opacity=%r dashes=%r lineCap=%r lineJoin=%r",
                            drawing_index,
                            stroke_color,
                            fill_color,
                            _round_float(shape_finish_kwargs.get("width")),
                            shape_style.get("closePath"),
                            fill_opacity,
                            stroke_opacity,
                            shape_style.get("dashes") if not fill_only_op else None,
                            shape_style.get("lineCap") if not fill_only_op else None,
                            shape_style.get("lineJoin") if not fill_only_op else None,
                        )
                    shape.finish(**shape_finish_kwargs)
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
                        fallback_shape_kwargs = {
                            "width": 0,
                            "closePath": shape_style.get("closePath", True),
                        }
                        fallback_shape_kwargs = _shape_finish_kwargs(
                            fill_color=fill_color,
                            stroke_color=stroke_color,
                            line_width=line_width,
                            stroke_opacity=stroke_opacity,
                            fill_opacity=fill_opacity,
                            shape_style=shape_style,
                            fill_only_op=fill_only_op,
                            include_fill_when_none=not fill_only_op,
                        )
                        shape.finish(**fallback_shape_kwargs)
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

            if op not in fill_ops and debug:
                _LOGGER.debug(
                    "Drawing[%s] unsupported Shape op=%r args=%r page=%s",
                    drawing_index,
                    op,
                    args,
                    page_no,
                )

        if use_shape:
            if debug:
                _LOGGER.debug(
                    "Drawing[%s] shape loop summary has_fill_op=%s has_geom=%s cmds=%s shape_disabled=%s",
                    drawing_index,
                    fill_op_executed,
                    has_geom,
                    shape_cmds,
                    shape_disable_reason,
                )
            if fill_key_present and fill_color is not None and not fill_op_executed:
                if debug:
                    _LOGGER.debug(
                        "Drawing[%s] fill key present but no fill op; trying fallback fill commit page=%s",
                        drawing_index,
                        page_no,
                    )
                shape_style = _shape_line_style(drawing)
                try:
                    if debug:
                        _LOGGER.debug(
                            "Drawing[%s] fill-on-commit kwargs color=%r fill=%r width=%r closePath=%r fill_opacity=%r stroke_opacity=%r dashes=%r lineCap=%r lineJoin=%r",
                            drawing_index,
                            stroke_color,
                            fill_color,
                            _round_float(line_width),
                            shape_style.get("closePath"),
                            fill_opacity,
                            stroke_opacity,
                            shape_style.get("dashes"),
                            shape_style.get("lineCap"),
                            shape_style.get("lineJoin"),
                        )
                    shape_finish_kwargs = _shape_finish_kwargs(
                        fill_color=fill_color,
                        stroke_color=stroke_color,
                        line_width=line_width,
                        stroke_opacity=stroke_opacity,
                        fill_opacity=fill_opacity,
                        shape_style=shape_style,
                        fill_only_op=True,
                        include_fill_when_none=True,
                    )
                    shape.finish(**shape_finish_kwargs)
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
    for item_idx, item in enumerate(items):
        if not item:
            if debug:
                _LOGGER.debug(
                    "Drawing[%s] fallback item=%s empty skip page=%s",
                    drawing_index,
                    item_idx,
                    page_no,
                )
            continue
        raw_op = item[0]
        op = _normalize_shape_op(raw_op)
        args = item[1:]
        if isinstance(op, str):
            fallback_ops[op] += 1
        if debug and item_idx < 120:
            _LOGGER.debug(
                "Drawing[%s] fallback item=%s raw_op=%r norm_op=%r args=%s page=%s",
                drawing_index,
                item_idx,
                raw_op,
                op,
                _shorten_shape_args(args),
                page_no,
            )

        if op == "m":
            current = _normalize_shape_point_from_args(args)
            if debug:
                _LOGGER.debug(
                    "Drawing[%s] fallback move point=%r page=%s",
                    drawing_index,
                    current,
                    page_no,
                )
            continue

        if op == "l":
            target = _normalize_shape_point_from_args(args)
            if debug and target is None:
                _LOGGER.debug(
                    "Drawing[%s] fallback line invalid target args=%s current=%s page=%s",
                    drawing_index,
                    _shorten_shape_args(args),
                    current,
                    page_no,
                )
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
            current = _normalize_shape_point_from_args(args)
            continue

        if op == "h":
            current = None
            continue

        if op == "c":
            target = _normalize_shape_point_from_args(args)
            if current is not None and target is not None:
                if debug:
                    _LOGGER.debug(
                        "Drawing[%s] fallback curve fallback target=%r current=%r page=%s",
                        drawing_index,
                        target,
                        current,
                        page_no,
                    )
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
            rect_values = _normalize_shape_rect(args)
            if not rect_values:
                if debug:
                    _LOGGER.debug(
                        "Drawing[%s] fallback invalid rect args=%r page=%s",
                        drawing_index,
                        args,
                        page_no,
                    )
                    _LOGGER.debug(
                        "Drawing[%s] fallback rect decode=%s normalized_try=%r page=%s",
                        drawing_index,
                        _render_debug_rect(args),
                        _normalize_shape_rect(args),
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
            if debug:
                _LOGGER.debug(
                    "Drawing[%s] fallback draw_rect call fill=%r fill_opacity=%r line_width=%r stroke=%r args=%s",
                    drawing_index,
                    fill_color,
                    fill_opacity,
                    line_width,
                    stroke_color,
                    _shorten_shape_args(args),
                )
            fallback_draw_rect_kwargs = {
                "fill": fill_color,
                "fill_opacity": fill_opacity if fill_opacity else 1.0,
            }
            if fill_color is None or has_stroke_fill_op:
                fallback_draw_rect_kwargs["color"] = stroke_color
                fallback_draw_rect_kwargs["width"] = line_width
            try:
                out_page.draw_rect(
                    pymupdf.Rect(x0, y0, x1, y1),
                    **fallback_draw_rect_kwargs,
                )
            except TypeError:
                try:
                    out_page.draw_rect(
                        pymupdf.Rect(x0, y0, x1, y1),
                        **fallback_draw_rect_kwargs,
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


def _collect_watermark_line_filter(
    page,
    page_no,
    source_text,
    enabled,
    header_ratio,
    footer_ratio,
    watermark_angle=_WATERMARK_ROTATION_DEGREE,
    watermark_tolerance=_WATERMARK_ROTATION_TOLERANCE,
    debug=False,
):
    if not enabled:
        return {
            "enabled": False,
            "removed_line_numbers": set(),
            "total_lines": 0,
            "watermark_lines": 0,
            "watermark_line_ratio": 0.0,
            "watermark_rotation": 0,
            "removed_count": 0,
            "kept_count": 0,
            "removed_lines": [],
        }

    try:
        target_lines = _extract_page_lines(
            page=page,
            page_no=page_no,
            source=source_text,
            header_ratio=header_ratio,
            footer_ratio=footer_ratio,
            preserve_newlines=False,
            debug=debug,
        )
        watermark_result = _remove_watermark_lines(
            target_lines,
            watermark_angle=watermark_angle,
            watermark_tolerance=watermark_tolerance,
            source=source_text,
            page_no=page_no,
            debug=debug,
        )
        removed_lines = watermark_result["removed_lines"]
        removed_line_numbers = watermark_result["removed_line_numbers"]
        watermark_lines = watermark_result["removed_count"]
        total_lines = len(target_lines)
        watermark_line_ratio = (
            round(watermark_lines / float(total_lines), 4) if watermark_lines and total_lines else 0.0
        )
        watermark_rotation = len(
            [item for item in removed_lines if item.get("reason") == "watermark-rotation"]
        )

        if debug:
            _LOGGER.debug(
                "Watermark line filter config: source=%s page=%s enabled=%s rotation=%s tolerance=%s header_ratio=%s footer_ratio=%s",
                source_text,
                page_no,
                True,
                _round_float(watermark_angle),
                _round_float(watermark_tolerance),
                _round_float(header_ratio),
                _round_float(footer_ratio),
            )
            _LOGGER.debug(
                "Watermark line filter summary: source=%s page=%s total_lines=%s watermark_lines=%s ratio=%s rotation=%s",
                source_text,
                page_no,
                total_lines,
                watermark_lines,
                watermark_line_ratio,
                watermark_rotation,
            )

        return {
            "enabled": True,
            "removed_line_numbers": removed_line_numbers,
            "total_lines": total_lines,
            "watermark_lines": watermark_lines,
            "watermark_line_ratio": watermark_line_ratio,
            "watermark_rotation": watermark_rotation,
            "removed_count": watermark_result["removed_count"],
            "kept_count": watermark_result["kept_count"],
            "removed_lines": removed_lines,
        }
    except Exception:
        if debug:
            _LOGGER.debug(
                "Watermark line filter setup failed: source=%s page=%s fallback=insert-all",
                source_text,
                page_no,
                exc_info=True,
            )

        return {
            "enabled": True,
            "removed_line_numbers": set(),
            "total_lines": 0,
            "watermark_lines": 0,
            "watermark_line_ratio": 0.0,
            "watermark_rotation": 0,
            "removed_count": 0,
            "kept_count": 0,
            "removed_lines": [],
        }


def _is_watermark_line(
    line,
    watermark_angle,
    watermark_tolerance,
):
    raw_text = _normalize_line(line.get("raw") or "")
    if not raw_text:
        return False, "empty-text"

    rotation = line.get("rotation")
    location = line.get("location", "body")
    if _is_rotation_match(rotation, watermark_angle, watermark_tolerance) and location == "body":
        return True, "watermark-rotation"

    return False, None


def _remove_watermark_lines(
    lines,
    watermark_angle,
    watermark_tolerance,
    source=None,
    page_no=None,
    debug=False,
):
    removed_lines = []
    removed_line_numbers = set()
    removed_count = 0
    kept_count = 0

    for line in lines:
        should_remove, removed_reason = _is_watermark_line(
            line=line,
            watermark_angle=watermark_angle,
            watermark_tolerance=watermark_tolerance,
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
            if debug and source is not None and page_no is not None:
                _LOGGER.debug(
                    "Removed watermark candidate: source=%s page=%s line=%s location=%s reason=%s rotation=%s text=%r",
                    source,
                    page_no,
                    line.get("line"),
                    line.get("location"),
                    removed_reason,
                    line.get("rotation"),
                    _normalize_line(line.get("raw") or "")[:120],
                )
            continue

        kept_count += 1

    return {
        "removed_count": removed_count,
        "removed_line_numbers": removed_line_numbers,
        "removed_lines": removed_lines,
        "kept_count": kept_count,
    }


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
    extract_tables=False,
    debug=False,
    table_debug=False,
    table_mode="auto",
    capture_raw_pages=False,
    raw_text_mode="rawdict",
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
            debug=debug,
        )
        for line in lines:
            line["source"] = source
        tables = []
        if extract_tables:
            tables = _extract_page_tables(
                page,
                page_no,
                source,
                lines=lines,
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
                    raw_text_mode=raw_text_mode,
                )
            )
        extracted.append((page_no, lines, tables, []))

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


def _extract_page_raw_payload(page, page_no, source, debug=False, raw_text_mode="rawdict"):
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

    try:
        if raw_text_mode == "rawjson":
            rawjson_text = page.get_text("rawjson")
            try:
                payload["text_rawjson"] = _to_json_value(json.loads(rawjson_text))
            except Exception:
                payload["text_rawjson"] = _to_json_value(rawjson_text)
        else:
            payload["text_rawdict"] = _to_json_value(page.get_text("rawdict"))
    except Exception as exc:
        if debug:
            _LOGGER.debug(
                "Failed to extract raw text mode=%s: source=%s page=%s error=%s",
                raw_text_mode,
                source,
                page_no,
                exc,
            )
        if raw_text_mode == "rawjson":
            payload["text_rawjson"] = {"error": str(exc)}
        else:
            payload["text_rawdict"] = {"error": str(exc)}

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


def _write_reconstructed_page_pdf(
    source_path,
    page_no,
    output_path,
    remove_watermark=False,
    watermark_angle=_WATERMARK_ROTATION_DEGREE,
    watermark_tolerance=_WATERMARK_ROTATION_TOLERANCE,
    header_ratio=0.08,
    footer_ratio=0.08,
    korean_fontfile=None,
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
        watermark_skip_stats = {
            "enabled": bool(remove_watermark),
            "total_lines": 0,
            "watermark_lines": 0,
            "watermark_line_ratio": 0.0,
            "watermark_rotation": 0,
            "span_total_count": 0,
            "line_skip_count": 0,
        }

        watermark_filter = _collect_watermark_line_filter(
            page=page,
            page_no=page_no,
            source_text=source_text,
            enabled=remove_watermark,
            header_ratio=header_ratio,
            footer_ratio=footer_ratio,
            watermark_angle=watermark_angle,
            watermark_tolerance=watermark_tolerance,
            debug=debug,
        )
        removed_line_numbers = watermark_filter["removed_line_numbers"]
        watermark_skip_stats["total_lines"] = watermark_filter["total_lines"]
        watermark_skip_stats["watermark_lines"] = watermark_filter["watermark_lines"]
        watermark_skip_stats["watermark_line_ratio"] = watermark_filter["watermark_line_ratio"]
        watermark_skip_stats["watermark_rotation"] = watermark_filter["watermark_rotation"]
        if watermark_skip_stats["total_lines"] > 0 and (
            watermark_skip_stats["watermark_lines"] / float(watermark_skip_stats["total_lines"])
        ) >= 0.98:
            if debug:
                _LOGGER.debug(
                    "Reconstruct watermark filter skipped as too aggressive; forcing render-all: source=%s page=%s ratio=%s",
                    source_text,
                    page_no,
                    watermark_skip_stats["watermark_line_ratio"],
                )
            removed_line_numbers = set()


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
                    _to_rgb_color(drawing.get("fill"), default=None)
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

        fontfile_for_korean = _get_reconstruct_fontfile(fontfile_override=korean_fontfile)
        korean_fontname = _ensure_page_font_resource(
            rendered_pdf=out_page,
            source_path=source_path,
            page_no=page_no,
            korean_fontfile=korean_fontfile,
            debug=debug,
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
            _render_shape_drawing(
                out_page=out_page,
                drawing=drawing,
                page_no=page_no,
                drawing_index=drawing_index,
                debug=debug,
            )

        if debug and fontfile_for_korean:
            _LOGGER.debug(
                "Reconstruct with Korean fallback fontfile=%s for page=%s (fontname=%r)",
                fontfile_for_korean,
                page_no,
                korean_fontname,
            )

        rendered_spans = 0
        skipped_lines = 0
        page_has_korean = False
        for raw_line_no, line in _iter_text_dict_lines(raw_text):
            if raw_line_no in removed_line_numbers:
                skipped_lines += 1
                continue
            spans = line.get("spans", [])
            if not spans:
                continue
            if not page_has_korean:
                page_has_korean = any(
                    _contains_korean((span.get("text") or ""))
                    for span in spans
                    if isinstance(span, dict)
                )
            if isinstance(spans, list):
                rendered_spans += len(spans)
            _draw_raw_spans_on_page(
                out_page=out_page,
                spans=line.get("spans", []),
                source=source_text,
                page_no=page_no,
                korean_fontname=korean_fontname,
                debug=debug,
            )

        if page_has_korean and not korean_fontname:
            _LOGGER.warning(
                "Korean text detected but no registered Korean fallback fontname for page=%s source=%s. "
                "Reconstruction may render as missing glyph boxes (dots).",
                page_no,
                source_text,
            )

        watermark_skip_stats["line_skip_count"] = skipped_lines
        watermark_skip_stats["span_total_count"] = rendered_spans

        if debug:
            _LOGGER.debug(
                "Reconstruct watermark apply summary: source=%s page=%s enabled=%s lines_total=%s lines_skipped=%s spans_rendered=%s",
                source_text,
                page_no,
                watermark_skip_stats["enabled"],
                watermark_skip_stats["total_lines"],
                watermark_skip_stats["line_skip_count"],
                watermark_skip_stats["span_total_count"],
            )

        rendered_pdf.save(output_path, deflate=True, garbage=4)
        rendered_pdf.close()

    return str(Path(output_path))


def _reconstruct_output_path(source_path, page_no):
    prefix = f"{Path(source_path).stem}_page{page_no}_"
    tmp = tempfile.NamedTemporaryFile(
        prefix=prefix, suffix="_reconstruct_from_extract.pdf", delete=False
    )
    tmp.close()
    return tmp.name


def _write_reconstructed_pages_pdf(
    source_path,
    page_numbers,
    output_path,
    remove_watermark=False,
    watermark_angle=_WATERMARK_ROTATION_DEGREE,
    watermark_tolerance=_WATERMARK_ROTATION_TOLERANCE,
    header_ratio=0.08,
    footer_ratio=0.08,
    korean_fontfile=None,
    debug=False,
):
    if not page_numbers:
        raise ValueError("No pages specified for reconstruction.")

    target_pages = _dedupe_pages(page_numbers)

    if not target_pages:
        raise ValueError("No unique pages specified for reconstruction.")

    with pymupdf.open() as rendered_pdf:
        temp_paths = []
        try:
            for page_no in target_pages:
                temp_path = _reconstruct_output_path(source_path, page_no)
                temp_paths.append(temp_path)
                _write_reconstructed_page_pdf(
                    source_path,
                    page_no,
                    temp_path,
                    remove_watermark=remove_watermark,
                    watermark_angle=watermark_angle,
                    watermark_tolerance=watermark_tolerance,
                    header_ratio=header_ratio,
                    footer_ratio=footer_ratio,
                    korean_fontfile=korean_fontfile,
                    debug=debug,
                )

                with pymupdf.open(temp_path) as page_pdf:
                    rendered_pdf.insert_pdf(page_pdf, from_page=0, to_page=0)
        finally:
            for temp_path in temp_paths:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

        rendered_pdf.save(output_path, deflate=True, garbage=4)

    return str(Path(output_path))


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
    debug=False,
    table_debug=None,
    table_mode="auto",
    return_raw_pages=False,
    raw_text_mode="rawdict",
):
    if raw_text_mode not in {"rawdict", "rawjson"}:
        raw_text_mode = "rawdict"

    path = Path(path)

    with pymupdf.open(path) as doc:
        if return_raw_pages:
            pages_with_lines, raw_pages = _extract_pages(
                doc,
                str(path),
                header_ratio,
                footer_ratio,
                max_pages=max_pages,
                pages=pages,
                preserve_newlines=preserve_newlines,
                extract_tables=extract_tables,
                debug=debug,
                table_debug=table_debug if table_debug is not None else debug,
                table_mode=table_mode,
                capture_raw_pages=True,
                raw_text_mode=raw_text_mode,
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
                extract_tables=extract_tables,
                debug=debug,
                table_debug=table_debug if table_debug is not None else debug,
                table_mode=table_mode,
                raw_text_mode=raw_text_mode,
            )
            raw_pages = None

    tables = []
    for _, _, page_tables, _ in pages_with_lines:
        tables.extend(page_tables)

    if extract_tables:
        # keep native PyMuPDF find_tables outputs only (no heuristic table merge/post-processing)
        pass

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


def write_rawdict_pages(raw_pages, output_path):
    with open(output_path, "w", encoding="utf-8", errors="replace") as f:
        for payload in raw_pages:
            if not isinstance(payload, dict):
                continue
            out_payload = {
                "page": payload.get("page"),
                "source": payload.get("source"),
                "rotation": payload.get("rotation"),
                "rect": payload.get("rect"),
            }
            if "text_rawdict" in payload:
                out_payload["text_rawdict"] = payload.get("text_rawdict")
            if "text_rawjson" in payload:
                out_payload["text_rawjson"] = payload.get("text_rawjson")
            f.write(
                json.dumps(
                    out_payload,
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
        help="Output path for mode output or page text JSONL. Defaults to <filename>_pages.jsonl.",
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
        "--tables-markdown",
        nargs="?",
        const="",
        default=None,
        help=(
            "Write detected tables into markdown. Optional path can be provided."
        ),
    )
    parser.add_argument(
        "--reconstruction",
        action="store_true",
        help="Reconstruct selected pages from --pages into a single PDF.",
    )
    parser.add_argument(
        "--watermark-angle",
        type=float,
        default=_WATERMARK_ROTATION_DEGREE,
        help="Target text rotation (degrees) treated as watermark when using --reconstruction.",
    )
    parser.add_argument(
        "--watermark-angle-tolerance",
        type=float,
        default=_WATERMARK_ROTATION_TOLERANCE,
        help="Angle tolerance used for watermark matching with --reconstruction.",
    )
    parser.add_argument(
        "--remove-watermark",
        action="store_true",
        help=(
            "When used with --reconstruction, remove watermark-like text using only rotated text (default 55°)."
        ),
    )
    parser.add_argument(
        "--reconstruction-fontfile",
        default=None,
        metavar="PATH",
        help="Optional font file to use for Korean fallback during --reconstruction.",
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
        "--raw",
        nargs=1,
        default=None,
        metavar="rawdict|rawjson",
        help="Write page.get_text('rawdict') or page.get_text('rawjson') output as JSONL. Must pass `rawdict` or `rawjson`.",
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    if args.reconstruction:
        if args.output is None:
            raise SystemExit("--reconstruction requires --output.")
        if args.reconstruction_fontfile and not Path(args.reconstruction_fontfile).is_file():
            raise SystemExit(f"Invalid --reconstruction-fontfile path: {args.reconstruction_fontfile}")

        try:
            requested_pages = _parse_pages(args.pages)
        except ValueError as exc:
            raise SystemExit(f"Invalid --pages argument: {exc}")

        if not requested_pages:
            raise SystemExit("--reconstruction requires --pages.")

        try:
            created_path = _write_reconstructed_pages_pdf(
                args.file,
                requested_pages,
                args.output,
                remove_watermark=args.remove_watermark,
                watermark_angle=args.watermark_angle,
                watermark_tolerance=args.watermark_angle_tolerance,
                header_ratio=args.header_ratio,
                footer_ratio=args.footer_ratio,
                korean_fontfile=args.reconstruction_fontfile,
                debug=args.debug or args.table_debug,
            )
        except Exception as exc:
            raise SystemExit(f"Failed to create reconstructed page PDF: {exc}")

        print(f"Reconstructed page PDF saved to: {created_path}")
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
    request_raw = args.raw is not None
    raw_mode = args.raw[0].lower() if request_raw and isinstance(args.raw[0], str) else None
    if request_raw and raw_mode not in {"rawdict", "rawjson"}:
        raise SystemExit("--raw requires value `rawdict` or `rawjson`.")

    raw_page_output = "raw_rawjson.jsonl" if raw_mode == "rawjson" else "raw.jsonl"
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
        debug=args.debug,
        table_debug=args.debug or args.table_debug,
        table_mode=args.table_mode,
        return_raw_pages=request_raw,
        raw_text_mode=raw_mode,
    )
    if request_raw:
        records, raw_pages = result
        write_rawdict_pages(raw_pages or [], raw_page_output)
    else:
        records = result
    if args.legacy_page_jsonl:
        write_jsonl_pages(records, output)
        print(f"Extracted {len(records)} pages -> {output}")
        if request_raw:
            print(f"Raw page output ({raw_mode}) -> {raw_page_output}")
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
        summary = (
            f"Extracted {len(records)} pages, {total_lines} text lines, {total_tables} tables -> {output}"
        )
        print(summary)
        if request_raw:
            print(f"Raw page output ({raw_mode}) -> {raw_page_output}")
        if request_tables_markdown:
            print(f"Tables markdown -> {tables_markdown_output}")


if __name__ == "__main__":
    main()
