import argparse
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
_LOGGER = logging.getLogger("read_pdf")

_SUPPORTED_ROTATIONS = {0, 90, 180, 270}


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
        return raw

    if context is None:
        return _SURROGATE_RE.sub("", raw)

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
    return _SURROGATE_RE.sub("", raw)


def _normalize_line(line):
    return re.sub(r"\s+", " ", _sanitize_text(line)).strip()


def _looks_like_repeated_watermark(line):
    if not line:
        return False
    if len(line) > 120:
        return False

    alpha_chars = [ch for ch in line if ch.isalpha()]
    if len(alpha_chars) < 4:
        return False

    upper_ratio = sum(ch.isupper() for ch in alpha_chars) / len(alpha_chars)
    return upper_ratio >= 0.7


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


def _extract_page_lines(page, page_no, source, header_ratio, footer_ratio):
    page_data = page.get_text("dict", sort=True)
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

            raw_text = _normalize_line("".join(raw_parts))
            if not raw_text:
                continue

            baseline_axis = axis
            baseline_value = _line_baseline(spans, baseline_axis)
            page_axis_size = float(page_rect.width) if baseline_axis == "x" else float(page_rect.height)
            baseline_ratio = baseline_value / page_axis_size if page_axis_size > 0 else None
            line_rotation = (int(rotation) + round(_line_tilt_angle(line, spans))) % 360

            dominant_font = font_counts.most_common(1)[0][0] if font_counts else None

            lines.append(
                {
                    "raw": raw_text,
                    "markdown": _normalize_line("".join(markdown_parts)) or raw_text,
                    "size": max_size,
                    "bbox": line_bbox,
                    "location": location,
                    "rotation": line_rotation,
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
                }
            )

    return lines


def _estimate_body_font_size(lines):
    sizes = [round(line.get("size", 0.0), 1) for line in lines if line.get("size")]
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


def _line_to_payload(line, markdown_line, target_region, removed_reason, removed):
    position = line.get("position", {})
    baseline = position.get("baseline", {})
    return {
        "region": target_region,
        "removed": removed,
        "removed_reason": removed_reason,
        "line_no": line["line"],
        "text": line["raw"],
        "markdown": markdown_line,
        "rotation": line["rotation"],
        "rotation_axis": line["rotation_axis"],
        "rotation_ratio": baseline.get("ratio"),
        "x": position.get("x"),
        "y": position.get("y"),
        "x_ratio": position.get("x_ratio"),
        "y_ratio": position.get("y_ratio"),
        "font_family": line.get("font_family"),
        "size": line.get("size"),
        "span_count": line.get("span_count", 0),
        "color": line.get("color"),
        "bbox": line.get("bbox"),
        "position": {
            "baseline": baseline,
        },
        "page": line["page"],
        "source": line.get("source"),
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


def _extract_pages(doc, source, header_ratio, footer_ratio, max_pages=None):
    extracted = []
    for page_no, page in enumerate(doc, start=1):
        if max_pages is not None and page_no > max_pages:
            break
        lines = _extract_page_lines(page, page_no, source, header_ratio, footer_ratio)
        for line in lines:
            line["source"] = source
        extracted.append(lines)
    return extracted


def _safe_text_list(values):
    return [_sanitize_text(value) for value in values if value is not None]


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
):
    path = Path(path)

    with pymupdf.open(path) as doc:
        pages_with_lines = _extract_pages(
            doc,
            str(path),
            header_ratio,
            footer_ratio,
            max_pages=max_pages,
        )

    compiled_patterns = _compile_patterns(patterns or [])
    repeated_watermarks = set()
    if strip_watermarks:
        repeated_watermarks = _collect_repeated_lines(
            pages_with_lines,
            ratio_threshold=ratio_threshold,
        )

    records = []
    for page_no, lines in enumerate(pages_with_lines, start=1):
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
                            item["markdown"] for item in sections["header"]["items"] if item["removed"]
                        )
                    )
                ),
                "footer": _sanitize_text(
                    "\n".join(
                        _safe_text_list(
                            item["markdown"] for item in sections["footer"]["items"] if item["removed"]
                        )
                    )
                ),
                "watermark": _sanitize_text(
                    "\n".join(
                        _safe_text_list(
                            item["markdown"] for item in sections["watermark"]["items"] if item["removed"]
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
            }
        )

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
                                "x": item.get("x"),
                                "y": item.get("y"),
                                "rotation": item.get("rotation"),
                                "text": _sanitize_text(item.get("text", "")),
                            },
                        )
                    )
            ordered_items.sort(key=lambda entry: entry[0] if entry[0] is not None else 0)
            for _, payload in ordered_items:
                f.write(json.dumps(payload, ensure_ascii=False))
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
            }
            f.write(json.dumps(safe_record, ensure_ascii=False))
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
        "--legacy-page-jsonl",
        action="store_true",
        help="Keep old page-based JSONL output instead of line-by-line JSONL output.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args = parse_args()
    output = args.output or f"{Path(args.file).stem}_pages.jsonl"
    records = read_pdf(
        args.file,
        strip_watermarks=args.strip_watermarks,
        strip_headers=args.strip_headers,
        strip_footers=args.strip_footers,
        patterns=args.watermark_patterns,
        ratio_threshold=args.watermark_ratio,
        header_ratio=args.header_ratio,
        footer_ratio=args.footer_ratio,
        max_pages=args.max_pages,
    )
    if args.legacy_page_jsonl:
        write_jsonl_pages(records, output)
        print(f"Extracted {len(records)} pages -> {output}")
    else:
        write_jsonl(records, output)
        total_lines = sum(
            len(record.get("regions", {}).get(region, []))
            for record in records
            for region in ("header", "body", "footer", "watermark")
        )
        print(f"Extracted {len(records)} pages, {total_lines} text lines -> {output}")


if __name__ == "__main__":
    main()
