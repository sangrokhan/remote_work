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
    for idx, pattern in enumerate(patterns):
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


def _classify_region(page_rect, bbox, rotation, header_ratio, footer_ratio):
    if page_rect is None:
        return "body"

    x0, y0, x1, y1 = bbox
    if rotation in (90, 270):
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


def _extract_page_lines(page, page_no, source, header_ratio, footer_ratio):
    page_data = page.get_text("dict", sort=True)
    page_rect = page.rect
    rotation = int(page.rotation or 0)
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

            raw_parts = []
            markdown_parts = []
            max_size = 0.0

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

            raw_text = _normalize_line("".join(raw_parts))
            if not raw_text:
                continue

            lines.append(
                {
                    "raw": raw_text,
                    "markdown": _normalize_line("".join(markdown_parts)) or raw_text,
                    "size": max_size,
                    "bbox": line_bbox,
                    "location": location,
                    "rotation": rotation,
                    "page": page_no,
                    "line": line_no,
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


def _build_sections(lines, body_size, repeated_watermarks, compiled_patterns, strip_headers, strip_footers, strip_watermarks):
    sections = {
        "header": [],
        "footer": [],
        "watermark": [],
        "body": [],
    }
    removed = {
        "header": 0,
        "footer": 0,
        "watermark": 0,
        "pattern": 0,
        "kept": 0,
        "total": len(lines),
    }

    for line in lines:
        raw = line["raw"]
        location = line["location"]
        markdown = _line_to_markdown(raw, line["size"], body_size)
        if not markdown:
            continue

        if compiled_patterns:
            _, match_pos = _first_pattern_hit(raw, compiled_patterns)
            if match_pos is not None:
                removed["pattern"] += 1
                removed["watermark"] += 1
                _LOGGER.warning(
                    "Removed by watermark-pattern: source=%s page=%s line=%s rotation=%s location=%s snippet=%s",
                    line.get("source"),
                    line.get("page"),
                    line.get("line"),
                    line.get("rotation"),
                    location,
                    _surrounding_snippet(raw, match_pos),
                )
                sections["watermark"].append(markdown)
                continue

        repeated_key = _normalize_line(raw).casefold()
        if strip_watermarks and repeated_key in repeated_watermarks:
            removed["watermark"] += 1
            removed["kept"] += 0
            _LOGGER.warning(
                "Removed by watermark-detection: source=%s page=%s line=%s rotation=%s location=%s snippet=%s",
                line.get("source"),
                line.get("page"),
                line.get("line"),
                line.get("rotation"),
                location,
                _surrounding_snippet(raw, max(0, len(raw) // 2)),
            )
            sections["watermark"].append(markdown)
            continue

        if location == "header" and strip_headers:
            removed["header"] += 1
            sections["header"].append(markdown)
            continue

        if location == "footer" and strip_footers:
            removed["footer"] += 1
            sections["footer"].append(markdown)
            continue

        sections["body"].append(markdown)
        removed["kept"] += 1

    removed["total"] = len(lines)
    return sections, removed


def _extract_pages(doc, source, header_ratio, footer_ratio):
    extracted = []
    for page_no, page in enumerate(doc, start=1):
        lines = _extract_page_lines(page, page_no, source, header_ratio, footer_ratio)
        for line in lines:
            line["source"] = source
        extracted.append(lines)
    return extracted


def read_pdf(
    path,
    strip_watermarks=True,
    strip_headers=True,
    strip_footers=True,
    patterns=None,
    ratio_threshold=0.6,
    header_ratio=0.08,
    footer_ratio=0.08,
):
    path = Path(path)

    with pymupdf.open(path) as doc:
        pages_with_lines = _extract_pages(doc, str(path), header_ratio, footer_ratio)

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
        sections, removed = _build_sections(
            lines,
            body_size,
            repeated_watermarks,
            compiled_patterns,
            strip_headers,
            strip_footers,
            strip_watermarks,
        )

        records.append(
            {
                "page": page_no,
                "rotation": lines[0]["rotation"] if lines else 0,
                "header": "\n".join(sections["header"]),
                "footer": "\n".join(sections["footer"]),
                "watermark": "\n".join(sections["watermark"]),
                "text": "\n".join(sections["body"]),
                "removed": removed,
            }
        )

    return records


def write_jsonl(records, output_path):
    with open(output_path, "w", encoding="utf-8", errors="replace") as f:
        for record in records:
            safe_record = {
                "page": record.get("page"),
                "rotation": record.get("rotation"),
                "header": _sanitize_text(record.get("header", "")),
                "footer": _sanitize_text(record.get("footer", "")),
                "watermark": _sanitize_text(record.get("watermark", "")),
                "text": _sanitize_text(record.get("text", "")),
                "removed": record.get("removed", {}),
            }
            f.write(json.dumps(safe_record, ensure_ascii=False))
            f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read PDF pages with PyMuPDF, split/remove headers/footers/watermarks.")
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
    )
    write_jsonl(records, output)
    print(f"Extracted {len(records)} pages -> {output}")


if __name__ == "__main__":
    main()
