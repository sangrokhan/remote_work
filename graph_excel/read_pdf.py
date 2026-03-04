import argparse
import logging
import json
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
    preview = [surrounding_snippet(raw, position) for position in positions[:3]]
    _LOGGER.warning(
        "Removed surrogate code units in PDF text. source=%s page=%s line=%s span=%s count=%s snippets=%s",
        context.get("source") if context else "unknown",
        context.get("page") if context else None,
        context.get("line") if context else None,
        context.get("span") if context else None,
        len(positions),
        ", ".join(preview),
    )
    return _SURROGATE_RE.sub("", raw)


def _normalize_line(line):
    line = _sanitize_text(line)
    return re.sub(r"\s+", " ", line).strip()


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


def _collect_repeated_lines(pages, ratio_threshold):
    if not pages:
        return set()

    page_count = len(pages)
    occurrence = Counter()

    for lines in pages:
        seen = set()
        for raw_line in lines:
            normalized = _normalize_line(raw_line)
            if not normalized or not _looks_like_repeated_watermark(normalized):
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            occurrence[key] += 1
            seen.add(key)

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


def _line_matches_patterns(line, patterns):
    return any(pattern.search(line) for pattern in patterns)


def _style_token(span):
    font = (span.get("font") or "").lower()
    flags = int(span.get("flags") or 0)
    is_bold = "bold" in font or bool(flags & 16)
    is_italic = "italic" in font or "oblique" in font or bool(flags & 2)
    is_mono = any(token in font for token in ("mono", "courier", "consola", "consolas"))
    return is_bold, is_italic, is_mono


def _span_to_markdown(raw, span):
    raw = str(raw).replace("\n", " ")
    if not raw:
        return ""

    is_bold, is_italic, is_mono = _style_token(span)
    text = raw.strip("\n\r")
    if not text:
        return ""

    if is_bold and is_italic:
        text = f"***{text}***"
    elif is_bold:
        text = f"**{text}**"
    elif is_italic:
        text = f"*{text}*"
    elif is_mono:
        text = f"`{text}`"
    return text


def _extract_page_lines(page, page_no, source):
    page_data = page.get_text("dict", sort=True)
    lines = []
    line_no = 0

    for block in page_data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_no += 1
            spans = line.get("spans", []) or []
            if not spans:
                continue

            styled_parts = []
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
                text = _span_to_markdown(raw, span)
                if not text:
                    continue
                if styled_parts and not text.startswith(" ") and not styled_parts[-1].endswith(" "):
                    styled_parts.append(" ")
                styled_parts.append(text)
                max_size = max(max_size, float(span.get("size") or 0.0))

            joined = "".join(styled_parts).strip()
            if joined:
                lines.append({"text": joined, "size": max_size})

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


def _extract_pages_to_markdown(doc, source):
    extracted = []
    for page_no, page in enumerate(doc, start=1):
        extracted.append(_extract_page_lines(page, page_no, source))
    return extracted


def read_pdf(path, strip_watermarks=True, patterns=None, ratio_threshold=0.6):
    path = Path(path)

    with pymupdf.open(path) as doc:
        pages_with_lines = _extract_pages_to_markdown(doc, str(path))

    compiled_patterns = _compile_patterns(patterns or [])
    repeated_watermarks = set()

    if strip_watermarks:
        page_text_lines = [[line["text"] for line in lines] for lines in pages_with_lines]
        repeated_watermarks = _collect_repeated_lines(page_text_lines, ratio_threshold)

    records = []
    for page_no, lines in enumerate(pages_with_lines, start=1):
        body_size = _estimate_body_font_size(lines)
        filtered_lines = []

        for line in lines:
            text = line["text"]
            if compiled_patterns and _line_matches_patterns(text, compiled_patterns):
                continue
            if strip_watermarks and _normalize_line(text).casefold() in repeated_watermarks:
                continue
            markdown_line = _line_to_markdown(text, line["size"], body_size)
            if markdown_line:
                filtered_lines.append(markdown_line)

        records.append(
            {
                "page": page_no,
                "text": "\n".join(filtered_lines),
            }
        )

    return records


def write_jsonl(records, output_path):
    with open(output_path, "w", encoding="utf-8", errors="replace") as f:
        for record in records:
            safe_record = {
                "page": record.get("page"),
                "text": _sanitize_text(record.get("text", "")),
            }
            f.write(json.dumps(safe_record, ensure_ascii=False))
            f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read PDF pages with PyMuPDF and output Markdown-style page text as JSONL."
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
        "--watermark-ratio",
        type=float,
        default=0.6,
        help="Ratio threshold to detect repeated watermark lines across pages.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    output = args.output or f"{Path(args.file).stem}_pages.jsonl"
    records = read_pdf(
        args.file,
        strip_watermarks=args.strip_watermarks,
        patterns=args.watermark_patterns,
        ratio_threshold=args.watermark_ratio,
    )
    write_jsonl(records, output)
    print(f"Extracted {len(records)} pages -> {output}")


if __name__ == "__main__":
    main()
