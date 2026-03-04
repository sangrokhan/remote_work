import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import fitz


def _normalize_line(line):
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


def read_pdf(path, strip_watermarks=True, patterns=None, ratio_threshold=0.6):
    path = Path(path)
    pages_raw = []

    with fitz.open(path) as doc:
        for page in doc:
            text = page.get_text() or ""
            pages_raw.append([_normalize_line(line) for line in text.splitlines() if line.strip()])

    compiled_patterns = _compile_patterns(patterns or [])
    repeated_watermarks = set()

    if strip_watermarks:
        repeated_watermarks = _collect_repeated_lines(pages_raw, ratio_threshold)

    records = []
    for page_no, lines in enumerate(pages_raw, start=1):
        filtered_lines = []
        for line in lines:
            if compiled_patterns and _line_matches_patterns(line, compiled_patterns):
                continue
            if strip_watermarks and line.casefold() in repeated_watermarks:
                continue
            filtered_lines.append(line)
        records.append(
            {
                "page": page_no,
                "text": "\n".join(filtered_lines),
            }
        )

    return records


def write_jsonl(records, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read PDF pages with PyMuPDF and output page text as JSONL."
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
