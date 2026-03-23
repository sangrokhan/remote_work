from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pdfplumber

from .debug import _collect_page_edge_debug, _collect_rotated_text_debug, _collect_table_drawing_debug
from .images import _extract_embedded_images
from .shared import TableRows, _merge_numeric_positions
from .tables import (
    _append_output_table,
    _continuation_regions_should_merge,
    _extract_tables,
    _gap_text_boxes_after_bbox,
    _gap_text_boxes_before_bbox,
    _maybe_merge_missing_first_column_chunk,
    _is_continuation_chunk,
    _should_try_table_continuation_merge,
    _split_repeated_header,
    _vertical_axes_for_bbox,
)
from .text import _detect_body_bounds, _extract_body_text, _extract_drawing_image_bboxes


def _document_text_profile(debug_pages: Sequence[dict]) -> dict:
    # Document-level text profile lets later structure rules pick thresholds from observed font sizes.
    font_size_counter: Counter[float] = Counter()
    fontname_counter: Counter[str] = Counter()
    pages_using_size: dict[float, set[int]] = {}
    for page in debug_pages:
        page_no = int(page.get("page", 0))
        page_profile = page.get("text_debug", {}).get("profile", {})
        for size_text, count in page_profile.get("font_size_histogram", {}).items():
            size = round(float(size_text), 2)
            font_size_counter[size] += int(count)
            pages_using_size.setdefault(size, set()).add(page_no)
        for fontname, count in page_profile.get("fontname_histogram", {}).items():
            if str(fontname):
                fontname_counter[str(fontname)] += int(count)

    dominant_font_size = max(font_size_counter, key=font_size_counter.get) if font_size_counter else 0.0
    dominant_fontname = max(fontname_counter, key=fontname_counter.get) if fontname_counter else ""
    return {
        "font_size_histogram": {
            f"{size:.2f}": count for size, count in sorted(font_size_counter.items())
        },
        "fontname_histogram": dict(sorted(fontname_counter.items())),
        "font_size_candidates": sorted(font_size_counter),
        "dominant_font_size": dominant_font_size,
        "dominant_fontname": dominant_fontname,
        "pages_using_size": {
            f"{size:.2f}": sorted(page_numbers) for size, page_numbers in sorted(pages_using_size.items())
        },
    }


def _body_excluded_bboxes(
    pending_table: Optional[TableRows],
    tables: Sequence[Tuple[TableRows, Tuple[float, float, float, float]]],
    image_regions: Sequence[Tuple[float, float, float, float]],
    body_top: float,
) -> List[Tuple[float, float, float, float]]:
    excluded = [bbox for _rows, bbox in tables]
    excluded.extend(list(image_regions))
    if not pending_table or not tables:
        return excluded

    first_rows, first_bbox = tables[0]
    if len(pending_table[0]) == 3 and first_rows and all(len(row) == 2 for row in first_rows):
        x0, _top, x1, bottom = first_bbox
        excluded[0] = (x0, body_top, x1, bottom)
    return excluded


def extract_pdf_to_outputs(
    pdf_path: Path,
    out_md_dir: Path,
    out_image_dir: Path,
    stem: str,
    header_margin: float = 90,
    footer_margin: float = 40,
    pages: Optional[Sequence[int]] = None,
    force_table: bool = False,
    debug: bool = False,
    debug_watermark: bool = False,
) -> dict:
    out_md_dir.mkdir(parents=True, exist_ok=True)

    output_text: List[str] = []
    output_tables: List[str] = []
    table_debug_pages: List[dict] = []
    edge_debug_pages: List[dict] = []
    rotated_debug: List[dict] = []
    selected_pages = set(int(page_no) for page_no in (pages or []))
    drawing_regions_by_page: dict[int, list[Tuple[float, float, float, float]]] = {}

    pending_table: Optional[TableRows] = None
    pending_page: Optional[int] = None
    pending_last_page: Optional[int] = None
    pending_bbox: Optional[Tuple[float, float, float, float]] = None
    pending_axes: List[float] = []
    pending_gap_text_boxes: List[Tuple[float, float, float, float]] = []

    def _flush_pending() -> None:
        # Tables are emitted only after we know the next page will not extend them.
        nonlocal pending_table, pending_page, pending_last_page, pending_bbox, pending_axes, pending_gap_text_boxes
        if pending_table is not None and pending_page is not None:
            _append_output_table(output_tables, pending_page, len(output_tables) + 1, pending_table)
        pending_table = None
        pending_page = None
        pending_last_page = None
        pending_bbox = None
        pending_axes = []
        pending_gap_text_boxes = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            if selected_pages and page_idx not in selected_pages:
                _flush_pending()
                continue

            if debug:
                table_debug_pages.append(
                    _collect_table_drawing_debug(page, page_no=page_idx, header_margin=header_margin, footer_margin=footer_margin)
                )
                edge_debug_pages.append(
                    _collect_page_edge_debug(page, page_no=page_idx, header_margin=header_margin, footer_margin=footer_margin)
                )
            if debug_watermark:
                rotated_debug.extend(_collect_rotated_text_debug(page, page_no=page_idx))

            tables = _extract_tables(page, force_table=force_table)
            image_regions = _extract_drawing_image_bboxes(
                page=page,
                header_margin=header_margin,
                footer_margin=footer_margin,
                excluded_bboxes=[bbox for _rows, bbox in tables],
            )
            drawing_regions_by_page[page_idx] = image_regions
            full_page_text = _extract_body_text(page, header_margin=header_margin, footer_margin=footer_margin)
            # Body text for the final page output excludes table areas so prose does not duplicate table content.
            page_text = _extract_body_text(
                page,
                header_margin=header_margin,
                footer_margin=footer_margin,
                excluded_bboxes=_body_excluded_bboxes(
                    pending_table=pending_table,
                    tables=tables,
                    image_regions=image_regions,
                    body_top=footer_margin,
                ),
            )
            if page_text.strip():
                output_text.append(f"### Page {page_idx}\n{page_text}")

            if not tables:
                continue

            body_top, body_bottom = _detect_body_bounds(page, header_margin=header_margin, footer_margin=footer_margin)
            table_bboxes = [table_bbox for _table_rows, table_bbox in tables]
            for table_rows, bbox in tables:
                cross_page_continuation = _should_try_table_continuation_merge(
                    pending_page=pending_last_page,
                    current_page=page_idx,
                )

                # Some continuation fragments lose the first column and need a specialized merge path.
                merged_missing_first = None
                if cross_page_continuation:
                    merged_missing_first = _maybe_merge_missing_first_column_chunk(
                        pending_table,
                        table_rows,
                        full_page_text,
                    )
                if merged_missing_first is not None:
                    pending_table = merged_missing_first
                    pending_last_page = page_idx
                    pending_bbox = bbox
                    pending_axes = _vertical_axes_for_bbox(page, bbox)
                    pending_gap_text_boxes = _gap_text_boxes_after_bbox(page, bbox, table_bboxes, header_margin=header_margin, footer_margin=footer_margin)
                    continue

                continuation_rows = table_rows
                if cross_page_continuation:
                    # Repeated headers should not be duplicated when the next page is clearly part of the same table.
                    continuation_rows = _split_repeated_header(pending_table or [], table_rows)
                    if pending_table is not None and _is_continuation_chunk(pending_table, continuation_rows):
                        pending_table.extend(continuation_rows)
                        pending_last_page = page_idx
                        if pending_bbox is not None:
                            pending_bbox = (
                                min(pending_bbox[0], bbox[0]),
                                min(pending_bbox[1], bbox[1]),
                                max(pending_bbox[2], bbox[2]),
                                max(pending_bbox[3], bbox[3]),
                            )
                        else:
                            pending_bbox = bbox
                        pending_axes = _merge_numeric_positions([*pending_axes, *_vertical_axes_for_bbox(page, bbox)], tolerance=1.0)
                        pending_gap_text_boxes = _gap_text_boxes_after_bbox(page, bbox, table_bboxes, header_margin=header_margin, footer_margin=footer_margin)
                        continue

                current_axes = _vertical_axes_for_bbox(page, bbox)
                current_gap_text_boxes = _gap_text_boxes_before_bbox(page, bbox, table_bboxes, header_margin=header_margin, footer_margin=footer_margin)
                if (
                    pending_table is not None
                    and pending_bbox is not None
                    and pending_last_page is not None
                    and cross_page_continuation
                    and _continuation_regions_should_merge(
                        prev_bbox=pending_bbox,
                        curr_bbox=bbox,
                        prev_axes=pending_axes,
                        curr_axes=current_axes,
                        body_top=body_top,
                        body_bottom=body_bottom,
                        gap_text_boxes=[*pending_gap_text_boxes, *current_gap_text_boxes],
                    )
                ):
                    pending_table.extend(continuation_rows)
                    pending_last_page = page_idx
                    pending_bbox = (
                        min(pending_bbox[0], bbox[0]),
                        min(pending_bbox[1], bbox[1]),
                        max(pending_bbox[2], bbox[2]),
                        max(pending_bbox[3], bbox[3]),
                    )
                    pending_axes = _merge_numeric_positions([*pending_axes, *current_axes], tolerance=1.0)
                    pending_gap_text_boxes = _gap_text_boxes_after_bbox(page, bbox, table_bboxes, header_margin=header_margin, footer_margin=footer_margin)
                    continue

                # Once continuation checks fail, the previous pending table is finalized and a new table starts.
                _flush_pending()
                pending_table = table_rows
                pending_page = page_idx
                pending_last_page = page_idx
                pending_bbox = bbox
                pending_axes = current_axes
                pending_gap_text_boxes = _gap_text_boxes_after_bbox(page, bbox, table_bboxes, header_margin=header_margin, footer_margin=footer_margin)

        _flush_pending()

    markdown = "\n\n".join(output_text)
    table_markdown = "\n\n".join(output_tables)

    text_file = out_md_dir / f"{stem}.txt"
    md_file = out_md_dir / f"{stem}.md"
    table_md_file = out_md_dir / f"{stem}_table.md"
    text_file.write_text(markdown, encoding="utf-8")
    md_file.write_text(markdown, encoding="utf-8")
    table_md_file.write_text(table_markdown, encoding="utf-8")

    # Image extraction happens after text/table rendering so image export stays independent from markdown generation.
    image_files = _extract_embedded_images(
        pdf_path=pdf_path,
        out_image_dir=out_image_dir,
        stem=stem,
        pages=pages,
        drawing_regions_by_page=drawing_regions_by_page,
    )

    summary = {
        "pdf": str(pdf_path),
        "text_file": str(text_file),
        "md_file": str(md_file),
        "table_md_file": str(table_md_file),
        "images": [str(p) for p in image_files],
        "table_count": len(output_tables),
    }
    summary_file = out_md_dir / f"{stem}_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    debug_file: Optional[Path] = None
    debug_edges_file: Optional[Path] = None
    if debug:
        debug_file = out_md_dir / f"{stem}_debug.json"
        debug_file.write_text(
            json.dumps(
                {
                    "pdf": str(pdf_path),
                    "document_text_profile": _document_text_profile(table_debug_pages),
                    "pages": table_debug_pages,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        debug_edges_file = out_md_dir / f"{stem}_edges_debug.json"
        debug_edges_file.write_text(json.dumps({"pdf": str(pdf_path), "pages": edge_debug_pages}, ensure_ascii=False, indent=2), encoding="utf-8")

    debug_watermark_file: Optional[Path] = None
    if debug_watermark:
        debug_watermark_file = out_md_dir / f"{stem}_watermark_debug.json"
        debug_watermark_file.write_text(json.dumps(rotated_debug, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "markdown": markdown,
        "table_markdown": table_markdown,
        "text_file": text_file,
        "md_file": md_file,
        "table_md_file": table_md_file,
        "debug_file": debug_file,
        "debug_edges_file": debug_edges_file,
        "debug_watermark_file": debug_watermark_file,
        "image_files": image_files,
        "summary": summary,
    }
