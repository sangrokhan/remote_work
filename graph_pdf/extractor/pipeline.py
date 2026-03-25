from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pdfplumber

from .debug import _collect_page_edge_debug, _collect_rotated_text_debug, _collect_table_drawing_debug
from .images import _collect_embedded_image_refs, _extract_embedded_images
from .raw import materialize_raw_dump
from .shared import TableRows, _merge_numeric_positions
from .tables import (
    _append_output_table,
    _continuation_regions_should_merge,
    _extract_tables,
    _gap_text_boxes_after_bbox,
    _gap_text_boxes_before_bbox,
    _looks_like_single_column_note,
    _maybe_merge_missing_first_column_chunk,
    _is_continuation_chunk,
    _should_try_table_continuation_merge,
    _single_column_note_body_text,
    _split_repeated_header,
    _vertical_axes_for_bbox,
)
from .text import _detect_body_bounds, _extract_body_text, _extract_drawing_image_bboxes


def _heading_level_from_rule(rule: dict) -> int | None:
    assign = rule.get("assign") or {}
    tag = str(assign.get("tag") or "").strip().lower()
    if len(tag) == 2 and tag.startswith("h") and tag[1].isdigit():
        level = int(tag[1])
        if 1 <= level <= 6:
            return level

    markdown_prefix = str(assign.get("markdown_prefix") or "")
    sharp_count = len(markdown_prefix.strip())
    return sharp_count if 1 <= sharp_count <= 6 else None


def _load_heading_levels(add_heading: Path | None) -> dict[float, int] | None:
    if add_heading is None:
        return None

    payload = json.loads(Path(add_heading).read_text(encoding="utf-8"))
    heading_levels: dict[float, int] = {}
    for rule in payload.get("heading_rules", []):
        match = rule.get("match") or {}
        if "font_size" not in match:
            continue
        level = _heading_level_from_rule(rule)
        if level is None:
            continue
        font_size = round(float(match["font_size"]), 2)
        heading_levels[font_size] = level
    return heading_levels


def _format_page_comment(page_no: int) -> str:
    return f"[//]: # (Page {page_no})"


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


def _content_ref_text(content_type: str, page_no: int, index: int, continued: bool = False) -> str:
    label = f"[{content_type} reference: Page {page_no} {content_type.lower()} {index}]"
    if continued:
        label += " (continued)"
    return label


def _should_continue_content_region(
    prev_bbox: Tuple[float, float, float, float],
    curr_bbox: Tuple[float, float, float, float],
    _prev_body_top: float,
    prev_body_bottom: float,
    curr_body_top: float,
    min_x_overlap_ratio: float = 0.35,
    edge_tolerance: float = 24.0,
) -> bool:
    prev_x0, _prev_top, prev_x1, prev_bottom = prev_bbox
    curr_x0, curr_top, curr_x1, _curr_bottom = curr_bbox
    if abs(prev_bottom - prev_body_bottom) > edge_tolerance:
        return False
    if abs(curr_top - curr_body_top) > edge_tolerance:
        return False

    overlap = min(prev_x1, curr_x1) - max(prev_x0, curr_x0)
    if overlap <= 0:
        return False

    prev_width = max(0.0, prev_x1 - prev_x0)
    curr_width = max(0.0, curr_x1 - curr_x0)
    if prev_width <= 0.0 or curr_width <= 0.0:
        return False
    return overlap / min(prev_width, curr_width) >= min_x_overlap_ratio


def _is_edge_candidate_for_continuation(
    bbox: Tuple[float, float, float, float],
    body_top: float,
    body_bottom: float,
    edge_tolerance: float = 24.0,
) -> bool:
    _x0, _top, _x1, bottom = bbox
    return abs(bottom - body_bottom) <= edge_tolerance


def _table_reference_text(page_no: int, table_no: int) -> str:
    return f"[Table reference: Page {page_no} table {table_no}]"


def extract_pdf_to_outputs(
    pdf_path: Path | None,
    out_md_dir: Path,
    out_image_dir: Path,
    stem: str,
    header_margin: float = 90,
    footer_margin: float = 40,
    pages: Optional[Sequence[int]] = None,
    force_table: bool = False,
    debug: bool = False,
    debug_watermark: bool = False,
    add_heading: Path | None = None,
    page_write: bool = False,
    from_raw: Path | None = None,
) -> dict:
    if from_raw is not None:
        with materialize_raw_dump(from_raw) as (materialized_pdf_path, _raw_payload):
            return extract_pdf_to_outputs(
                pdf_path=materialized_pdf_path,
                out_md_dir=out_md_dir,
                out_image_dir=out_image_dir,
                stem=stem,
                header_margin=header_margin,
                footer_margin=footer_margin,
                pages=pages,
                force_table=force_table,
                debug=debug,
                debug_watermark=debug_watermark,
                add_heading=add_heading,
                page_write=page_write,
                from_raw=None,
            )
    if pdf_path is None:
        raise ValueError("pdf_path is required when from_raw is not provided")
    out_md_dir.mkdir(parents=True, exist_ok=True)

    output_text: List[str] = []
    output_tables: List[str] = []
    table_debug_pages: List[dict] = []
    edge_debug_pages: List[dict] = []
    rotated_debug: List[dict] = []
    selected_pages = set(int(page_no) for page_no in (pages or []))
    drawing_regions_by_page: dict[int, list[Tuple[float, float, float, float]]] = {}
    heading_levels = _load_heading_levels(add_heading)

    pending_table: Optional[TableRows] = None
    pending_page: Optional[int] = None
    pending_last_page: Optional[int] = None
    pending_bbox: Optional[Tuple[float, float, float, float]] = None
    pending_table_no: Optional[int] = None
    pending_axes: List[float] = []
    pending_gap_text_boxes: List[Tuple[float, float, float, float]] = []
    pending_page_height: Optional[float] = None
    next_table_no = 1
    pending_image_ref_bbox: Optional[Tuple[float, float, float, float]] = None
    pending_image_body_top: Optional[float] = None
    pending_image_body_bottom: Optional[float] = None
    pending_drawing_ref_bbox: Optional[Tuple[float, float, float, float]] = None
    pending_drawing_body_top: Optional[float] = None
    pending_drawing_body_bottom: Optional[float] = None
    emitted_table_references: set[tuple[int, int]] = set()

    embedded_image_refs_by_page = _collect_embedded_image_refs(
        pdf_path=pdf_path,
        pages=pages,
        header_margin=header_margin,
        footer_margin=footer_margin,
    )

    def _flush_pending() -> None:
        # Tables are emitted only after we know the next page will not extend them.
        nonlocal pending_table, pending_page, pending_last_page, pending_bbox, pending_table_no, pending_axes, pending_gap_text_boxes, pending_page_height
        if pending_table is not None and pending_page is not None and pending_table_no is not None:
            _append_output_table(output_tables, pending_page, pending_table_no, pending_table)
        pending_table = None
        pending_page = None
        pending_last_page = None
        pending_bbox = None
        pending_table_no = None
        pending_axes = []
        pending_gap_text_boxes = []
        pending_page_height = None

    def _append_table_reference(
        refs: List[dict],
        page_no: int,
        table_no: int,
        bbox: Tuple[float, float, float, float],
    ) -> None:
        key = (page_no, table_no)
        if key in emitted_table_references:
            return
        emitted_table_references.add(key)
        refs.append({"text": _table_reference_text(page_no, table_no), "bbox": bbox})

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            if selected_pages and page_idx not in selected_pages:
                _flush_pending()
                pending_image_ref_bbox = None
                pending_image_body_top = None
                pending_image_body_bottom = None
                pending_drawing_ref_bbox = None
                pending_drawing_body_top = None
                pending_drawing_body_bottom = None
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

            detected_tables = _extract_tables(page, force_table=force_table)
            tables: List[Tuple[TableRows, Tuple[float, float, float, float]]] = []
            note_references: List[dict] = []
            detected_table_payloads: List[dict] = []
            for table_rows, bbox in detected_tables:
                row_count = len(table_rows)
                col_count = max((len(row) for row in table_rows), default=0)
                is_note = _looks_like_single_column_note(
                    rows=table_rows,
                    page=page,
                    bbox=bbox,
                )
                detected_table_payloads.append(
                    {
                        "kind": "note" if is_note else "table",
                        "bbox": [round(float(value), 2) for value in bbox],
                        "row_count": int(row_count),
                        "col_count": int(col_count),
                    }
                )
                if is_note:
                    note_text = _single_column_note_body_text(table_rows)
                    if note_text:
                        note_references.append({"text": note_text, "bbox": bbox})
                    continue
                tables.append((table_rows, bbox))
            if debug and table_debug_pages:
                table_debug_pages[-1]["detected_tables"] = detected_table_payloads

            body_top, body_bottom = _detect_body_bounds(page, header_margin=header_margin, footer_margin=footer_margin)
            image_regions = _extract_drawing_image_bboxes(
                page=page,
                header_margin=header_margin,
                footer_margin=footer_margin,
                excluded_bboxes=[bbox for _rows, bbox in detected_tables],
            )
            drawing_regions_by_page[page_idx] = image_regions
            embedded_image_refs = embedded_image_refs_by_page.get(page_idx, [])
            embedded_image_regions = [
                tuple(entry.get("bbox", ()))
                for entry in embedded_image_refs
                if isinstance(entry, dict) and len(entry.get("bbox", ())) == 4
            ]
            full_page_text = _extract_body_text(page, header_margin=header_margin, footer_margin=footer_margin)
            page_pending_table = pending_table
            page_excluded_bboxes = _body_excluded_bboxes(
                pending_table=page_pending_table,
                tables=tables,
                image_regions=[
                    *image_regions,
                    *embedded_image_regions,
                ],
                body_top=footer_margin,
            )
            page_excluded_bboxes.extend([bbox for _rows, bbox in detected_tables])
            page_table_references: List[dict] = []
            page_content_references: List[dict] = []

            if not tables:
                for image_idx, entry in enumerate(embedded_image_refs, start=1):
                    bbox_obj = entry.get("bbox") if isinstance(entry, dict) else None
                    if not bbox_obj or len(bbox_obj) != 4:
                        continue
                    bbox = tuple(bbox_obj)
                    is_cont = False
                    if (
                        pending_image_ref_bbox is not None
                        and pending_image_body_top is not None
                        and pending_image_body_bottom is not None
                    ):
                        is_cont = _should_continue_content_region(
                            prev_bbox=pending_image_ref_bbox,
                            curr_bbox=bbox,
                            prev_body_top=pending_image_body_top,
                            prev_body_bottom=pending_image_body_bottom,
                            curr_body_top=body_top,
                        )
                    page_content_references.append(
                        {
                            "text": _content_ref_text("Image", page_idx, image_idx, continued=is_cont),
                            "bbox": bbox,
                        }
                    )
                    if _is_edge_candidate_for_continuation(bbox=bbox, body_top=body_top, body_bottom=body_bottom):
                        pending_image_ref_bbox = bbox
                        pending_image_body_top = body_top
                        pending_image_body_bottom = body_bottom
                    else:
                        pending_image_ref_bbox = None
                        pending_image_body_top = None
                        pending_image_body_bottom = None

                page_text = _extract_body_text(
                    page,
                    header_margin=header_margin,
                    footer_margin=footer_margin,
                    excluded_bboxes=page_excluded_bboxes,
                    reference_lines=page_table_references
                    + note_references
                    + [
                        {
                            "text": entry["text"],
                            "bbox": entry["bbox"],
                        }
                        for entry in page_content_references
                    ],
                    heading_levels=heading_levels,
                )
                if page_text.strip():
                    if page_write:
                        output_text.append(f"{_format_page_comment(page_idx)}\n{page_text}")
                    else:
                        output_text.append(page_text)
                continue

            table_bboxes = [table_bbox for _table_rows, table_bbox in tables]
            for table_rows, bbox in tables:
                cross_page_continuation = _should_try_table_continuation_merge(
                    pending_page=pending_last_page,
                    current_page=page_idx,
                )
                current_gap_text_boxes = _gap_text_boxes_before_bbox(
                    page,
                    bbox,
                    table_bboxes,
                    header_margin=header_margin,
                    footer_margin=footer_margin,
                )
                current_axes = _vertical_axes_for_bbox(page, bbox)

                # Some continuation fragments lose the first column and need a specialized merge path.
                merged_missing_first = None
                if (
                    cross_page_continuation
                    and not pending_gap_text_boxes
                    and not current_gap_text_boxes
                ):
                    merged_missing_first = _maybe_merge_missing_first_column_chunk(
                        pending_table,
                        table_rows,
                        full_page_text,
                    )
                if merged_missing_first is not None:
                    if pending_page is not None and pending_table_no is not None:
                        _append_table_reference(
                            refs=page_table_references,
                            page_no=pending_page,
                            table_no=pending_table_no,
                            bbox=bbox,
                        )
                    pending_table = merged_missing_first
                    pending_last_page = page_idx
                    pending_page_height = float(page.height)
                    pending_bbox = bbox
                    pending_axes = _vertical_axes_for_bbox(page, bbox)
                    pending_gap_text_boxes = _gap_text_boxes_after_bbox(page, bbox, table_bboxes, header_margin=header_margin, footer_margin=footer_margin)
                    continue

                continuation_rows = table_rows
                if cross_page_continuation:
                    # Repeated headers should not be duplicated when the next page is clearly part of the same table.
                    if pending_gap_text_boxes or current_gap_text_boxes:
                        continuation_rows = table_rows
                    else:
                        continuation_rows = _split_repeated_header(pending_table or [], table_rows)
                    if pending_table is not None and _is_continuation_chunk(pending_table, continuation_rows):
                        if pending_page is not None and pending_table_no is not None:
                            _append_table_reference(
                                refs=page_table_references,
                                page_no=pending_page,
                                table_no=pending_table_no,
                                bbox=bbox,
                            )
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
                            pending_page_height = float(page.height)
                            pending_axes = _merge_numeric_positions([*pending_axes, *_vertical_axes_for_bbox(page, bbox)], tolerance=1.0)
                            pending_gap_text_boxes = _gap_text_boxes_after_bbox(page, bbox, table_bboxes, header_margin=header_margin, footer_margin=footer_margin)
                            continue

                # current_axes is intentionally reused by both continuation and non-continuation paths below.
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
                        prev_page_height=pending_page_height,
                    )
                ):
                    if pending_page is not None and pending_table_no is not None:
                        _append_table_reference(
                            refs=page_table_references,
                            page_no=pending_page,
                            table_no=pending_table_no,
                            bbox=bbox,
                        )
                    pending_table.extend(continuation_rows)
                    pending_last_page = page_idx
                    pending_bbox = (
                        min(pending_bbox[0], bbox[0]),
                        min(pending_bbox[1], bbox[1]),
                        max(pending_bbox[2], bbox[2]),
                        max(pending_bbox[3], bbox[3]),
                    )
                    pending_page_height = float(page.height)
                    pending_axes = _merge_numeric_positions([*pending_axes, *current_axes], tolerance=1.0)
                    pending_gap_text_boxes = _gap_text_boxes_after_bbox(page, bbox, table_bboxes, header_margin=header_margin, footer_margin=footer_margin)
                    continue

                # Once continuation checks fail, the previous pending table is finalized and a new table starts.
                _flush_pending()
                current_table_no = next_table_no
                next_table_no += 1
                _append_table_reference(
                    refs=page_table_references,
                    page_no=page_idx,
                    table_no=current_table_no,
                    bbox=bbox,
                )
                pending_table = table_rows
                pending_page = page_idx
                pending_last_page = page_idx
                pending_page_height = float(page.height)
                pending_bbox = bbox
                pending_table_no = current_table_no
                pending_axes = current_axes
                pending_gap_text_boxes = _gap_text_boxes_after_bbox(page, bbox, table_bboxes, header_margin=header_margin, footer_margin=footer_margin)

            for image_idx, entry in enumerate(embedded_image_refs, start=1):
                bbox_obj = entry.get("bbox") if isinstance(entry, dict) else None
                if not bbox_obj or len(bbox_obj) != 4:
                    continue
                bbox = tuple(bbox_obj)
                is_cont = False
                if (
                    pending_image_ref_bbox is not None
                    and pending_image_body_top is not None
                    and pending_image_body_bottom is not None
                ):
                    is_cont = _should_continue_content_region(
                        prev_bbox=pending_image_ref_bbox,
                        curr_bbox=bbox,
                        prev_body_top=pending_image_body_top,
                        prev_body_bottom=pending_image_body_bottom,
                        curr_body_top=body_top,
                    )

                page_content_references.append(
                    {
                        "text": _content_ref_text("Image", page_idx, image_idx, continued=is_cont),
                        "bbox": bbox,
                    }
                )

                if _is_edge_candidate_for_continuation(bbox=bbox, body_top=body_top, body_bottom=body_bottom):
                    pending_image_ref_bbox = bbox
                    pending_image_body_top = body_top
                    pending_image_body_bottom = body_bottom
                else:
                    pending_image_ref_bbox = None
                    pending_image_body_top = None
                    pending_image_body_bottom = None

            page_text = _extract_body_text(
                page,
                header_margin=header_margin,
                footer_margin=footer_margin,
                excluded_bboxes=page_excluded_bboxes,
                reference_lines=[
                    {
                        "text": entry["text"],
                        "bbox": entry["bbox"],
                    }
                    for entry in page_table_references
                ]
                + ([
                    {
                        "text": entry["text"],
                        "bbox": entry["bbox"],
                    }
                    for entry in page_content_references
                    if isinstance(entry.get("bbox"), tuple)
                ])
                + note_references,
                heading_levels=heading_levels,
            )
            if page_text.strip():
                if page_write:
                    output_text.append(f"{_format_page_comment(page_idx)}\n{page_text}")
                else:
                    output_text.append(page_text)

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
        image_refs_by_page=embedded_image_refs_by_page,
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
