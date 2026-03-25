from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import pdfplumber

from .debug import _collect_page_edge_debug, _collect_rotated_text_debug, _collect_table_drawing_debug
from .images import _collect_embedded_image_refs, _extract_embedded_images
from .raw import materialize_raw_dump
from .shared import TableRows, _merge_numeric_positions, _normalize_text
from .tables import (
    _append_output_table,
    _body_text_boxes,
    _continuation_regions_should_merge,
    _extract_tables,
    _has_gap_text_after_bbox,
    _has_gap_text_before_bbox,
    _looks_like_single_column_note,
    _should_try_table_continuation_merge,
    _single_column_note_body_text,
    _vertical_axes_for_bbox,
)
from .text import _detect_body_bounds, _extract_body_text, _extract_drawing_image_bboxes


def _to_float_bbox(raw_bbox: object) -> tuple[float, float, float, float] | None:
    if not isinstance(raw_bbox, Sequence) or len(raw_bbox) != 4:
        return None
    try:
        return tuple(float(value) for value in raw_bbox)
    except (TypeError, ValueError):
        return None


def _rounded_bbox(raw_bbox: Sequence[float] | tuple[float, float, float, float]) -> list[float]:
    return [round(float(value), 2) for value in raw_bbox]


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


def _heading_max_x0_from_rule(match: dict[str, Any]) -> float | None:
    if "max_x0" not in match:
        return None
    try:
        max_x0 = float(match["max_x0"])
    except (TypeError, ValueError):
        return None
    return max_x0 if max_x0 >= 0 else None


def _load_heading_levels(add_heading: Path | None) -> dict[float, dict[str, float | int]] | None:
    if add_heading is None:
        return None

    payload = json.loads(Path(add_heading).read_text(encoding="utf-8"))
    heading_levels: dict[float, dict[str, float | int]] = {}
    for rule in payload.get("heading_rules", []):
        match = rule.get("match") or {}
        if "font_size" not in match:
            continue
        level = _heading_level_from_rule(rule)
        if level is None:
            continue
        font_size = round(float(match["font_size"]), 2)
        heading_levels[font_size] = {
            "level": level,
            "max_x0": _heading_max_x0_from_rule(match),
        }
    return heading_levels


def _format_page_comment(page_no: int) -> str:
    return f"[//]: # (Page {page_no})"


_DOC_ID_HEADING_PREFIX_RE = re.compile(r"^##\s+(?P<doc_id>.+?)\s*$")
_UNSAFE_DOC_ID_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]")


def _extract_document_id_from_markdown_line(line: str) -> str | None:
    match = _DOC_ID_HEADING_PREFIX_RE.match(line.strip())
    if not match:
        return None
    doc_id = match.group("doc_id").strip()
    if not doc_id:
        return None
    first_token = doc_id.split(maxsplit=1)[0]
    if not first_token:
        return None
    return first_token[:10] if len(first_token) >= 10 else first_token


def _safe_document_id(document_id: str) -> str:
    document_id = document_id.strip()
    safe = _UNSAFE_DOC_ID_CHARS_RE.sub("_", document_id)
    return safe or "document"


def _extract_document_id(markdown: str) -> str | None:
    for line in markdown.splitlines():
        doc_id = _extract_document_id_from_markdown_line(line)
        if doc_id:
            return doc_id
    return None


def _infer_document_id_from_pdf(
    pdf_path: Path,
    heading_levels: Optional[dict[float, dict[str, float | int] | int]],
    header_margin: float,
    footer_margin: float,
    selected_pages: Optional[set[int]] = None,
) -> str | None:
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                page_no = int(getattr(page, "page_number", 0) or 0)
                if selected_pages and page_no and page_no not in selected_pages:
                    continue
                page_markdown = _extract_body_text(
                    page,
                    header_margin=header_margin,
                    footer_margin=footer_margin,
                    heading_levels=heading_levels,
                )
                doc_id = _extract_document_id(page_markdown)
                if doc_id:
                    return doc_id
    except Exception:
        return None
    return None


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
    tables: Sequence[Tuple[TableRows, Tuple[float, float, float, float]]],
    image_regions: Sequence[Tuple[float, float, float, float]],
) -> List[Tuple[float, float, float, float]]:
    excluded = [bbox for _rows, bbox in tables]
    excluded.extend(list(image_regions))
    return excluded


def _content_ref_text(content_type: str, document_id: str, index: int, continued: bool = False) -> str:
    label = f"[{content_type} reference: {document_id} {content_type.lower()} {index}]"
    if continued:
        label += " (continued)"
    return label


def _should_continue_content_region(
    prev_bbox: Tuple[float, float, float, float],
    curr_bbox: Tuple[float, float, float, float],
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


@dataclass
class _PendingTableState:
    chunks: list[TableRows] = field(default_factory=list)
    table_no: int | None = None
    start_page: int | None = None
    last_page: int | None = None
    bbox: Tuple[float, float, float, float] | None = None
    axes: List[float] = field(default_factory=list)
    has_gap_text: bool = False
    page_height: float | None = None

    def is_active(self) -> bool:
        return bool(self.chunks)

    def clear(self) -> None:
        self.chunks.clear()
        self.table_no = None
        self.start_page = None
        self.last_page = None
        self.bbox = None
        self.axes.clear()
        self.has_gap_text = False
        self.page_height = None

    def append_chunk(self, rows: TableRows) -> None:
        self.chunks.append(rows)

    def flattened_rows(self) -> TableRows:
        normalized = _strip_repeated_headers_by_chunk(self.chunks)
        return normalized

    @property
    def last_chunk_first_row_signature(self) -> tuple[str, ...]:
        if not self.chunks:
            return ()
        return _first_table_row_signature(self.chunks[-1][0]) if self.chunks[-1] else ()


@dataclass
class _DocumentOutputState:
    document_id: str
    output_text: List[str] = field(default_factory=list)
    output_tables: List[str] = field(default_factory=list)
    table_debug_pages: List[dict] = field(default_factory=list)
    edge_debug_pages: List[dict] = field(default_factory=list)
    rotated_debug: List[dict] = field(default_factory=list)
    pending_table_state: _PendingTableState = field(default_factory=_PendingTableState)
    next_table_no: int = 1
    next_image_no: int = 1
    pending_image_ref_bbox: Optional[Tuple[float, float, float, float]] = None
    pending_image_body_bottom: Optional[float] = None
    emitted_table_references: set[int] = field(default_factory=set)
    pages: List[int] = field(default_factory=list)

    def clear_transient_content_state(self) -> None:
        self.pending_image_ref_bbox = None
        self.pending_image_body_bottom = None


def _first_table_row_signature(table_rows: Sequence[Sequence[str]]) -> tuple[str, ...]:
    if not table_rows:
        return ()
    return tuple(_normalize_text(cell) for cell in table_rows[0])


def _strip_repeated_headers_by_chunk(chunks: Sequence[TableRows]) -> TableRows:
    normalized_rows: TableRows = []
    previous_header_signature: tuple[str, ...] = ()
    for chunk in chunks:
        if not chunk:
            continue
        current_signature = _first_table_row_signature(chunk)
        if normalized_rows and current_signature and current_signature == previous_header_signature:
            normalized_rows.extend(chunk[1:])
        else:
            normalized_rows.extend(chunk)
        previous_header_signature = _first_table_row_signature(chunk)
    return normalized_rows


def _has_intervening_regions(
    page_regions: dict[str, Any],
    table_bbox: Tuple[float, float, float, float],
    after_table_bottom: bool,
) -> bool:
    table_top = float(table_bbox[1])
    table_bottom = float(table_bbox[3])
    table_bbox_tuple = (
        float(table_bbox[0]),
        table_top,
        float(table_bbox[2]),
        table_bottom,
    )
    for region_key in ("text", "tables", "images"):
        entries = page_regions.get(region_key)
        if not entries:
            continue
        for entry in entries:
            raw_bbox = entry.get("bbox") if isinstance(entry, dict) else entry
            bbox = _to_float_bbox(raw_bbox)
            if bbox is None:
                continue
            if bbox == table_bbox_tuple:
                continue
            if after_table_bottom:
                if bbox[1] >= table_bottom + 1.0:
                    return True
            elif bbox[3] <= table_top - 1.0:
                return True
    return False


def _has_intervening_regions_before_table(
    page_regions: dict[str, Any],
    table_bbox: Tuple[float, float, float, float],
) -> bool:
    return _has_intervening_regions(
        page_regions=page_regions,
        table_bbox=table_bbox,
        after_table_bottom=False,
    )


def _has_intervening_regions_after_table(
    page_regions: dict[str, Any],
    table_bbox: Tuple[float, float, float, float],
) -> bool:
    return _has_intervening_regions(
        page_regions=page_regions,
        table_bbox=table_bbox,
        after_table_bottom=True,
    )


def _has_cross_page_gap_blocked(
    region_map: dict[int, dict[str, Any]],
    previous_page: int,
    previous_table_bbox: Tuple[float, float, float, float],
    current_page: int,
    current_table_bbox: Tuple[float, float, float, float],
) -> bool:
    previous_regions = region_map.get(previous_page, {})
    current_regions = region_map.get(current_page, {})
    if not previous_regions:
        return False
    if _has_intervening_regions_after_table(previous_regions, previous_table_bbox):
        return True

    if not current_regions:
        return True
    if _has_intervening_regions_before_table(current_regions, current_table_bbox):
        return True

    return False


def _table_reference_text(document_id: str, table_no: int) -> str:
    return f"[Table reference: {document_id} table {table_no}]"


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
    region_log: Path | None = None,
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
                region_log=region_log,
            )
    if pdf_path is None:
        raise ValueError("pdf_path is required when from_raw is not provided")
    out_md_dir.mkdir(parents=True, exist_ok=True)
    out_image_dir.mkdir(parents=True, exist_ok=True)

    selected_pages = set(int(page_no) for page_no in (pages or []))
    drawing_regions_by_page: dict[int, list[Tuple[float, float, float, float]]] = {}
    note_regions_by_page: dict[int, list[Tuple[float, float, float, float]]] = {}
    region_map: dict[int, dict[str, Any]] = {}
    heading_levels = _load_heading_levels(add_heading)
    initial_document_id = _safe_document_id(stem)
    current_document_state = _DocumentOutputState(document_id=initial_document_id)
    document_artifacts: list[dict] = []
    image_files: list[Path] = []
    table_debug_pages: List[dict] | None = [] if debug else None
    edge_debug_pages: List[dict] | None = [] if debug else None
    rotated_debug: List[dict] | None = [] if debug_watermark else None
    total_table_count = 0

    embedded_image_refs_by_page = _collect_embedded_image_refs(
        pdf_path=pdf_path,
        pages=pages,
        header_margin=header_margin,
        footer_margin=footer_margin,
    )

    def _flush_pending_table(state: _DocumentOutputState) -> None:
        if state.pending_table_state.is_active() and state.pending_table_state.start_page is not None and state.pending_table_state.table_no is not None:
            _append_output_table(
                state.output_tables,
                state.document_id,
                state.pending_table_state.table_no,
                state.pending_table_state.flattened_rows(),
                page_no=state.pending_table_state.start_page if page_write else None,
            )
        state.pending_table_state.clear()

    def _append_table_reference(
        state: _DocumentOutputState,
        refs: List[dict],
        table_no: int,
        bbox: Tuple[float, float, float, float],
    ) -> None:
        key = table_no
        if key in state.emitted_table_references:
            return
        state.emitted_table_references.add(key)
        refs.append({"text": _table_reference_text(state.document_id, table_no), "bbox": bbox})

    def _collect_embedded_image_references(
        embedded_refs: Sequence[dict],
        body_top: float,
        body_bottom: float,
    ) -> List[dict]:
        page_content_references: list[dict] = []
        for entry in embedded_refs:
            bbox = _to_float_bbox(entry.get("bbox") if isinstance(entry, dict) else None)
            if bbox is None:
                continue

            is_cont = False
            if (
                current_document_state.pending_image_ref_bbox is not None
                and current_document_state.pending_image_body_bottom is not None
            ):
                is_cont = _should_continue_content_region(
                    prev_bbox=current_document_state.pending_image_ref_bbox,
                    curr_bbox=bbox,
                    prev_body_bottom=current_document_state.pending_image_body_bottom,
                    curr_body_top=body_top,
                )

            page_content_references.append(
                {
                    "text": _content_ref_text(
                        "Image",
                        current_document_state.document_id,
                        current_document_state.next_image_no,
                        continued=is_cont,
                    ),
                    "bbox": bbox,
                }
            )
            current_document_state.next_image_no += 1

            if _is_edge_candidate_for_continuation(bbox=bbox, body_top=body_top, body_bottom=body_bottom):
                current_document_state.pending_image_ref_bbox = bbox
                current_document_state.pending_image_body_bottom = body_bottom
            else:
                current_document_state.pending_image_ref_bbox = None
                current_document_state.pending_image_body_bottom = None
        return page_content_references

    def _flush_current_document(state: _DocumentOutputState) -> dict:
        nonlocal total_table_count
        _flush_pending_table(state)
        state.clear_transient_content_state()

        markdown = "\n\n".join(state.output_text)
        table_markdown = "\n\n".join(state.output_tables)

        document_id = state.document_id
        text_file = out_md_dir / f"{document_id}.txt"
        md_file = out_md_dir / f"{document_id}.md"
        table_md_file = out_md_dir / f"{document_id}_table.md"
        text_file.write_text(markdown, encoding="utf-8")
        md_file.write_text(markdown, encoding="utf-8")
        table_md_file.write_text(table_markdown, encoding="utf-8")

        # Image extraction happens after text/table rendering so image export stays independent from markdown generation.
        document_images = _extract_embedded_images(
            pdf_path=pdf_path,
            out_image_dir=out_image_dir,
            stem=document_id,
            pages=state.pages,
            drawing_regions_by_page=drawing_regions_by_page,
            image_refs_by_page=embedded_image_refs_by_page,
            excluded_regions_by_page=note_regions_by_page,
        )
        image_files.extend(document_images)

        summary = {
            "pdf": str(pdf_path),
            "document_id": document_id,
            "text_file": str(text_file),
            "md_file": str(md_file),
            "table_md_file": str(table_md_file),
            "images": [str(p) for p in document_images],
            "table_count": len(state.output_tables),
            "pages": state.pages,
        }
        summary_file = out_md_dir / f"{document_id}_summary.json"
        summary_file.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        total_table_count += len(state.output_tables)

        artifact = {
            "document_id": document_id,
            "markdown": markdown,
            "table_markdown": table_markdown,
            "text_file": str(text_file),
            "md_file": str(md_file),
            "table_md_file": str(table_md_file),
            "summary_file": str(summary_file),
            "summary": summary,
            "image_files": [str(p) for p in document_images],
        }
        document_artifacts.append(artifact)
        return artifact

    def _commit_document_switch(new_document_id: str) -> None:
        nonlocal current_document_state
        current = current_document_state
        if current is None:
            current_document_state = _DocumentOutputState(document_id=_safe_document_id(new_document_id))
            return

        if _safe_document_id(current.document_id) != _safe_document_id(new_document_id):
            _flush_current_document(current)
            current_document_state = _DocumentOutputState(document_id=_safe_document_id(new_document_id))

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            if selected_pages and page_idx not in selected_pages:
                _flush_pending_table(current_document_state)
                current_document_state.clear_transient_content_state()
                continue

            preview_markdown = _extract_body_text(
                page,
                header_margin=header_margin,
                footer_margin=footer_margin,
                heading_levels=heading_levels,
            )
            detected_document_id = _extract_document_id(preview_markdown)
            if detected_document_id is not None:
                _commit_document_switch(detected_document_id)

            current_document_state.pages.append(page_idx)

            if debug:
                table_debug_pages.append(
                    _collect_table_drawing_debug(page, page_no=page_idx, header_margin=header_margin, footer_margin=footer_margin)
                )
                edge_debug_pages.append(
                    _collect_page_edge_debug(page, page_no=page_idx, header_margin=header_margin, footer_margin=footer_margin)
                )
            if debug_watermark:
                rotated_debug.extend(_collect_rotated_text_debug(page, page_no=page_idx))

            strategy_debug: list[dict] | None = [] if debug else None
            detected_tables = _extract_tables(
                page,
                force_table=force_table,
                strategy_debug=strategy_debug,
            )
            tables: List[Tuple[TableRows, Tuple[float, float, float, float]]] = []
            note_references: List[dict] = []
            detected_table_payloads: List[dict] = []
            table_exclusion_regions: list[Tuple[float, float, float, float]] = []
            for table_rows, bbox in detected_tables:
                table_exclusion_regions.append(bbox)
                row_count = len(table_rows)
                col_count = max((len(row) for row in table_rows), default=0)
                is_note = _looks_like_single_column_note(
                    rows=table_rows,
                    page=page,
                    bbox=bbox,
                )
                if is_note:
                    note_regions_by_page.setdefault(page_idx, []).append(bbox)
                    detected_table_payloads.append(
                        {
                            "kind": "note",
                            "bbox": [round(float(value), 2) for value in bbox],
                            "row_count": int(row_count),
                            "col_count": int(col_count),
                        }
                    )
                    note_text = _single_column_note_body_text(table_rows)
                    if note_text:
                        note_references.append({"text": note_text, "bbox": bbox})
                    continue
                tables.append((table_rows, bbox))
                detected_table_payloads.append(
                    {
                        "kind": "table",
                        "bbox": _rounded_bbox(bbox),
                        "row_count": int(len(table_rows)),
                        "col_count": int(max((len(row) for row in table_rows), default=0)),
                    }
                )
            if debug and table_debug_pages:
                table_debug_pages[-1]["detected_tables"] = detected_table_payloads
                if strategy_debug is not None:
                    table_debug_pages[-1]["strategy_debug"] = strategy_debug

            body_top, body_bottom = _detect_body_bounds(page, header_margin=header_margin, footer_margin=footer_margin)
            image_regions = _extract_drawing_image_bboxes(
                page=page,
                header_margin=header_margin,
                footer_margin=footer_margin,
                excluded_bboxes=table_exclusion_regions,
            )
            drawing_regions_by_page[page_idx] = image_regions
            embedded_image_refs = embedded_image_refs_by_page.get(page_idx, [])
            embedded_image_regions: list[Tuple[float, float, float, float]] = []
            for image_ref in embedded_image_refs:
                if not isinstance(image_ref, dict):
                    continue
                image_bbox = _to_float_bbox(image_ref.get("bbox"))
                if image_bbox is None:
                    continue
                embedded_image_regions.append(image_bbox)
            body_text_regions = _body_text_boxes(
                page=page,
                header_margin=header_margin,
                footer_margin=footer_margin,
                excluded_bboxes=table_exclusion_regions,
            )
            table_regions = list(table_exclusion_regions)
            if region_log is not None:
                embedded_regions_payload = []
                for image_idx, entry in enumerate(embedded_image_refs, start=1):
                    if not isinstance(entry, dict):
                        continue
                    image_bbox = _to_float_bbox(entry.get("bbox"))
                    if image_bbox is None:
                        continue
                    embedded_regions_payload.append(
                        {
                            "kind": "image",
                            "source": "embedded",
                            "bbox": _rounded_bbox(image_bbox),
                            "page_index": image_idx,
                        }
                    )
                tables_regions_payload = [
                    {
                        "kind": entry["kind"],
                        "bbox": _rounded_bbox(entry["bbox"]),
                        "row_count": int(entry["row_count"]),
                        "col_count": int(entry["col_count"]),
                    }
                    for entry in detected_table_payloads
                ]
                unique_tables_regions: list[dict[str, Any]] = []
                table_region_keys: set[tuple[Any, ...]] = set()
                for entry in tables_regions_payload:
                    key = (entry["kind"], tuple(entry["bbox"]), int(entry["row_count"]), int(entry["col_count"]))
                    if key in table_region_keys:
                        continue
                    table_region_keys.add(key)
                    unique_tables_regions.append(entry)

                region_map[page_idx] = {
                    "tables": unique_tables_regions,
                    "text": [{"bbox": _rounded_bbox(bbox)} for bbox in body_text_regions],
                    "images": embedded_regions_payload
                    + [{"kind": "image", "source": "drawing", "bbox": _rounded_bbox(image_bbox)} for image_bbox in image_regions],
                    "body_top": float(body_top),
                    "body_bottom": float(body_bottom),
                    "header_margin": float(header_margin),
                    "footer_margin": float(footer_margin),
                }
            else:
                region_map[page_idx] = {
                    "tables": table_regions,
                    "text": body_text_regions,
                    "images": [*embedded_image_regions, *image_regions],
                }
                obsolete_page_threshold = page_idx - 1
                for obsolete_page in [p for p in region_map.keys() if p < obsolete_page_threshold]:
                    region_map.pop(obsolete_page, None)

            page_excluded_bboxes = _body_excluded_bboxes(
                tables=tables,
                image_regions=[
                    *image_regions,
                    *embedded_image_regions,
                ],
            )
            page_excluded_bboxes.extend(table_exclusion_regions)
            page_table_references: List[dict] = []
            page_content_references: List[dict] = []

            if not tables:
                page_content_references = _collect_embedded_image_references(
                    embedded_refs=embedded_image_refs,
                    body_top=body_top,
                    body_bottom=body_bottom,
                )

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
                        current_document_state.output_text.append(f"{_format_page_comment(page_idx)}\n{page_text}")
                    else:
                        current_document_state.output_text.append(page_text)
                continue

            tables = sorted(tables, key=lambda item: item[1][1])
            for table_index, (table_rows, bbox) in enumerate(tables):
                cross_page_continuation = _should_try_table_continuation_merge(
                    pending_page=current_document_state.pending_table_state.last_page,
                    current_page=page_idx,
                )
                has_current_gap_text = _has_gap_text_before_bbox(
                    body_text_regions,
                    bbox,
                )
                current_axes = _vertical_axes_for_bbox(page, bbox)

                can_merge_cross_page = (
                    table_index == 0
                    and cross_page_continuation
                    and current_document_state.pending_table_state.is_active()
                    and current_document_state.pending_table_state.last_page is not None
                    and not _has_cross_page_gap_blocked(
                        region_map=region_map,
                        previous_page=current_document_state.pending_table_state.last_page,
                        previous_table_bbox=current_document_state.pending_table_state.bbox if current_document_state.pending_table_state.bbox is not None else bbox,
                        current_page=page_idx,
                        current_table_bbox=bbox,
                    )
                )
                if can_merge_cross_page:
                    if _continuation_regions_should_merge(
                        prev_bbox=current_document_state.pending_table_state.bbox if current_document_state.pending_table_state.bbox is not None else bbox,
                        curr_bbox=bbox,
                        prev_axes=current_document_state.pending_table_state.axes,
                        curr_axes=current_axes,
                        body_top=body_top,
                        body_bottom=body_bottom,
                        has_gap_text=current_document_state.pending_table_state.has_gap_text or has_current_gap_text,
                        prev_page_height=current_document_state.pending_table_state.page_height,
                    ):
                        if current_document_state.pending_table_state.start_page is not None and current_document_state.pending_table_state.table_no is not None:
                            _append_table_reference(
                                state=current_document_state,
                                refs=page_table_references,
                                table_no=current_document_state.pending_table_state.table_no,
                                bbox=bbox,
                            )
                        current_document_state.pending_table_state.append_chunk(table_rows)
                        current_document_state.pending_table_state.last_page = page_idx
                        current_document_state.pending_table_state.bbox = (
                            min(current_document_state.pending_table_state.bbox[0], bbox[0]),
                            min(current_document_state.pending_table_state.bbox[1], bbox[1]),
                            max(current_document_state.pending_table_state.bbox[2], bbox[2]),
                            max(current_document_state.pending_table_state.bbox[3], bbox[3]),
                        )
                        current_document_state.pending_table_state.page_height = float(page.height)
                        current_document_state.pending_table_state.axes = _merge_numeric_positions([*current_document_state.pending_table_state.axes, *current_axes], tolerance=1.0)
                        current_document_state.pending_table_state.has_gap_text = _has_gap_text_after_bbox(body_text_regions, bbox)
                        continue
                _flush_pending_table(current_document_state)

                current_table_no = current_document_state.next_table_no
                current_document_state.next_table_no += 1
                _append_table_reference(
                    state=current_document_state,
                    refs=page_table_references,
                    table_no=current_table_no,
                    bbox=bbox,
                )
                current_document_state.pending_table_state.chunks = [table_rows]
                current_document_state.pending_table_state.table_no = current_table_no
                current_document_state.pending_table_state.start_page = page_idx
                current_document_state.pending_table_state.last_page = page_idx
                current_document_state.pending_table_state.bbox = bbox
                current_document_state.pending_table_state.axes = current_axes
                current_document_state.pending_table_state.page_height = float(page.height)
                current_document_state.pending_table_state.has_gap_text = _has_gap_text_after_bbox(
                    body_text_regions,
                    bbox,
                )

            page_content_references = _collect_embedded_image_references(
                embedded_refs=embedded_image_refs,
                body_top=body_top,
                body_bottom=body_bottom,
            )

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
                    current_document_state.output_text.append(f"{_format_page_comment(page_idx)}\n{page_text}")
                else:
                    current_document_state.output_text.append(page_text)

    _flush_current_document(current_document_state)
    if not document_artifacts:
        _flush_current_document(current_document_state)

    markdown = "\n\n".join(artifact["markdown"] for artifact in document_artifacts if artifact["markdown"])
    table_markdown = "\n\n".join(artifact["table_markdown"] for artifact in document_artifacts if artifact["table_markdown"])

    primary_document = document_artifacts[0]
    summary = {
        "pdf": str(pdf_path),
        "text_file": primary_document["text_file"],
        "md_file": primary_document["md_file"],
        "table_md_file": primary_document["table_md_file"],
        "images": [str(path) for path in image_files],
        "table_count": total_table_count,
        "document_count": len(document_artifacts),
        "documents": [artifact["summary"] for artifact in document_artifacts],
    }

    summary_file = out_md_dir / f"{_safe_document_id(stem)}_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_file"] = str(summary_file)

    debug_file: Optional[Path] = None
    debug_edges_file: Optional[Path] = None
    if debug:
        debug_file = out_md_dir / f"{_safe_document_id(stem)}_debug.json"
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
        debug_edges_file = out_md_dir / f"{_safe_document_id(stem)}_edges_debug.json"
        debug_edges_file.write_text(
            json.dumps({"pdf": str(pdf_path), "pages": edge_debug_pages}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    debug_watermark_file: Optional[Path] = None
    if debug_watermark:
        debug_watermark_file = out_md_dir / f"{_safe_document_id(stem)}_watermark_debug.json"
        debug_watermark_file.write_text(json.dumps(rotated_debug, ensure_ascii=False, indent=2), encoding="utf-8")

    region_log_file: Optional[Path] = None
    if region_log is not None:
        region_log.parent.mkdir(parents=True, exist_ok=True)
        region_log_file = region_log
        region_log.write_text(
            json.dumps(
                {
                    "pdf": str(pdf_path),
                    "pages": region_map,
                    "header_margin": header_margin,
                    "footer_margin": footer_margin,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return {
        "markdown": markdown,
        "table_markdown": table_markdown,
        "text_file": Path(primary_document["text_file"]),
        "md_file": Path(primary_document["md_file"]),
        "table_md_file": Path(primary_document["table_md_file"]),
        "debug_file": debug_file,
        "debug_edges_file": debug_edges_file,
        "debug_watermark_file": debug_watermark_file,
        "region_log_file": region_log_file,
        "image_files": image_files,
        "documents": document_artifacts,
        "summary": summary,
    }
