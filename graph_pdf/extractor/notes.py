from __future__ import annotations

import re
from typing import List, Sequence

import pdfplumber

from .tables import (
    _candidate_image_regions_for_notes,
    _compact_fallback_rows,
    _extract_region_line_payloads,
    _extract_region_line_rows,
    _note_anchors_for_bbox,
    _note_group_region_candidates,
    _select_note_anchor_for_bbox,
)
from .shared import _normalize_text
from .text import _normalize_cell_lines


def _note_body_text(rows: Sequence[Sequence[str]]) -> str:
    # Convert multi-line note-like rows into a single body sentence.
    parts: List[str] = []
    for row in rows:
        if not row:
            continue
        leading_text = ""
        for cell in row:
            normalized = _normalize_text(cell)
            if normalized:
                leading_text = normalized
                break
        if not leading_text:
            continue
        for line in _normalize_cell_lines(str(leading_text)):
            normalized = _normalize_text(line)
            if normalized:
                parts.append(normalized)
    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    text = re.sub(r"(?<=[A-Za-z0-9])([‘“])", r" \1", text)
    if not text:
        return ""
    if re.match(r"(?i)^note\s*:", text):
        return re.sub(r"(?i)^note\s*:\s*", "Note: ", text, count=1)
    return f"Note: {text}"


def _split_note_rows_by_anchors(
    page: pdfplumber.page.PageObject,
    bbox: tuple[float, float, float, float],
    *,
    image_regions: Sequence[tuple[float, float, float, float]] | None = None,
) -> list[dict[str, object]]:
    anchors = _note_anchors_for_bbox(page, bbox, image_regions=image_regions)
    if len(anchors) <= 1:
        return []

    line_payloads = _extract_region_line_payloads(page, bbox)
    if not line_payloads:
        return []

    rows_by_anchor: list[list[list[str]]] = [[] for _ in anchors]
    tops_by_anchor: list[list[float]] = [[] for _ in anchors]
    bottoms_by_anchor: list[list[float]] = [[] for _ in anchors]
    for line in line_payloads:
        line_text = _normalize_text(str(line.get("text") or ""))
        if not line_text:
            continue
        anchor_index = 0
        line_top = float(line["top"])
        for idx, anchor in enumerate(anchors):
            next_anchor_top = float(anchors[idx + 1][1]) if idx + 1 < len(anchors) else float("inf")
            if line_top < next_anchor_top:
                anchor_index = idx
                break
        rows_by_anchor[anchor_index].append([line_text])
        tops_by_anchor[anchor_index].append(float(line["top"]))
        bottoms_by_anchor[anchor_index].append(float(line["bottom"]))

    split_candidates: list[dict[str, object]] = []
    for idx, anchor in enumerate(anchors):
        rows = _compact_fallback_rows(rows_by_anchor[idx])
        if not rows:
            continue
        split_bbox = (
            bbox[0],
            min(anchor[1], min(tops_by_anchor[idx], default=anchor[1])),
            bbox[2],
            max(anchor[3], max(bottoms_by_anchor[idx], default=anchor[3])),
        )
        split_candidates.append(
            {
                "bbox": split_bbox,
                "rows": rows,
                "note_anchor": tuple(round(value, 2) for value in anchor),
            }
        )
    return split_candidates


def _collect_note_candidates(
    page: pdfplumber.page.PageObject,
) -> List[dict]:
    # Collect note candidates strictly from note-group geometry and note anchors.
    candidate_rows: List[dict] = []
    image_regions = _candidate_image_regions_for_notes(page)
    note_group_candidates = _note_group_region_candidates(page, image_regions=image_regions)
    for bbox in note_group_candidates:
        split_candidates = _split_note_rows_by_anchors(page, bbox, image_regions=image_regions)
        if split_candidates:
            for split_candidate in split_candidates:
                candidate_rows.append(
                    {
                        "bbox": split_candidate["bbox"],
                        "raw_bbox": bbox,
                        "rows": split_candidate["rows"],
                        "is_white_content": False,
                        "is_note_like": True,
                        "note_anchor": split_candidate["note_anchor"],
                        "note_band": (bbox[1], bbox[3]),
                        "note_group_source": True,
                    }
                )
            continue

        group_rows = _compact_fallback_rows(_extract_region_line_rows(page, bbox))
        if not group_rows:
            continue
        note_anchor = _select_note_anchor_for_bbox(page, bbox, image_regions=image_regions)
        candidate_rows.append(
            {
                "bbox": bbox,
                "raw_bbox": bbox,
                "rows": group_rows,
                "is_white_content": False,
                "is_note_like": True,
                "note_anchor": (
                    tuple(round(value, 2) for value in note_anchor)
                    if note_anchor is not None
                    else None
                ),
                "note_band": (bbox[1], bbox[3]),
                "note_group_source": True,
            }
        )
    return candidate_rows
