from __future__ import annotations

import re
from typing import Any, List, Sequence

import pdfplumber

from .tables import (
    _THIN_FILL_RECT_MAX_HEIGHT,
    _bbox_x_overlap_ratio,
    _compact_fallback_rows,
    _extract_region_line_payloads,
    _extract_region_line_rows,
    _line_color_key,
    _normalize_color_match,
)
from .shared import _normalize_text
from .text import _detect_body_bounds, _normalize_cell_lines


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


def _candidate_image_regions_for_notes(
    page: pdfplumber.page.PageObject,
    min_width: float = 12.0,
    min_height: float = 10.0,
) -> List[tuple[float, float, float, float]]:
    body_top, body_bottom = _detect_body_bounds(page, header_margin=90.0, footer_margin=40.0)
    regions: List[tuple[float, float, float, float]] = []
    for image in getattr(page, "images", []) or []:
        x0 = float(image.get("x0", 0.0))
        top = float(image.get("top", 0.0))
        x1 = float(image.get("x1", x0))
        bottom = float(image.get("bottom", top))
        if x1 <= x0 or bottom <= top:
            continue
        if (x1 - x0) < min_width or (bottom - top) < min_height:
            continue
        if bottom <= body_top or top >= body_bottom:
            continue
        regions.append((x0, top, x1, bottom))
    regions.sort(key=lambda bbox: (bbox[1], bbox[0]))
    return regions


def _blue_note_horizontal_segments(
    page: pdfplumber.page.PageObject,
    min_length: float = 90.0,
) -> List[dict[str, Any]]:
    body_top, body_bottom = _detect_body_bounds(page, header_margin=90.0, footer_margin=40.0)

    def is_blue_color(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return False
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            r = float(value[0])
            g = float(value[1])
            b = float(value[2])
            return b > 0.45 and r < 0.35 and g < 0.35
        return False

    segments: list[dict[str, Any]] = []
    for edge in getattr(page, "horizontal_edges", []):
        x0 = float(edge.get("x0", 0.0))
        x1 = float(edge.get("x1", x0))
        top = float(edge.get("top", edge.get("y0", 0.0)))
        bottom = float(edge.get("bottom", top))
        if top <= body_top or bottom >= body_bottom:
            continue
        if x1 - x0 < min_length:
            continue
        color = _line_color_key(edge)
        if not is_blue_color(color):
            continue
        segments.append({"x0": x0, "x1": x1, "top": top, "bottom": bottom, "color": color})

    for rect in getattr(page, "rects", []) or []:
        if not bool(rect.get("fill")) or bool(rect.get("stroke")):
            continue
        rect_top = float(rect.get("top", 0.0))
        rect_bottom = float(rect.get("bottom", rect_top))
        if rect_top <= body_top or rect_bottom >= body_bottom:
            continue
        if abs(rect_bottom - rect_top) > _THIN_FILL_RECT_MAX_HEIGHT:
            continue
        x0 = float(rect.get("x0", 0.0))
        x1 = float(rect.get("x1", x0))
        if x1 - x0 < min_length:
            continue
        color = rect.get("non_stroking_color")
        if color is None:
            color = rect.get("stroking_color")
        if not is_blue_color(color):
            continue
        segments.append(
            {
                "x0": x0,
                "x1": x1,
                "top": rect_top,
                "bottom": rect_bottom,
                "color": _line_color_key({"stroking_color": color}),
            }
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float, float, tuple[Any, ...] | int | float | None]] = set()
    for segment in sorted(segments, key=lambda item: (float(item["top"]), float(item["x0"]))):
        key = (
            round(float(segment["x0"]), 2),
            round(float(segment["top"]), 2),
            round(float(segment["x1"]), 2),
            round(float(segment["bottom"]), 2),
            segment["color"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(segment)
    return deduped


def _note_group_region_candidates(
    page: pdfplumber.page.PageObject,
    image_regions: Sequence[tuple[float, float, float, float]] | None = None,
) -> List[tuple[float, float, float, float]]:
    image_regions = list(image_regions or _candidate_image_regions_for_notes(page))
    if not image_regions:
        return []

    segments = _blue_note_horizontal_segments(page)
    if not segments:
        return []

    candidates: list[tuple[float, float, float, float]] = []
    start_tolerance = 12.0
    end_tolerance = 12.0

    for anchor in image_regions:
        anchor_area = (float(anchor[0]), float(anchor[1]), float(anchor[2]), float(anchor[3]))
        top_segment: dict[str, Any] | None = None
        bottom_segment: dict[str, Any] | None = None

        for segment in segments:
            segment_area = (
                float(segment["x0"]),
                float(segment["top"]),
                float(segment["x1"]),
                float(segment["bottom"]),
            )
            if _bbox_x_overlap_ratio(anchor_area, segment_area) < 0.10:
                continue
            if float(segment["bottom"]) <= float(anchor[1]):
                top_segment = segment
            elif float(segment["top"]) >= float(anchor[3]):
                if bottom_segment is None:
                    bottom_segment = segment
                break

        if top_segment is None or bottom_segment is None:
            continue
        if not _normalize_color_match(top_segment["color"], bottom_segment["color"]):
            continue
        if abs(float(top_segment["x0"]) - float(bottom_segment["x0"])) > start_tolerance:
            continue
        if abs(float(top_segment["x1"]) - float(bottom_segment["x1"])) > end_tolerance:
            continue

        candidates.append(
            (
                min(float(top_segment["x0"]), float(bottom_segment["x0"])),
                min(float(top_segment["top"]), float(top_segment["bottom"])),
                max(float(top_segment["x1"]), float(bottom_segment["x1"])),
                max(float(bottom_segment["top"]), float(bottom_segment["bottom"])),
            )
        )

    if not candidates:
        return []

    deduped: list[tuple[float, float, float, float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for bbox in sorted(candidates, key=lambda item: (item[1], item[0])):
        key = tuple(round(float(value), 2) for value in bbox)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(bbox)
    return deduped


def _select_note_anchor_for_bbox(
    page: pdfplumber.page.PageObject,
    bbox: tuple[float, float, float, float],
    image_regions: Sequence[tuple[float, float, float, float]] | None = None,
) -> tuple[float, float, float, float] | None:
    image_regions = list(image_regions or _candidate_image_regions_for_notes(page))
    if not image_regions:
        return None

    note_y0, note_y1 = bbox[1], bbox[3]
    note_center_y = (note_y0 + note_y1) / 2.0
    best_anchor: tuple[float, float, float, float] | None = None
    best_score = -1.0

    for region in image_regions:
        _image_x0, image_top, _image_x1, image_bottom = region
        x_overlap = _bbox_x_overlap_ratio(region, bbox)
        if x_overlap < 0.10:
            continue
        if image_bottom < note_y0 or image_top > note_y1:
            continue

        image_center_y = (image_top + image_bottom) / 2.0
        score = x_overlap * 100.0 - abs(note_center_y - image_center_y)
        if score > best_score:
            best_score = score
            best_anchor = region

    return best_anchor


def _note_anchors_for_bbox(
    page: pdfplumber.page.PageObject,
    bbox: tuple[float, float, float, float],
    image_regions: Sequence[tuple[float, float, float, float]] | None = None,
) -> List[tuple[float, float, float, float]]:
    image_regions = list(image_regions or _candidate_image_regions_for_notes(page))
    if not image_regions:
        return []

    note_y0, note_y1 = bbox[1], bbox[3]
    matches: list[tuple[float, float, float, float]] = []
    for region in image_regions:
        _image_x0, image_top, _image_x1, image_bottom = region
        x_overlap = _bbox_x_overlap_ratio(region, bbox)
        if x_overlap < 0.10:
            continue
        if image_bottom < note_y0 or image_top > note_y1:
            continue
        matches.append(region)

    deduped: list[tuple[float, float, float, float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for region in sorted(matches, key=lambda item: (item[1], item[0])):
        key = tuple(round(float(value), 2) for value in region)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(region)
    return deduped


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
    # note는 표와 같은 bbox라도 취급이 다르므로,
    # note-group geometry와 anchor를 기준으로 먼저 분리된 후보 집합을 만든다.
    candidate_rows: List[dict] = []
    image_regions = _candidate_image_regions_for_notes(page)
    note_group_candidates = _note_group_region_candidates(page, image_regions=image_regions)
    for bbox in note_group_candidates:
        split_candidates = _split_note_rows_by_anchors(page, bbox, image_regions=image_regions)
        if split_candidates:
            # 한 note group 안에 anchor가 여러 개면 실제로는 독립된 note 블록일 가능성이 높다.
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
