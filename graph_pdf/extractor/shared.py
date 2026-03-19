from __future__ import annotations

import math
import re
from typing import List, Sequence, Tuple

TableRows = List[List[str]]
TableChunk = Tuple[TableRows, Tuple[float, float, float, float]]

WATERMARK_ROTATION_MIN_DEGREES = 53.0
WATERMARK_ROTATION_MAX_DEGREES = 57.0
WATERMARK_GRAY_MIN = 0.88
WATERMARK_GRAY_MAX = 0.96
WATERMARK_GRAY_NEUTRAL_TOLERANCE = 0.03

BULLET_PREFIX_RE = re.compile(
    r"^(?:[-*•●○◦◯▪▫■□◆◇◈◊‣∙◉]|[0-9]+(?:[-.][0-9]+)*[.)]|o|\?|\uFFFD)\s+"
)


def _parse_pages_spec(spec: str) -> List[int]:
    # Accept `1,3-5,8` style CLI input and normalize it into a sorted unique page list.
    values = set()
    for part in str(spec or "").split(","):
        token = part.strip()
        if not token:
            raise ValueError("empty page token in --pages")
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError(f"invalid page range: {token}")
            start = int(start_text)
            end = int(end_text)
            if start < 1 or end < 1 or start > end:
                raise ValueError(f"invalid page range: {token}")
            values.update(range(start, end + 1))
            continue
        if not token.isdigit():
            raise ValueError(f"invalid page number: {token}")
        page_no = int(token)
        if page_no < 1:
            raise ValueError(f"invalid page number: {token}")
        values.add(page_no)
    if not values:
        raise ValueError("no pages selected")
    return sorted(values)


def _normalize_text(text: str) -> str:
    # Most table/body comparisons only care about semantic text, not spacing differences.
    return re.sub(r"\s+", " ", text or "").strip()


def _char_rotation_degrees(char: dict) -> float:
    # pdfplumber exposes the text matrix directly, so rotation comes from the matrix rather than a dedicated field.
    matrix = char.get("matrix")
    if not isinstance(matrix, tuple) or len(matrix) < 2:
        return 0.0
    return math.degrees(math.atan2(float(matrix[1]), float(matrix[0])))


def _merge_numeric_positions(values: Sequence[float], tolerance: float = 1.0) -> List[float]:
    # Nearby edge coordinates often differ by sub-pixel noise; collapse them before higher-level reasoning.
    merged: List[float] = []
    for value in sorted(float(v) for v in values):
        if not merged or abs(value - merged[-1]) > tolerance:
            merged.append(value)
            continue
        merged[-1] = (merged[-1] + value) / 2.0
    return merged


def _cluster_axis_values(values: Sequence[float], tolerance: float = 1.0) -> List[List[float]]:
    # Segment grouping starts by clustering nearly-identical coordinates on one axis.
    clusters: List[List[float]] = []
    for value in sorted(float(v) for v in values):
        if not clusters or abs(value - clusters[-1][-1]) > tolerance:
            clusters.append([value])
            continue
        clusters[-1].append(value)
    return clusters


def _round_segment(
    edge: dict,
    body_top: float | None = None,
    body_bottom: float | None = None,
) -> dict:
    # Debug payloads use rounded values so JSON stays readable and stable across runs.
    payload = {
        "x0": round(float(edge["x0"]), 2),
        "x1": round(float(edge["x1"]), 2),
        "top": round(float(edge["top"]), 2),
        "bottom": round(float(edge["bottom"]), 2),
    }
    if body_top is not None and body_bottom is not None:
        payload["in_body_bounds"] = (
            float(edge["bottom"]) > body_top and float(edge["top"]) < body_bottom
        )
    return payload


def _normalize_band_segments(
    segments: Sequence[dict],
    start_key: str,
    end_key: str,
    orth_min_key: str,
    orth_max_key: str,
    tolerance: float = 1.0,
) -> List[dict]:
    # Horizontal and vertical segment merging share the same overlap/containment rules.
    normalized: List[dict] = []
    ordered = sorted(
        (dict(edge) for edge in segments),
        key=lambda item: (float(item[start_key]), float(item[end_key])),
    )
    for edge in ordered:
        if not normalized:
            normalized.append(edge)
            continue

        current = normalized[-1]
        current_start = float(current[start_key])
        current_end = float(current[end_key])
        edge_start = float(edge[start_key])
        edge_end = float(edge[end_key])

        if edge_start >= current_start and edge_end <= current_end:
            current[orth_min_key] = min(float(current[orth_min_key]), float(edge[orth_min_key]))
            current[orth_max_key] = max(float(current[orth_max_key]), float(edge[orth_max_key]))
            continue

        if edge_start - current_end <= tolerance:
            current[end_key] = max(current_end, edge_end)
            current[orth_min_key] = min(float(current[orth_min_key]), float(edge[orth_min_key]))
            current[orth_max_key] = max(float(current[orth_max_key]), float(edge[orth_max_key]))
            continue

        normalized.append(edge)

    return normalized


def _merge_horizontal_band_segments(
    segments: Sequence[dict],
    tolerance: float = 1.0,
    body_top: float | None = None,
    body_bottom: float | None = None,
) -> List[dict]:
    # Horizontal lines are merged on x-range while preserving the full vertical span seen in source edges.
    merged = _normalize_band_segments(
        segments,
        start_key="x0",
        end_key="x1",
        orth_min_key="top",
        orth_max_key="bottom",
        tolerance=tolerance,
    )
    return [_round_segment(edge, body_top=body_top, body_bottom=body_bottom) for edge in merged]


def _merge_vertical_band_segments(
    segments: Sequence[dict],
    tolerance: float = 1.0,
    body_top: float | None = None,
    body_bottom: float | None = None,
) -> List[dict]:
    # Vertical lines are merged on y-range while preserving the full horizontal span seen in source edges.
    merged = _normalize_band_segments(
        segments,
        start_key="top",
        end_key="bottom",
        orth_min_key="x0",
        orth_max_key="x1",
        tolerance=tolerance,
    )
    return [_round_segment(edge, body_top=body_top, body_bottom=body_bottom) for edge in merged]


def _build_segment_groups(
    segments: Sequence[dict],
    axis_key: str,
    merge_fn,
    tolerance: float = 1.0,
    body_top: float | None = None,
    body_bottom: float | None = None,
) -> List[dict]:
    # Group raw edges by shared axis position first, then expose both original and merged segment views for debug.
    clusters = _cluster_axis_values([float(edge[axis_key]) for edge in segments], tolerance=tolerance)
    groups: List[dict] = []
    for cluster in clusters:
        axis = sum(cluster) / len(cluster)
        members = [
            edge
            for edge in segments
            if any(abs(float(edge[axis_key]) - value) <= tolerance for value in cluster)
        ]
        groups.append(
            {
                "axis": round(axis, 2),
                "segments": [
                    _round_segment(edge, body_top=body_top, body_bottom=body_bottom)
                    for edge in members
                ],
                "merged_segments": merge_fn(
                    members,
                    tolerance=tolerance,
                    body_top=body_top,
                    body_bottom=body_bottom,
                ),
            }
        )
    return groups


def _bboxes_intersect(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> bool:
    # Text/table/image exclusion logic only needs simple rectangle overlap checks.
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0
