from __future__ import annotations

from .debug import (
    _collect_page_edge_debug,
    _collect_rotated_text_debug,
    _collect_table_drawing_debug,
)
from .font_profile import profile_pdf_fonts
from .images import _extract_embedded_images
from .pipeline import extract_pdf_to_outputs
from .shared import (
    _char_rotation_degrees,
    _merge_horizontal_band_segments,
    _merge_numeric_positions,
    _merge_vertical_band_segments,
    _parse_pages_spec,
)
from .tables import (
    _append_output_table,
    _collapse_structural_triplet_columns,
    _continuation_regions_should_merge,
    _extract_tables,
    _merge_split_rows,
    _should_try_table_continuation_merge,
    _table_regions,
    _table_rejection_reason,
    _table_text_from_rows,
)
from .text import (
    _build_body_blocks,
    _detect_body_bounds,
    _extract_body_text,
    _extract_body_text_lines,
    _extract_body_word_lines,
    _is_gray_color,
    _is_non_watermark_obj,
    _normalize_cell_lines,
    _should_merge_paragraph_lines,
)

# Re-export the main entrypoint and selected helpers so tests and scripts can
# keep importing from `extractor` after the package split.
__all__ = [
    "extract_pdf_to_outputs",
    "profile_pdf_fonts",
    "_append_output_table",
    "_build_body_blocks",
    "_char_rotation_degrees",
    "_collapse_structural_triplet_columns",
    "_collect_page_edge_debug",
    "_collect_rotated_text_debug",
    "_collect_table_drawing_debug",
    "_continuation_regions_should_merge",
    "_detect_body_bounds",
    "_extract_body_text",
    "_extract_body_text_lines",
    "_extract_body_word_lines",
    "_extract_embedded_images",
    "_extract_tables",
    "_is_gray_color",
    "_is_non_watermark_obj",
    "_merge_horizontal_band_segments",
    "_merge_numeric_positions",
    "_merge_split_rows",
    "_merge_vertical_band_segments",
    "_normalize_cell_lines",
    "_parse_pages_spec",
    "_should_merge_paragraph_lines",
    "_should_try_table_continuation_merge",
    "_table_regions",
    "_table_rejection_reason",
    "_table_text_from_rows",
]
