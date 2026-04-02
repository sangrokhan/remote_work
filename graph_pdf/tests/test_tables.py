from __future__ import annotations

import unittest
from types import SimpleNamespace

from extractor.notes import _collect_note_candidates, _note_body_text, _note_group_region_candidates
from extractor.shared import _merge_horizontal_band_segments, _merge_vertical_band_segments
from extractor.tables import (
    _is_black_fill_rect,
    _build_column_bands,
    _collapse_empty_columns,
    _build_grid_rows_from_black_lines,
    _continuation_regions_should_merge,
    _extract_tables_from_crop,
    _extract_tables,
    _dedupe_redundant_rectangles,
    _is_black_line_segment,
    _is_horizontal_separator_rect,
    _normalize_extracted_table,
    _to_rect_entry,
    _rows_from_payload_grid,
    _is_vertical_separator_rect,
    _split_repeated_header,
    _should_try_table_continuation_merge,
    _table_regions,
    _table_rejection_reason,
    _table_text_from_rows,
    _header_row_count,
)


class TableModuleTests(unittest.TestCase):
    def test_build_column_bands_collapses_nearby_visible_vertical_edges_into_single_boundary(self) -> None:
        crop_bbox = (71.3, 645.98, 525.57, 666.54)
        row_bands = [(71.3, 647.98, 525.57, 664.54)]
        vertical_segments = [
            {"x0": 77.42, "x1": 77.42, "top": 647.98, "bottom": 664.54},
            {"x0": 208.36, "x1": 208.36, "top": 647.98, "bottom": 664.54},
            {"x0": 214.00, "x1": 214.00, "top": 647.98, "bottom": 664.54},
            {"x0": 219.17, "x1": 219.17, "top": 647.98, "bottom": 664.54},
            {"x0": 328.87, "x1": 328.87, "top": 647.98, "bottom": 664.54},
            {"x0": 334.56, "x1": 334.56, "top": 647.98, "bottom": 664.54},
            {"x0": 339.67, "x1": 339.67, "top": 647.98, "bottom": 664.54},
            {"x0": 520.17, "x1": 520.17, "top": 647.98, "bottom": 664.54},
        ]

        column_lines, column_bands, column_error = _build_column_bands(crop_bbox, row_bands, vertical_segments)

        self.assertIsNone(column_error)
        self.assertEqual(2, len(column_lines))
        self.assertTrue(212.0 <= column_lines[0] <= 217.0)
        self.assertTrue(333.0 <= column_lines[1] <= 337.0)
        self.assertEqual(3, len(column_bands))

    def test_is_black_line_segment_accepts_visible_black_line(self) -> None:
        self.assertTrue(
            _is_black_line_segment(
                {
                    "object_type": "line",
                    "stroking_color": 0.0,
                    "linewidth": 1.0,
                }
            )
        )

    def test_is_black_line_segment_accepts_visible_black_rect_edge(self) -> None:
        self.assertTrue(
            _is_black_line_segment(
                {
                    "object_type": "rect_edge",
                    "stroking_color": 0.0,
                    "stroke": True,
                    "linewidth": 1.0,
                }
            )
        )

    def test_black_fill_rect_separator_threshold_uses_point_five(self) -> None:
        self.assertTrue(
            _is_black_fill_rect(
                {
                    "fill": True,
                    "non_stroking_color": 0.0,
                    "width": 0.48,
                    "height": 16.08,
                }
            )
        )
        self.assertTrue(
            _is_vertical_separator_rect(
                {
                    "fill": True,
                    "non_stroking_color": 0.0,
                    "width": 0.48,
                    "height": 16.08,
                }
            )
        )
        self.assertTrue(
            _is_horizontal_separator_rect(
                {
                    "fill": True,
                    "non_stroking_color": 0.0,
                    "width": 141.5,
                    "height": 0.48,
                }
            )
        )
        self.assertFalse(
            _is_horizontal_separator_rect(
                {
                    "fill": True,
                    "non_stroking_color": 0.0,
                    "width": 454.27,
                    "height": 0.96,
                }
            )
        )

    def test_build_grid_rows_from_black_lines_reconstructs_rows_and_columns_from_black_geometry(self) -> None:
        crop_bbox = (40.0, 100.0, 240.0, 190.0)
        crop = SimpleNamespace(
            extract_words=lambda **kwargs: [
                {"text": "Header Alpha", "x0": 52.0, "x1": 108.0, "top": 118.0, "bottom": 128.0},
                {"text": "Header Beta", "x0": 126.0, "x1": 174.0, "top": 118.0, "bottom": 128.0},
                {"text": "Header Gamma", "x0": 186.0, "x1": 228.0, "top": 118.0, "bottom": 128.0},
                {"text": "v1", "x0": 56.0, "x1": 68.0, "top": 148.0, "bottom": 158.0},
                {"text": "v2", "x0": 128.0, "x1": 140.0, "top": 148.0, "bottom": 158.0},
                {"text": "v3", "x0": 188.0, "x1": 200.0, "top": 148.0, "bottom": 158.0},
            ]
        )
        page = SimpleNamespace(
            horizontal_edges=[
                {"x0": 40.0, "x1": 240.0, "top": 110.0, "bottom": 110.0, "stroking_color": 0.0},
                {"x0": 40.0, "x1": 240.0, "top": 140.0, "bottom": 140.0, "stroking_color": (0.0, 0.0, 0.0)},
                {"x0": 40.0, "x1": 240.0, "top": 170.0, "bottom": 170.0, "stroking_color": 0.1},
                {"x0": 40.0, "x1": 240.0, "top": 125.0, "bottom": 125.0, "stroking_color": (0.4, 0.4, 0.4)},
            ],
            vertical_edges=[
                {"x0": 120.0, "x1": 120.0, "top": 110.0, "bottom": 170.0, "stroking_color": 0.0},
                {"x0": 180.0, "x1": 180.0, "top": 110.0, "bottom": 170.0, "stroking_color": 0.0},
                {"x0": 150.0, "x1": 150.0, "top": 110.0, "bottom": 170.0, "stroking_color": (0.0, 0.0, 1.0)},
            ],
            filter=lambda fn: page,
            crop=lambda bbox: crop,
        )

        rows, row_lines, column_lines, row_bands, column_bands, debug = _build_grid_rows_from_black_lines(
            page,
            crop_bbox,
        )

        self.assertEqual(
            [
                ["Header Alpha", "Header Beta", "Header Gamma"],
                ["v1", "v2", "v3"],
            ],
            rows,
        )
        self.assertEqual([110.0, 140.0, 170.0], row_lines)
        self.assertEqual([120.0, 180.0], column_lines)
        self.assertEqual(
            [
                (40.0, 110.0, 240.0, 140.0),
                (40.0, 140.0, 240.0, 170.0),
            ],
            row_bands,
        )
        self.assertEqual(
            [
                (40.0, 100.0, 120.0, 190.0),
                (120.0, 100.0, 180.0, 190.0),
                (180.0, 100.0, 240.0, 190.0),
            ],
            column_bands,
        )
        self.assertEqual(
            {
                "stage": "line_assignment",
                "raw_row_lines": 3,
                "raw_vertical_lines": 2,
                "raw_payload_count": 6,
                "assigned_payload_count": 6,
                "ambiguous_payload_count": 0,
                "unassigned_payload_count": 0,
                "row_count": 2,
                "column_count": 3,
                "column_error": None,
                "candidate_rows": 0,
                "merged_rows": 0,
                "ignored_rows": 0,
            },
            debug,
        )

    def test_build_grid_rows_from_black_lines_uses_crop_bbox_as_full_width_band_for_row_only_tables(self) -> None:
        crop_bbox = (40.0, 100.0, 240.0, 190.0)
        crop = SimpleNamespace(
            extract_words=lambda **kwargs: [
                {"text": "Header", "x0": 86.0, "x1": 126.0, "top": 118.0, "bottom": 128.0},
                {"text": "Value", "x0": 92.0, "x1": 124.0, "top": 148.0, "bottom": 158.0},
            ]
        )
        page = SimpleNamespace(
            horizontal_edges=[
                {"x0": 40.0, "x1": 240.0, "top": 110.0, "bottom": 110.0, "stroking_color": 0.0},
                {"x0": 40.0, "x1": 240.0, "top": 140.0, "bottom": 140.0, "stroking_color": 0.0},
                {"x0": 40.0, "x1": 240.0, "top": 170.0, "bottom": 170.0, "stroking_color": 0.0},
            ],
            vertical_edges=[
                {"x0": 40.0, "x1": 40.0, "top": 110.0, "bottom": 170.0, "stroking_color": 0.0},
                {"x0": 240.0, "x1": 240.0, "top": 110.0, "bottom": 170.0, "stroking_color": 0.0},
                {"x0": 150.0, "x1": 150.0, "top": 110.0, "bottom": 170.0, "stroking_color": (0.4, 0.4, 0.4)},
            ],
            filter=lambda fn: page,
            crop=lambda bbox: crop,
        )

        rows, row_lines, column_lines, row_bands, column_bands, debug = _build_grid_rows_from_black_lines(
            page,
            crop_bbox,
        )

        self.assertEqual([["Header"], ["Value"]], rows)
        self.assertEqual([110.0, 140.0, 170.0], row_lines)
        self.assertEqual([], column_lines)
        self.assertEqual(
            [
                (40.0, 110.0, 240.0, 140.0),
                (40.0, 140.0, 240.0, 170.0),
            ],
            row_bands,
        )
        self.assertEqual([(40.0, 100.0, 240.0, 190.0)], column_bands)
        self.assertEqual(
            {
                "stage": "line_assignment",
                "raw_row_lines": 3,
                "raw_vertical_lines": 2,
                "raw_payload_count": 2,
                "assigned_payload_count": 2,
                "ambiguous_payload_count": 0,
                "unassigned_payload_count": 0,
                "row_count": 2,
                "column_count": 1,
                "column_error": "no_internal_vertical_lines",
                "candidate_rows": 0,
                "merged_rows": 0,
                "ignored_rows": 0,
            },
            debug,
        )

    def test_table_regions_accept_two_horizontal_lines_with_vertical_connections(self) -> None:
        page = SimpleNamespace(
            rects=[
                {"x0": 72.0, "x1": 213.5, "top": 648.0, "bottom": 648.48, "width": 141.5, "height": 0.48, "fill": True, "non_stroking_color": 0.0},
                {"x0": 214.0, "x1": 334.0, "top": 648.0, "bottom": 648.48, "width": 120.0, "height": 0.48, "fill": True, "non_stroking_color": 0.0},
                {"x0": 334.5, "x1": 525.0, "top": 648.0, "bottom": 648.48, "width": 190.5, "height": 0.48, "fill": True, "non_stroking_color": 0.0},
                {"x0": 72.0, "x1": 213.5, "top": 664.5, "bottom": 664.98, "width": 141.5, "height": 0.48, "fill": True, "non_stroking_color": 0.0},
                {"x0": 214.0, "x1": 334.0, "top": 664.5, "bottom": 664.98, "width": 120.0, "height": 0.48, "fill": True, "non_stroking_color": 0.0},
                {"x0": 334.5, "x1": 525.0, "top": 664.5, "bottom": 664.98, "width": 190.5, "height": 0.48, "fill": True, "non_stroking_color": 0.0},
                {"x0": 213.53, "x1": 214.01, "top": 648.0, "bottom": 665.0, "width": 0.48, "height": 17.0, "fill": True, "non_stroking_color": 0.0},
                {"x0": 334.03, "x1": 334.51, "top": 648.0, "bottom": 665.0, "width": 0.48, "height": 17.0, "fill": True, "non_stroking_color": 0.0},
            ],
            lines=[],
            horizontal_edges=[],
            vertical_edges=[],
            width=595.32,
            height=841.92,
            chars=[],
        )

        regions = _table_regions(page)

        self.assertEqual(1, len(regions))
        x0, x1, lines = regions[0]
        self.assertEqual((72.0, 525.0), (round(x0, 1), round(x1, 1)))
        self.assertEqual(2, len(lines))

    def test_table_rejection_reason_allows_single_column_and_sparse_tables(self) -> None:
        self.assertIsNone(_table_rejection_reason([["Status"], ["Ready"]]))
        self.assertIsNone(_table_rejection_reason([["Status", "", ""], ["Ready", "", ""]]))

    def test_table_rejection_reason_no_longer_rejects_large_row_count_by_size_only(self) -> None:
        table = [["Value"] for _ in range(81)]
        self.assertIsNone(_table_rejection_reason(table))

    def test_continuation_regions_require_shared_axes_and_empty_gap(self) -> None:
        self.assertTrue(
            _continuation_regions_should_merge(
                prev_bbox=(100.0, 620.0, 400.0, 705.0),
                curr_bbox=(102.0, 88.0, 402.0, 190.0),
                prev_axes=[180.0, 280.0],
                curr_axes=[180.4, 279.8],
                body_top=72.0,
                body_bottom=722.0,
                has_gap_text=[],
            )
        )
        self.assertFalse(
            _continuation_regions_should_merge(
                prev_bbox=(100.0, 520.0, 400.0, 620.0),
                curr_bbox=(102.0, 180.0, 402.0, 280.0),
                prev_axes=[180.0, 280.0],
                curr_axes=[180.4, 279.8],
                body_top=72.0,
                body_bottom=722.0,
                has_gap_text=[(40.0, 730.0, 80.0, 742.0)],
            )
        )

    def test_same_page_tables_do_not_trigger_continuation_merge(self) -> None:
        self.assertFalse(_should_try_table_continuation_merge(pending_page=2, current_page=2))
        self.assertTrue(_should_try_table_continuation_merge(pending_page=2, current_page=3))

    def test_table_regions_split_or_merge_by_shared_vertical_edges(self) -> None:
        split_page = SimpleNamespace(
            width=600.0,
            height=800.0,
            horizontal_edges=[
                {"x0": 100.0, "x1": 400.0, "top": 100.0, "bottom": 100.0},
                {"x0": 100.0, "x1": 400.0, "top": 130.0, "bottom": 130.0},
                {"x0": 100.0, "x1": 400.0, "top": 160.0, "bottom": 160.0},
                {"x0": 420.0, "x1": 560.0, "top": 250.0, "bottom": 250.0},
                {"x0": 420.0, "x1": 560.0, "top": 280.0, "bottom": 280.0},
                {"x0": 420.0, "x1": 560.0, "top": 310.0, "bottom": 310.0},
            ],
            vertical_edges=[
                {"x0": 180.0, "x1": 180.0, "top": 100.0, "bottom": 160.0},
                {"x0": 280.0, "x1": 280.0, "top": 100.0, "bottom": 160.0},
                {"x0": 470.0, "x1": 470.0, "top": 250.0, "bottom": 310.0},
                {"x0": 520.0, "x1": 520.0, "top": 250.0, "bottom": 310.0},
            ],
        )
        merged_page = SimpleNamespace(
            width=600.0,
            height=800.0,
            horizontal_edges=[
                {"x0": 100.0, "x1": 400.0, "top": 100.0, "bottom": 100.0},
                {"x0": 100.0, "x1": 400.0, "top": 130.0, "bottom": 130.0},
                {"x0": 100.0, "x1": 400.0, "top": 160.0, "bottom": 160.0},
                {"x0": 102.0, "x1": 402.0, "top": 250.0, "bottom": 250.0},
                {"x0": 102.0, "x1": 402.0, "top": 280.0, "bottom": 280.0},
                {"x0": 102.0, "x1": 402.0, "top": 310.0, "bottom": 310.0},
            ],
            vertical_edges=[
                {"x0": 180.0, "x1": 180.0, "top": 100.0, "bottom": 310.0},
                {"x0": 280.0, "x1": 280.0, "top": 100.0, "bottom": 310.0},
            ],
        )

        self.assertEqual(2, len(_table_regions(split_page)))
        self.assertEqual(1, len(_table_regions(merged_page)))

    def test_merge_band_segments_merge_overlaps_but_keep_large_gaps(self) -> None:
        horizontal = [
            {"x0": 10.0, "x1": 50.0, "top": 100.0, "bottom": 100.6},
            {"x0": 50.4, "x1": 80.0, "top": 99.6, "bottom": 100.0},
            {"x0": 90.5, "x1": 110.0, "top": 100.0, "bottom": 100.0},
        ]
        vertical = [
            {"x0": 199.6, "x1": 200.0, "top": 10.0, "bottom": 50.0},
            {"x0": 200.0, "x1": 200.6, "top": 49.5, "bottom": 80.0},
            {"x0": 200.0, "x1": 200.0, "top": 90.2, "bottom": 110.0},
        ]

        horizontal_merged = _merge_horizontal_band_segments(horizontal, tolerance=1.0)
        vertical_merged = _merge_vertical_band_segments(vertical, tolerance=1.0)

        self.assertEqual(
            [
                {"x0": 10.0, "x1": 80.0, "top": 99.6, "bottom": 100.6},
                {"x0": 90.5, "x1": 110.0, "top": 100.0, "bottom": 100.0},
            ],
            [{key: segment[key] for key in ("x0", "x1", "top", "bottom")} for segment in horizontal_merged],
        )
        self.assertEqual(
            [
                {"x0": 199.6, "x1": 200.6, "top": 10.0, "bottom": 80.0},
                {"x0": 200.0, "x1": 200.0, "top": 90.2, "bottom": 110.0},
            ],
            [{key: segment[key] for key in ("x0", "x1", "top", "bottom")} for segment in vertical_merged],
        )

    def test_to_rect_entry_handles_merge_and_dict_inputs(self) -> None:
        rect_dict = {
            "x0": 1.0,
            "y0": 2.0,
            "x1": 9.0,
            "y1": 10.0,
            "fill": False,
            "stroke": True,
            "non_stroking_color": (0.1, 0.1, 0.1),
        }
        rect_tuple = (1.0, 2.0, 9.0, 10.0)
        self.assertEqual(
            {
                "x0": 1.0,
                "top": 2.0,
                "x1": 9.0,
                "bottom": 10.0,
                "fill": False,
                "stroke": True,
                "non_stroking_color": (0.1, 0.1, 0.1),
                "stroking_color": None,
            },
            _to_rect_entry(rect_dict),
        )
        self.assertEqual(
            {
                "x0": 1.0,
                "top": 2.0,
                "x1": 9.0,
                "bottom": 10.0,
                "fill": True,
                "stroke": False,
                "non_stroking_color": None,
                "stroking_color": None,
            },
            _to_rect_entry(rect_tuple),
        )

    def test_dedupe_redundant_rectangles_removes_nested_white_rect(self) -> None:
        rects = [
            (0.0, 0.0, 100.0, 100.0),
            {
                "x0": 10.0,
                "top": 10.0,
                "x1": 90.0,
                "bottom": 90.0,
                "fill": True,
                "stroke": False,
                "non_stroking_color": 1.0,
                "stroking_color": None,
            },
        ]
        deduped = _dedupe_redundant_rectangles(rects)
        self.assertEqual(1, len(deduped))
        self.assertEqual(
            {
                "x0": 0.0,
                "top": 0.0,
                "x1": 100.0,
                "bottom": 100.0,
                "fill": True,
                "stroke": False,
                "non_stroking_color": None,
                "stroking_color": None,
            },
            deduped[0],
        )

    def test_extract_tables_skips_page_wide_fallback_by_default(self) -> None:
        page = SimpleNamespace(
            width=600.0,
            height=800.0,
            horizontal_edges=[],
            vertical_edges=[],
            extract_tables=lambda **kwargs: [[["A", "B"], ["1", "2"]]],
            filter=lambda fn: page,
        )
        self.assertEqual([], _extract_tables(page))

    def test_extract_tables_from_crop_falls_back_to_text_lines_when_no_table_detected(self) -> None:
        crop = SimpleNamespace(
            extract_tables=lambda **kwargs: [],
            extract_words=lambda **kwargs: [
                {"text": "Escalation", "x0": 50.0, "x1": 110.0, "top": 100.0, "bottom": 108.0},
                {"text": "lane", "x0": 114.0, "x1": 148.0, "top": 100.0, "bottom": 108.0},
                {"text": "summary", "x0": 152.0, "x1": 210.0, "top": 100.0, "bottom": 108.0},
                {"text": "Owner", "x0": 52.0, "x1": 84.0, "top": 115.0, "bottom": 123.0},
                {"text": "approved", "x0": 88.0, "x1": 144.0, "top": 115.0, "bottom": 123.0},
            ],
        )
        page = SimpleNamespace(
            width=600.0,
            height=800.0,
            vertical_edges=[],
            crop=lambda bbox: crop,
            filter=lambda fn: page,
        )

        result = _extract_tables_from_crop(
            page,
            crop_bbox=(40.0, 96.0, 240.0, 132.0),
            fallback_to_text_rows=True,
        )

        self.assertEqual(
            [([["Escalation lane summary"], ["Owner approved"]], (40.0, 96.0, 240.0, 132.0))],
            result,
        )

    def test_extract_tables_from_crop_reconstructs_black_line_grid(self) -> None:
        words = [
            {"text": "ColA1", "x0": 12.0, "x1": 36.0, "top": 24.0, "bottom": 30.0},
            {"text": "ColB1", "x0": 86.0, "x1": 112.0, "top": 24.0, "bottom": 30.0},
            {"text": "ColC1", "x0": 152.0, "x1": 182.0, "top": 24.0, "bottom": 30.0},
            {"text": "ColA2", "x0": 12.0, "x1": 36.0, "top": 72.0, "bottom": 78.0},
            {"text": "ColB2", "x0": 86.0, "x1": 112.0, "top": 72.0, "bottom": 78.0},
            {"text": "ColC2", "x0": 152.0, "x1": 182.0, "top": 72.0, "bottom": 78.0},
        ]
        page = SimpleNamespace(
            width=220.0,
            height=120.0,
            horizontal_edges=[
                {"x0": 10.0, "x1": 210.0, "top": 20.0, "bottom": 20.0, "stroking_color": (0.0, 0.0, 0.0)},
                {"x0": 10.0, "x1": 210.0, "top": 50.0, "bottom": 50.0, "stroking_color": (0.0, 0.0, 0.0)},
                {"x0": 10.0, "x1": 210.0, "top": 100.0, "bottom": 100.0, "stroking_color": (0.0, 0.0, 0.0)},
            ],
            vertical_edges=[
                {"x0": 10.0, "x1": 10.0, "top": 20.0, "bottom": 100.0, "stroking_color": (0.0, 0.0, 0.0)},
                {"x0": 80.0, "x1": 80.0, "top": 20.0, "bottom": 100.0, "stroking_color": (0.0, 0.0, 0.0)},
                {"x0": 140.0, "x1": 140.0, "top": 20.0, "bottom": 100.0, "stroking_color": (0.0, 0.0, 0.0)},
                {"x0": 210.0, "x1": 210.0, "top": 20.0, "bottom": 100.0, "stroking_color": (0.0, 0.0, 0.0)},
            ],
            filter=lambda fn: page,
            crop=lambda bbox: SimpleNamespace(
                extract_words=lambda **kwargs: words,
            ),
        )
        debug: list[dict] = []
        result = _extract_tables_from_crop(
            page,
            crop_bbox=(10.0, 20.0, 210.0, 100.0),
            fallback_to_text_rows=False,
            strategy_debug=debug,
            strategy_source="test",
            strategy_source_name="grid",
        )

        self.assertEqual(
            [(
                [
                    ["ColA1", "ColB1", "ColC1"],
                    ["ColA2", "ColB2", "ColC2"],
                ],
                (10.0, 20.0, 210.0, 100.0),
            )],
            result,
        )
        self.assertEqual(1, len(debug))
        self.assertEqual("success", debug[0].get("status"))
        self.assertEqual(2, debug[0].get("column_line_count"))
        self.assertEqual(2, debug[0].get("row_band_count"))

    def test_extract_tables_from_crop_reconstructs_row_only_when_no_internal_columns(self) -> None:
        words = [
            {"text": "KeyOne", "x0": 12.0, "x1": 44.0, "top": 24.0, "bottom": 30.0},
            {"text": "ValueOne", "x0": 12.0, "x1": 80.0, "top": 24.0, "bottom": 30.0},
            {"text": "KeyTwo", "x0": 12.0, "x1": 44.0, "top": 72.0, "bottom": 78.0},
            {"text": "ValueTwo", "x0": 12.0, "x1": 88.0, "top": 72.0, "bottom": 78.0},
        ]
        page = SimpleNamespace(
            width=220.0,
            height=120.0,
            horizontal_edges=[
                {"x0": 10.0, "x1": 210.0, "top": 20.0, "bottom": 20.0, "stroking_color": (0.0, 0.0, 0.0)},
                {"x0": 10.0, "x1": 210.0, "top": 50.0, "bottom": 50.0, "stroking_color": (0.0, 0.0, 0.0)},
                {"x0": 10.0, "x1": 210.0, "top": 100.0, "bottom": 100.0, "stroking_color": (0.0, 0.0, 0.0)},
            ],
            vertical_edges=[
                {"x0": 10.0, "x1": 10.0, "top": 20.0, "bottom": 100.0, "stroking_color": (0.0, 0.0, 0.0)},
                {"x0": 210.0, "x1": 210.0, "top": 20.0, "bottom": 100.0, "stroking_color": (0.0, 0.0, 0.0)},
            ],
            filter=lambda fn: page,
            crop=lambda bbox: SimpleNamespace(
                extract_words=lambda **kwargs: words,
            ),
        )
        debug: list[dict] = []
        result = _extract_tables_from_crop(
            page,
            crop_bbox=(10.0, 20.0, 210.0, 100.0),
            fallback_to_text_rows=False,
            strategy_debug=debug,
            strategy_source="test",
            strategy_source_name="row-only",
        )

        self.assertEqual(
            [(
                [["KeyOne ValueOne"], ["KeyTwo ValueTwo"]],
                (10.0, 20.0, 210.0, 100.0),
            )],
            result,
        )
        self.assertEqual(1, len(debug))
        self.assertEqual(0, debug[0].get("column_line_count"))
        self.assertEqual(1, debug[0].get("column_band_count"))

    def test_extract_tables_from_crop_records_failure_metadata_when_grid_not_buildable(self) -> None:
        page = SimpleNamespace(
            width=220.0,
            height=120.0,
            horizontal_edges=[
                {"x0": 10.0, "x1": 210.0, "top": 20.0, "bottom": 20.0, "stroking_color": (0.5, 0.5, 0.5)},
            ],
            vertical_edges=[],
            filter=lambda fn: page,
            crop=lambda bbox: SimpleNamespace(
                extract_words=lambda **kwargs: [
                    {"text": "Loose", "x0": 12.0, "x1": 40.0, "top": 25.0, "bottom": 31.0},
                ]
            ),
        )
        debug: list[dict] = []
        result = _extract_tables_from_crop(
            page,
            crop_bbox=(10.0, 20.0, 210.0, 100.0),
            fallback_to_text_rows=False,
            strategy_debug=debug,
            strategy_source="test",
            strategy_source_name="fail",
        )

        self.assertEqual([], result)
        self.assertEqual(1, len(debug))
        self.assertEqual("failed", debug[0].get("status"))
        self.assertEqual("insufficient_row_lines", debug[0].get("failure_reason"))

    def test_extract_tables_uses_text_fallback_for_region_candidates(self) -> None:
        from unittest.mock import patch

        page = SimpleNamespace(
            width=600.0,
            height=800.0,
            vertical_edges=[],
            horizontal_edges=[],
            filter=lambda fn: page,
            extract_tables=lambda **kwargs: [],
            crop=lambda bbox: SimpleNamespace(extract_words=lambda **kwargs: []),
        )

        with patch(
            "extractor.tables._table_regions",
            return_value=[(40.0, 540.0, [{"top": 121.0}, {"top": 148.0}])],
        ) as table_regions, patch(
            "extractor.tables._extract_tables_from_crop",
            return_value=[],
        ) as extract_from_crop:
            _extract_tables(page)

        expected_bbox = (40.0, 119.0, 540.0, 150.0)
        table_regions.assert_called_once_with(page, excluded_bboxes=None)
        extract_from_crop.assert_called_once_with(
            page,
            expected_bbox,
            fallback_to_text_rows=False,
            strategy_debug=None,
            strategy_source="table_region",
            strategy_source_name="table_region#0",
        )

    def test_collect_note_candidates_returns_empty_without_note_groups(self) -> None:
        from unittest.mock import patch

        page = SimpleNamespace(width=600.0, height=800.0)

        with patch(
            "extractor.notes._note_group_region_candidates",
            return_value=[],
        ):
            candidates = _collect_note_candidates(page)

        self.assertEqual([], candidates)

    def test_note_group_region_candidates_detect_blue_group_from_blue_lines_and_anchors(self) -> None:
        from unittest.mock import patch

        page = SimpleNamespace(
            horizontal_edges=[
                {"x0": 40.0, "x1": 540.0, "top": 120.0, "bottom": 120.0, "stroking_color": (0.0, 0.0, 1.0)},
                {"x0": 40.0, "x1": 540.0, "top": 220.0, "bottom": 220.0, "stroking_color": (0.0, 0.0, 1.0)},
            ],
            rects=[],
            height=800.0,
        )

        with patch(
            "extractor.notes._detect_body_bounds",
            return_value=(80.0, 760.0),
        ):
            groups = _note_group_region_candidates(
                page,
                image_regions=[
                    (50.0, 122.0, 60.0, 132.0),
                    (50.0, 168.0, 60.0, 178.0),
                ],
            )

        self.assertEqual([(40.0, 120.0, 540.0, 220.0)], groups)

    def test_table_text_from_rows_collapses_two_header_rows_into_single_markdown_header(self) -> None:
        rows = [
            ["Stage", "Team", "Notes"],
            ["Group", "Function", "Deliverable"],
            ["Phase A", "Discovery", "Kickoff scope lock"],
        ]

        markdown = _table_text_from_rows(rows)

        self.assertIn("| Stage<br>Group | Team<br>Function | Notes<br>Deliverable |", markdown)
        self.assertIn("| Phase A", markdown)
        self.assertIn("Kickoff scope lock", markdown)

    def test_header_row_count_does_not_promote_first_data_row_to_header(self) -> None:
        rows = [
            ["Column A", "Column B", "Column C"],
            ["Path A / Left", "Node 1", "Node 2"],
            ["Path B / Right", "Node 3", "Node 4"],
        ]

        self.assertEqual(1, _header_row_count(rows))

    def test_note_body_text_uses_first_non_empty_cell(self) -> None:
        rows = [["", "", "Escalation lane summary"], ["", "", "Owner confirmed"]]
        self.assertEqual(
            "Note: Escalation lane summary Owner confirmed",
            _note_body_text(rows),
        )

    def test_note_body_text_inserts_missing_space_before_open_quote(self) -> None:
        rows = [["The counter of A and‘B’ is provided."]]
        self.assertEqual(
            "Note: The counter of A and ‘B’ is provided.",
            _note_body_text(rows),
        )

    def test_note_body_text_detects_single_row_multiline_cell(self) -> None:
        rows = [
            [
                "Escalation lane summary\nOwner confirmed for regional review and exception routing.\n"
                "Backup approver stays on the same visual box and must not become a second table row.",
            ],
        ]
        self.assertEqual(
            "Note: Escalation lane summary Owner confirmed for regional review and exception routing. Backup approver stays on the same visual box and must not become a second table row.",
            _note_body_text(rows),
        )

    def test_note_body_text_does_not_duplicate_existing_prefix(self) -> None:
        rows = [["Note: Escalation lane summary"], ["Owner confirmed"]]
        self.assertEqual(
            "Note: Escalation lane summary Owner confirmed",
            _note_body_text(rows),
        )

    def test_table_text_from_rows_preserves_single_column_header_when_short(self) -> None:
        rows = [
            ["Status"],
            ["Ready"],
        ]

        markdown = _table_text_from_rows(rows)

        self.assertIn("| Status |", markdown)
        self.assertIn("Status", markdown)
        self.assertIn("Ready", markdown)
        self.assertNotIn("Column 1", markdown)

    def test_table_text_from_rows_does_not_generate_synthetic_header_for_single_row(self) -> None:
        rows = [["Status only"]]

        markdown = _table_text_from_rows(rows)

        self.assertIn("| Status only |", markdown)
        self.assertNotIn("Column 1", markdown)

    def test_split_repeated_header_removes_repeated_two_row_header_block(self) -> None:
        prev_rows = [
            ["Stage", "Team", "Notes"],
            ["Group", "Function", "Deliverable"],
            ["Phase A", "Discovery", "Kickoff scope lock"],
        ]
        curr_rows = [
            ["Stage", "Team", "Notes"],
            ["Group", "Function", "Deliverable"],
            ["", "Design", "UX skeleton review"],
        ]

        self.assertEqual(
            [["", "Design", "UX skeleton review"]],
            _split_repeated_header(prev_rows, curr_rows),
        )

    def test_split_repeated_header_trims_carried_body_text_from_repeated_second_header_row(self) -> None:
        prev_rows = [
            ["Stage", "Team", "Notes"],
            ["Group", "Function", "Deliverable"],
            ["Phase C", "Documentation", "Publish handoff pack"],
            ["", "Legal", "Terms and compliance checks"],
        ]
        curr_rows = [
            ["Stage", "Team", "Notes"],
            ["Group", "Function", "Deliverable\n- consent language review\n- archive plan"],
            ["", "Accessibility", "Review deep pass"],
        ]

        self.assertEqual(
            [
                ["", "", "- consent language review\n- archive plan"],
                ["", "Accessibility", "Review deep pass"],
            ],
            _split_repeated_header(prev_rows, curr_rows),
        )
