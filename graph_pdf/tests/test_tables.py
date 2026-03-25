from __future__ import annotations

import unittest
from types import SimpleNamespace

from extractor.shared import _merge_horizontal_band_segments, _merge_vertical_band_segments
from extractor.tables import (
    _collapse_structural_triplet_columns,
    _continuation_regions_should_merge,
    _extract_tables_from_crop,
    _extract_tables,
    _dedupe_redundant_rectangles,
    _to_rect_entry,
    _merge_single_column_fragment_rows,
    _single_column_box_regions,
    _single_column_boxes_share_index,
    _split_repeated_header,
    _single_column_note_body_text,
    _should_try_table_continuation_merge,
    _table_regions,
    _looks_like_single_column_note,
    _table_rejection_reason,
    _table_text_from_rows,
)


class TableModuleTests(unittest.TestCase):
    def test_table_rejection_reason_allows_single_column_and_sparse_tables(self) -> None:
        self.assertIsNone(_table_rejection_reason([["Status"], ["Ready"]]))
        self.assertIsNone(_table_rejection_reason([["Status", "", ""], ["Ready", "", ""]]))

    def test_table_rejection_reason_no_longer_rejects_large_row_count_by_size_only(self) -> None:
        table = [["Value"] for _ in range(81)]
        self.assertIsNone(_table_rejection_reason(table))

    def test_collapse_structural_triplet_columns_removes_empty_side_columns(self) -> None:
        table = [
            ["", "Area", "", "", "Status", "", "", "Action", ""],
            ["", "Docs", "", "", "READY", "", "", "Finalize", ""],
            ["", "QA", "", "", "TODO", "", "", "Confirm", ""],
        ]
        self.assertEqual(
            [
                ["Area", "Status", "Action"],
                ["Docs", "READY", "Finalize"],
                ["QA", "TODO", "Confirm"],
            ],
            _collapse_structural_triplet_columns(table),
        )

    def test_collapse_structural_triplet_columns_keeps_non_empty_side_columns(self) -> None:
        table = [
            ["", "Area", "", "", "Status", "", "", "Action", ""],
            ["note", "Docs", "", "", "READY", "", "", "Finalize", ""],
        ]
        self.assertEqual(
            [["", "Area", "Status", "Action"], ["note", "Docs", "READY", "Finalize"]],
            _collapse_structural_triplet_columns(table),
        )

    def test_collapse_structural_triplet_columns_removes_empty_vertical_columns(self) -> None:
        table = [
            ["A", "", "B", ""],
            ["C", "", "D", ""],
            ["E", "", "F"],
        ]
        self.assertEqual(
            [["A", "B"], ["C", "D"], ["E", "F"]],
            _collapse_structural_triplet_columns(table),
        )

    def test_continuation_regions_require_shared_axes_and_empty_gap(self) -> None:
        self.assertTrue(
            _continuation_regions_should_merge(
                prev_bbox=(100.0, 620.0, 400.0, 705.0),
                curr_bbox=(102.0, 88.0, 402.0, 190.0),
                prev_axes=[180.0, 280.0],
                curr_axes=[180.4, 279.8],
                body_top=72.0,
                body_bottom=722.0,
                gap_text_boxes=[],
            )
        )
        self.assertFalse(
            _continuation_regions_should_merge(
                prev_bbox=(100.0, 620.0, 400.0, 705.0),
                curr_bbox=(102.0, 88.0, 402.0, 190.0),
                prev_axes=[180.0, 280.0],
                curr_axes=[180.4, 279.8],
                body_top=72.0,
                body_bottom=722.0,
                gap_text_boxes=[(40.0, 730.0, 80.0, 742.0)],
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
        ) as extract_from_crop, patch(
            "extractor.tables._single_column_box_region_candidates",
            return_value=[],
        ):
            _extract_tables(page)

        expected_bbox = (40.0, 119.0, 540.0, 150.0)
        table_regions.assert_called_once_with(page)
        extract_from_crop.assert_called_once_with(page, expected_bbox, fallback_to_text_rows=True)

    def test_extract_tables_filters_single_column_box_overlapping_detected_table(self) -> None:
        table_bbox = (40.0, 119.0, 540.0, 150.0)
        page = SimpleNamespace(
            width=600.0,
            height=800.0,
            rects=[
                {
                    "x0": 40.0,
                    "x1": 540.0,
                    "top": 120.0,
                    "bottom": 121.0,
                    "fill": True,
                    "stroke": False,
                    "non_stroking_color": (0.2, 0.5, 0.9),
                },
                {
                    "x0": 40.0,
                    "x1": 290.0,
                    "top": 121.0,
                    "bottom": 148.0,
                    "fill": True,
                    "stroke": False,
                    "non_stroking_color": 1,
                },
                {
                    "x0": 290.0,
                    "x1": 540.0,
                    "top": 121.0,
                    "bottom": 148.0,
                    "fill": True,
                    "stroke": False,
                    "non_stroking_color": 1,
                },
                {
                    "x0": 40.0,
                    "x1": 540.0,
                    "top": 148.0,
                    "bottom": 149.0,
                    "fill": True,
                    "stroke": False,
                    "non_stroking_color": (0.2, 0.5, 0.9),
                },
            ],
            horizontal_edges=[],
            vertical_edges=[],
            chars=[],
            filter=lambda fn: page,
            crop=lambda bbox: SimpleNamespace(
                extract_words=lambda **kwargs: [
                    {"text": "Escalation", "x0": 50.0, "x1": 100.0, "top": 126.0, "bottom": 134.0},
                    {"text": "lane", "x0": 104.0, "x1": 130.0, "top": 126.0, "bottom": 134.0},
                    {"text": "summary", "x0": 134.0, "x1": 190.0, "top": 126.0, "bottom": 134.0},
                ]
            ),
            extract_tables=lambda **kwargs: [],
        )

        from unittest.mock import patch

        with patch(
            "extractor.tables._table_regions",
            return_value=[(40.0, 540.0, [{"top": 121.0}, {"top": 148.0}])],
        ), patch(
            "extractor.tables._extract_tables_from_crop",
            return_value=[(
                [["Escalation lane summary"], ["Owner confirmed for regional review and exception routing."]],
                table_bbox,
            )],
        ), patch(
            "extractor.tables._single_column_box_region_candidates",
            return_value=[
                {"bbox": (40.0, 121.0, 540.0, 148.0), "is_white_content": False},
                {"bbox": (40.0, 148.2, 540.0, 173.0), "is_white_content": False},
            ],
        ), patch(
            "extractor.tables._extract_text_from_box_region",
            side_effect=[
                "Escalation lane summary",
                "Owner confirmed for regional review and exception routing.",
            ],
        ):
            tables = _extract_tables(page)

        self.assertEqual([], tables)

    def test_extract_tables_keeps_layout_candidate_if_not_note(self) -> None:
        table_bbox = (40.0, 119.0, 540.0, 150.0)
        page = SimpleNamespace(
            width=600.0,
            height=800.0,
            rects=[
                {
                    "x0": 40.0,
                    "x1": 540.0,
                    "top": 120.0,
                    "bottom": 121.0,
                    "fill": True,
                    "stroke": False,
                    "non_stroking_color": (0.2, 0.5, 0.9),
                },
                {
                    "x0": 40.0,
                    "x1": 290.0,
                    "top": 121.0,
                    "bottom": 148.0,
                    "fill": True,
                    "stroke": False,
                    "non_stroking_color": 1,
                },
                {
                    "x0": 290.0,
                    "x1": 540.0,
                    "top": 121.0,
                    "bottom": 148.0,
                    "fill": True,
                    "stroke": False,
                    "non_stroking_color": 1,
                },
                {
                    "x0": 40.0,
                    "x1": 540.0,
                    "top": 148.0,
                    "bottom": 149.0,
                    "fill": True,
                    "stroke": False,
                    "non_stroking_color": (0.2, 0.5, 0.9),
                },
            ],
            horizontal_edges=[],
            vertical_edges=[],
            chars=[],
            filter=lambda fn: page,
            crop=lambda bbox: SimpleNamespace(
                extract_words=lambda **kwargs: [
                    {"text": "Escalation", "x0": 50.0, "x1": 100.0, "top": 126.0, "bottom": 134.0},
                    {"text": "lane", "x0": 104.0, "x1": 130.0, "top": 126.0, "bottom": 134.0},
                    {"text": "summary", "x0": 134.0, "x1": 190.0, "top": 126.0, "bottom": 134.0},
                ]
            ),
            extract_tables=lambda **kwargs: [],
        )

        from unittest.mock import patch

        expected_table = (
            [["Escalation"], ["lane"], ["summary"]],
            table_bbox,
        )

        with patch(
            "extractor.tables._table_regions",
            return_value=[(40.0, 540.0, [{"top": 121.0}, {"top": 148.0}])],
        ), patch(
            "extractor.tables._extract_tables_from_crop",
            return_value=[expected_table],
        ), patch(
            "extractor.tables._single_column_box_region_candidates",
            return_value=[
                {"bbox": (40.0, 121.0, 540.0, 148.0), "is_white_content": False},
                {"bbox": (40.0, 148.2, 540.0, 173.0), "is_white_content": False},
            ],
        ), patch(
            "extractor.tables._extract_text_from_box_region",
            side_effect=[
                "Escalation",
                "lane summary",
            ],
        ):
            tables = _extract_tables(page)

        self.assertEqual([expected_table], tables)

    def test_extract_tables_preserves_single_column_table_when_note_box_is_white_background(self) -> None:
        table_bbox = (40.0, 119.0, 540.0, 150.0)
        page = SimpleNamespace(
            width=600.0,
            height=800.0,
            rects=[],
            horizontal_edges=[],
            vertical_edges=[],
            chars=[],
            filter=lambda fn: page,
            crop=lambda bbox: SimpleNamespace(
                extract_words=lambda **kwargs: [
                    {"text": "Escalation", "x0": 50.0, "x1": 100.0, "top": 126.0, "bottom": 134.0},
                    {"text": "lane", "x0": 104.0, "x1": 130.0, "top": 126.0, "bottom": 134.0},
                    {"text": "summary", "x0": 134.0, "x1": 190.0, "top": 126.0, "bottom": 134.0},
                ]
            ),
            extract_tables=lambda **kwargs: [],
        )

        from unittest.mock import patch

        expected_table = (
            [["Escalation lane summary"], ["Owner confirmed"]],
            table_bbox,
        )

        with patch(
            "extractor.tables._table_regions",
            return_value=[(40.0, 540.0, [{"top": 121.0}, {"top": 148.0}])],
        ), patch(
            "extractor.tables._extract_tables_from_crop",
            return_value=[expected_table],
        ), patch(
            "extractor.tables._single_column_box_region_candidates",
            return_value=[
                {"bbox": (40.0, 121.0, 540.0, 148.0), "is_white_content": True}
            ],
        ), patch(
            "extractor.tables._extract_text_from_box_region",
            return_value="Escalation lane summary Owner confirmed for regional review and exception routing.",
        ):
            tables = _extract_tables(page)

        self.assertEqual([expected_table], tables)

    def test_single_column_boxes_share_index_when_aligned(self) -> None:
        self.assertTrue(
            _single_column_boxes_share_index(
                (120.0, 100.0, 520.0, 150.0),
                (118.0, 151.0, 518.0, 190.0),
            )
        )
        self.assertFalse(
            _single_column_boxes_share_index(
                (40.0, 100.0, 180.0, 150.0),
                (120.0, 151.0, 250.0, 190.0),
            )
        )

    def test_merge_single_column_fragment_rows_joins_same_index_fragments(self) -> None:
        merged_rows, merged_bbox = _merge_single_column_fragment_rows(
            (120.0, 100.0, 520.0, 140.0),
            [["Line one"], ["duplicate"]],
            (120.0, 150.0, 520.0, 190.0),
            [["Line two"], ["Line two"]],
        )
        self.assertEqual([["Line one"], ["duplicate"], ["Line two"]], merged_rows)
        self.assertEqual((120.0, 100.0, 520.0, 190.0), merged_bbox)

    def test_merge_single_column_fragment_rows_uses_first_non_empty_cell(self) -> None:
        merged_rows, merged_bbox = _merge_single_column_fragment_rows(
            (120.0, 100.0, 520.0, 140.0),
            [["", "Owner"], ["", "Owner"]],
            (120.0, 150.0, 520.0, 190.0),
            [["", "Confirmed"], ["", "Confirmed"]],
        )
        self.assertEqual([["", "Owner"], ["", "Confirmed"]], merged_rows)
        self.assertEqual((120.0, 100.0, 520.0, 190.0), merged_bbox)

    def test_single_column_box_region_candidates_allows_border_tolerance(self) -> None:
        page = SimpleNamespace(
            width=600.0,
            height=800.0,
            horizontal_edges=[],
            vertical_edges=[],
            rects=[
                {
                    "x0": 70.0,
                    "x1": 530.0,
                    "top": 101.6,
                    "bottom": 102.0,
                    "fill": True,
                    "stroke": False,
                    "non_stroking_color": (0.7, 0.8, 0.9),
                },
                {
                    "x0": 72.0,
                    "x1": 525.0,
                    "top": 102.0,
                    "bottom": 128.0,
                    "fill": True,
                    "stroke": False,
                    "non_stroking_color": (0.7, 0.8, 0.9),
                },
                {
                    "x0": 73.0,
                    "x1": 529.0,
                    "top": 128.0,
                    "bottom": 128.5,
                    "fill": True,
                    "stroke": False,
                    "non_stroking_color": (0.7, 0.8, 0.9),
                },
            ],
            filter=lambda fn: page,
            crop=lambda bbox: SimpleNamespace(extract_words=lambda **kwargs: []),
        )
        candidates = _single_column_box_region_candidates(page)
        self.assertEqual(1, len(candidates))
        self.assertEqual(
            (72.0, 102.0, 525.0, 128.0),
            tuple(round(value, 1) for value in candidates[0]["bbox"]),
        )

    def test_table_text_from_rows_collapses_two_header_rows_into_single_markdown_header(self) -> None:
        rows = [
            ["Stage", "Team", "Notes"],
            ["Group", "Function", "Deliverable"],
            ["Phase A", "Discovery", "Kickoff scope lock"],
        ]

        markdown = _table_text_from_rows(rows)

        self.assertIn("| Stage<br>Group | Team<br>Function | Notes<br>Deliverable |", markdown)
        self.assertIn("| Phase A | Discovery | Kickoff scope lock |", markdown)

    def test_table_text_from_rows_treats_long_single_column_as_note_content(self) -> None:
        rows = [
            ["F1-U path is not present in the integrated CU-DU shape. Hence, the counters for"],
            ["F1-U are not provided in this shape."],
        ]

        markdown = _table_text_from_rows(rows)

        self.assertIn("| Column 1 |", markdown)
        self.assertIn("| F1-U path is not present in the integrated CU-DU shape. Hence, the counters for |", markdown)
        self.assertIn("| F1-U are not provided in this shape. |", markdown)
        self.assertNotIn("| F1-U path is not present in the integrated CU-DU shape. Hence, the counters for | --- |", markdown)

    def test_single_column_note_body_text_is_single_line(self) -> None:
        rows = [
            ["For F1-U/Xn-U interface, GTP SN marking & Loss/OOS counting is always"],
            ["activated."],
        ]
        self.assertTrue(_looks_like_single_column_note(rows))
        self.assertEqual(
            "For F1-U/Xn-U interface, GTP SN marking & Loss/OOS counting is always activated.",
            _single_column_note_body_text(rows),
        )

    def test_single_column_note_body_text_uses_first_non_empty_cell(self) -> None:
        rows = [["", "", "Escalation lane summary"], ["", "", "Owner confirmed"]]
        self.assertTrue(_looks_like_single_column_note(rows))
        self.assertEqual(
            "Escalation lane summary Owner confirmed",
            _single_column_note_body_text(rows),
        )

    def test_single_column_note_detects_single_row_multiline_cell(self) -> None:
        rows = [
            [
                "Escalation lane summary\nOwner confirmed for regional review and exception routing.\n"
                "Backup approver stays on the same visual box and must not become a second table row.",
            ],
        ]
        self.assertTrue(_looks_like_single_column_note(rows))
        self.assertEqual(
            "Escalation lane summary Owner confirmed for regional review and exception routing. Backup approver stays on the same visual box and must not become a second table row.",
            _single_column_note_body_text(rows),
        )

    def test_single_column_classification_uses_page_geometry_and_grid_signals(self) -> None:
        long_rows = [
            ["For F1-U/Xn-U interface, GTP SN marking & Loss/OOS counting is always"],
            ["activated."],
        ]
        table_like_page = SimpleNamespace(
            width=600.0,
            height=800.0,
            vertical_edges=[
                {"x0": 280.0, "x1": 280.0, "top": 105.0, "bottom": 165.0, "stroke": True},
                {"x0": 300.0, "x1": 300.0, "top": 105.0, "bottom": 165.0, "stroke": True},
            ],
            horizontal_edges=[],
            filter=lambda fn: table_like_page,
            crop=lambda bbox: SimpleNamespace(
                extract_words=lambda **kwargs: [
                    {"text": "For", "x0": 50.0, "x1": 80.0, "top": 110.0, "bottom": 118.0},
                    {"text": "F1-U/Xn-U", "x0": 84.0, "x1": 140.0, "top": 110.0, "bottom": 118.0},
                    {"text": "interface,", "x0": 142.0, "x1": 190.0, "top": 110.0, "bottom": 118.0},
                    {"text": "GTP", "x0": 60.0, "x1": 90.0, "top": 126.0, "bottom": 134.0},
                    {"text": "SN", "x0": 94.0, "x1": 110.0, "top": 126.0, "bottom": 134.0},
                    {"text": "marking", "x0": 114.0, "x1": 170.0, "top": 126.0, "bottom": 134.0},
                    {"text": "Loss/OOS", "x0": 172.0, "x1": 230.0, "top": 126.0, "bottom": 134.0},
                    {"text": "counting", "x0": 234.0, "x1": 290.0, "top": 126.0, "bottom": 134.0},
                    {"text": "is", "x0": 50.0, "x1": 66.0, "top": 142.0, "bottom": 150.0},
                    {"text": "always", "x0": 70.0, "x1": 110.0, "top": 142.0, "bottom": 150.0},
                    {"text": "activated.", "x0": 114.0, "x1": 174.0, "top": 142.0, "bottom": 150.0},
                ]
            ),
        )
        self.assertFalse(_looks_like_single_column_note(rows=long_rows, page=table_like_page, bbox=(50.0, 100.0, 540.0, 170.0)))

        note_like_page = SimpleNamespace(
            width=600.0,
            height=800.0,
            vertical_edges=[],
            horizontal_edges=[],
            filter=lambda fn: note_like_page,
            crop=lambda bbox: table_like_page.crop(bbox),
        )
        self.assertTrue(_looks_like_single_column_note(rows=long_rows, page=note_like_page, bbox=(50.0, 100.0, 540.0, 170.0)))

    def test_looks_like_single_column_note_with_short_header_row(self) -> None:
        rows = [["Status"], ["Ready"]]
        self.assertFalse(_looks_like_single_column_note(rows))

    def test_parameter_description_rows_are_not_treated_as_note(self) -> None:
        rows = [
            ["", "Parameter", "", "", "Description", ""],
            ["", "ue-timer-poll-retransmit", "", "", "This parameter is the UE timer to retransmit the poll in a transmitting AM RLC entity.", ""],
            ["", "qci", "", "", "This parameter is the QoS Class Identifier(QCI).", ""],
        ]
        self.assertFalse(_looks_like_single_column_note(rows))

    def test_table_text_from_rows_preserves_single_column_header_when_short(self) -> None:
        rows = [
            ["Status"],
            ["Ready"],
        ]

        markdown = _table_text_from_rows(rows)

        self.assertIn("| Status |", markdown)
        self.assertIn("| --- |", markdown)
        self.assertIn("| Ready |", markdown)

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

    def test_single_column_box_regions_allow_short_height_boxes(self) -> None:
        page = SimpleNamespace(
            width=600.0,
            height=800.0,
            rects=[
                {
                    "x0": 40.0,
                    "x1": 540.0,
                    "top": 120.0,
                    "bottom": 121.0,
                    "fill": True,
                    "stroke": False,
                    "non_stroking_color": (0.2, 0.5, 0.9),
                },
                {
                    "x0": 40.0,
                    "x1": 290.0,
                    "top": 121.0,
                    "bottom": 148.0,
                    "fill": True,
                    "stroke": False,
                    "non_stroking_color": 1,
                },
                {
                    "x0": 290.0,
                    "x1": 540.0,
                    "top": 121.0,
                    "bottom": 148.0,
                    "fill": True,
                    "stroke": False,
                    "non_stroking_color": 1,
                },
                {
                    "x0": 40.0,
                    "x1": 540.0,
                    "top": 148.0,
                    "bottom": 149.0,
                    "fill": True,
                    "stroke": False,
                    "non_stroking_color": (0.2, 0.5, 0.9),
                },
            ],
            horizontal_edges=[],
            vertical_edges=[],
        )

        self.assertEqual([(40.0, 121.0, 540.0, 148.0)], _single_column_box_regions(page))
