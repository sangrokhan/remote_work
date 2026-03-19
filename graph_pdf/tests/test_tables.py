from __future__ import annotations

import unittest
from types import SimpleNamespace

from extractor.shared import _merge_horizontal_band_segments, _merge_vertical_band_segments
from extractor.tables import (
    _collapse_structural_triplet_columns,
    _continuation_regions_should_merge,
    _extract_tables,
    _split_repeated_header,
    _should_try_table_continuation_merge,
    _table_regions,
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
            [["", "Area", "", "Status", "Action"], ["note", "Docs", "", "READY", "Finalize"]],
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

    def test_table_text_from_rows_collapses_two_header_rows_into_single_markdown_header(self) -> None:
        rows = [
            ["Stage", "Team", "Notes"],
            ["Group", "Function", "Deliverable"],
            ["Phase A", "Discovery", "Kickoff scope lock"],
        ]

        markdown = _table_text_from_rows(rows)

        self.assertIn("| Stage<br>Group | Team<br>Function | Notes<br>Deliverable |", markdown)
        self.assertIn("| Phase A | Discovery | Kickoff scope lock |", markdown)

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
