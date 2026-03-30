from __future__ import annotations

import unittest
from types import SimpleNamespace

from extractor.notes import _collect_note_candidates, _note_body_text
from extractor.shared import _merge_horizontal_band_segments, _merge_vertical_band_segments
from extractor.tables import (
    _collapse_structural_triplet_columns,
    _continuation_regions_should_merge,
    _extract_tables_from_crop,
    _extract_tables,
    _dedupe_redundant_rectangles,
    _to_rect_entry,
    _note_group_region_candidates,
    _split_repeated_header,
    _should_try_table_continuation_merge,
    _table_owned_body_line_bboxes,
    _table_regions,
    _table_rejection_reason,
    _table_text_from_rows,
    _header_row_count,
)


class TableModuleTests(unittest.TestCase):
    def test_table_owned_body_line_bboxes_claims_orphan_header_near_table(self) -> None:
        from unittest.mock import patch

        page = SimpleNamespace()
        tables = [
            (
                [["Family Display Name", "Type Name", "Type Description"], ["Docs", "READY", "Finalize"]],
                (72.0, 668.0, 525.0, 720.0),
            )
        ]

        with patch(
            "extractor.tables._extract_body_word_lines",
            return_value=[
                {
                    "text": "Family Display Name Type Name Type Description",
                    "x0": 77.42,
                    "x1": 405.88,
                    "top": 652.9,
                    "bottom": 661.9,
                }
            ],
        ):
            owned = _table_owned_body_line_bboxes(
                page,
                tables=tables,
                header_margin=90.0,
                footer_margin=40.0,
            )

        self.assertEqual([(77.42, 652.9, 405.88, 661.9)], owned)

    def test_table_owned_body_line_bboxes_ignores_normal_body_line(self) -> None:
        from unittest.mock import patch

        page = SimpleNamespace()
        tables = [
            (
                [["Family Display Name", "Type Name", "Type Description"], ["Docs", "READY", "Finalize"]],
                (72.0, 668.0, 525.0, 720.0),
            )
        ]

        with patch(
            "extractor.tables._extract_body_word_lines",
            return_value=[
                {
                    "text": "This paragraph should remain body text.",
                    "x0": 77.42,
                    "x1": 320.0,
                    "top": 652.9,
                    "bottom": 661.9,
                }
            ],
        ):
            owned = _table_owned_body_line_bboxes(
                page,
                tables=tables,
                header_margin=90.0,
                footer_margin=40.0,
            )

        self.assertEqual([], owned)

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

    def test_collapse_structural_triplet_columns_collapses_sparse_split_pairs(self) -> None:
        table = [
            ["", "Family Displa", "y Name", "", "", "Type Name", "", "", "Type Description", ""],
            ["", "X2-U Interfa QCI", "ce per eNB IP per", "", "", "X2URxPacketLossCnt", "", "", "Lost packets", ""],
            ["", "", "", "", "", "X2URxPacketOosCnt", "", "", "OOS packets", ""],
        ]
        self.assertEqual(
            [
                ["Family Display Name", "Type Name", "Type Description"],
                ["X2-U Interface per eNB IP per QCI", "X2URxPacketLossCnt", "Lost packets"],
                ["", "X2URxPacketOosCnt", "OOS packets"],
            ],
            _collapse_structural_triplet_columns(table),
        )

    def test_collapse_structural_triplet_columns_keeps_space_for_full_word_companion(self) -> None:
        table = [
            ["", "Family Displa", "y Name", "", "", "Type Name", "", "", "Type Description", ""],
            ["", "N3 Interface", "per UPF IP", "", "", "N3RxPacketLossCnt", "", "", "Lost packets", ""],
        ]
        self.assertEqual(
            [
                ["Family Display Name", "Type Name", "Type Description"],
                ["N3 Interface per UPF IP", "N3RxPacketLossCnt", "Lost packets"],
            ],
            _collapse_structural_triplet_columns(table),
        )

    def test_collapse_structural_triplet_columns_reorders_collected_suffix_phrase(self) -> None:
        table = [
            ["", "Family Displa", "y Name", "", "", "Type Name", "", "", "Type Description", ""],
            ["", "N3 Interface UPF IP", "collected in UP per", "", "", "N3RxPacketLossCnt", "", "", "Lost packets", ""],
        ]
        self.assertEqual(
            [
                ["Family Display Name", "Type Name", "Type Description"],
                ["N3 Interface collected in UP per UPF IP", "N3RxPacketLossCnt", "Lost packets"],
            ],
            _collapse_structural_triplet_columns(table),
        )

    def test_collapse_structural_triplet_columns_reorders_collected_suffix_with_tail_qualifier(self) -> None:
        table = [
            ["", "Family Displa", "y Name", "", "", "Type Name", "", "", "Type Description", ""],
            ["", "F1-U Interfa per gNB-DU", "ce collected in UP per 5QI", "", "", "F1URxPacketLossCnt", "", "", "Lost packets", ""],
        ]
        self.assertEqual(
            [
                ["Family Display Name", "Type Name", "Type Description"],
                ["F1-U Interface collected in UP per 5QI per gNB-DU", "F1URxPacketLossCnt", "Lost packets"],
            ],
            _collapse_structural_triplet_columns(table),
        )

    def test_collapse_structural_triplet_columns_compresses_mixed_width_chunks(self) -> None:
        table = [
            ["", "Family Display Name", "", "", "Type Name", "", "", "Type Description", ""],
            ["", "S1-U Interface collected in UP per sGW IP per QCI", "", "", "S1URxPacketLossCnt", "", "", "Lost packets", ""],
            ["", "", "", "", "S1URxPacketOosCnt", "", "", "OOS packets", ""],
            ["", "F1-U Interfa per gNB-DU", "ce collected in UP per QCI", "", "", "F1URxPacketLossCnt", "", "", "Lost packets", ""],
            ["", "", "", "", "", "F1URxPacketOosCnt", "", "", "OOS packets", ""],
            ["", "X2-U Interfa per eNB IP p", "ce collected in UP er QCI", "", "", "X2URxPacketLossCnt", "", "", "Lost packets", ""],
            ["", "", "", "", "", "X2URxPacketOosCnt", "", "", "OOS packets", ""],
        ]
        self.assertEqual(
            [
                ["Family Display Name", "Type Name", "Type Description"],
                ["S1-U Interface collected in UP per sGW IP per QCI", "S1URxPacketLossCnt", "Lost packets"],
                ["", "S1URxPacketOosCnt", "OOS packets"],
                ["F1-U Interface collected in UP per QCI per gNB-DU", "F1URxPacketLossCnt", "Lost packets"],
                ["", "F1URxPacketOosCnt", "OOS packets"],
                ["X2-U Interface collected in UP per eNB IP per QCI", "X2URxPacketLossCnt", "Lost packets"],
                ["", "X2URxPacketOosCnt", "OOS packets"],
            ],
            _collapse_structural_triplet_columns(table),
        )

    def test_collapse_structural_triplet_columns_builds_family_type_description_layout_from_four_columns(self) -> None:
        table = [
            [
                "F1-U, XN-U collected in SNSSAI F1-U, XN-U collected in SNSSAI",
                "UL Interface UPC per 5QI per UL Interface UPP per 5QI per",
                "PacketLossCntUL",
                "Lost packets",
            ],
            ["", "", "PacketOosCntUL", "OOS packets"],
        ]
        self.assertEqual(
            [
                ["Family Display Name", "Type Name", "Type Description"],
                [
                    "F1-U, XN-U collected in UL Interface UPC per 5QI per SNSSAI\nF1-U, XN-U collected in UL Interface UPP per 5QI per SNSSAI",
                    "PacketLossCntUL",
                    "Lost packets",
                ],
                ["", "PacketOosCntUL", "OOS packets"],
            ],
            _collapse_structural_triplet_columns(table),
        )

    def test_collapse_structural_triplet_columns_normalizes_dl_family_display_name(self) -> None:
        table = [
            ["", "Family Displa", "y Name", "", "", "Type Name", "", "", "Type Description", ""],
            ["", "DL F1-U, Xn PRC per 5Q", "-U Interface per I per S-NSSAI", "", "", "PacketLossCntDL", "", "", "Lost packets", ""],
        ]
        self.assertEqual(
            [
                ["Family Display Name", "Type Name", "Type Description"],
                ["DL F1-U, Xn-U Interface per PRC per 5QI per S-NSSAI", "PacketLossCntDL", "Lost packets"],
            ],
            _collapse_structural_triplet_columns(table),
        )

    def test_table_text_from_rows_preserves_explicit_multiline_cells(self) -> None:
        rows = [
            ["Family Display Name", "Type Name", "Type Description"],
            [
                "F1-U, XN-U collected in UL Interface UPC per 5QI per SNSSAI\nF1-U, XN-U collected in UL Interface UPP per 5QI per SNSSAI",
                "PacketLossCntUL",
                "Lost packets",
            ],
        ]
        markdown = _table_text_from_rows(rows)
        self.assertIn(
            "F1-U, XN-U collected in UL Interface UPC per 5QI per SNSSAI<br>F1-U, XN-U collected in UL Interface UPP per 5QI per SNSSAI",
            markdown,
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
        table_regions.assert_called_once_with(page)
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
            "extractor.tables._detect_body_bounds",
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

    def test_collect_note_candidates_splits_sparse_multi_anchor_note_group(self) -> None:
        from unittest.mock import patch

        page = SimpleNamespace(width=600.0, height=800.0)
        all_line_payloads = [
            {
                "text": "The counter ‘F1-U UL Interface collected in UP per UP’ is provided only for SA",
                "top": 456.4,
                "bottom": 467.5,
                "x0": 70.0,
                "x1": 380.0,
            },
            {
                "text": "operation.",
                "top": 469.0,
                "bottom": 480.0,
                "x0": 70.0,
                "x1": 130.0,
            },
            {
                "text": "The counter for ‘F1-U UL Interface collected in UP per UP’ might not be",
                "top": 494.0,
                "bottom": 505.0,
                "x0": 70.0,
                "x1": 390.0,
            },
            {
                "text": "provided to the operator to which ‘F1-U, XN-U UL Interface collected in UPP per 5QI per",
                "top": 507.0,
                "bottom": 518.0,
                "x0": 70.0,
                "x1": 470.0,
            },
            {
                "text": "SNSSAI’ is provided.",
                "top": 519.4,
                "bottom": 530.5,
                "x0": 70.0,
                "x1": 180.0,
            },
            {
                "text": "The counter for the F1-U section is provided only for the CU-DU separation scenario.",
                "top": 544.6,
                "bottom": 568.3,
                "x0": 70.0,
                "x1": 420.0,
            },
        ]

        def _payloads_for_bbox(_page: SimpleNamespace, bbox: tuple[float, float, float, float]) -> list[dict[str, float | str]]:
            _x0, top, _x1, bottom = bbox
            return [
                payload
                for payload in all_line_payloads
                if float(payload["bottom"]) >= top and float(payload["top"]) <= bottom
            ]

        with patch(
            "extractor.notes._note_group_region_candidates",
            return_value=[(40.0, 444.5, 540.0, 571.6)],
        ), patch(
            "extractor.notes._select_note_anchor_for_bbox",
            side_effect=[
                (50.0, 447.2, 60.0, 465.0),
                (50.0, 535.4, 60.0, 553.2),
            ],
        ), patch(
            "extractor.notes._candidate_image_regions_for_notes",
            return_value=[
                (50.0, 447.2, 60.0, 465.0),
                (50.0, 485.0, 60.0, 502.8),
                (50.0, 535.4, 60.0, 553.2),
            ],
        ), patch(
            "extractor.notes._extract_region_line_payloads",
            side_effect=_payloads_for_bbox,
        ):
            candidates = _collect_note_candidates(page)

        note_candidates = [candidate for candidate in candidates if candidate["is_note_like"] and not candidate["is_white_content"]]
        self.assertEqual(3, len(note_candidates))
        self.assertEqual(
            [["The counter ‘F1-U UL Interface collected in UP per UP’ is provided only for SA"], ["operation."]],
            note_candidates[0]["rows"],
        )
        self.assertEqual(
            [
                ["The counter for ‘F1-U UL Interface collected in UP per UP’ might not be"],
                ["provided to the operator to which ‘F1-U, XN-U UL Interface collected in UPP per 5QI per"],
                ["SNSSAI’ is provided."],
            ],
            note_candidates[1]["rows"],
        )
        self.assertEqual(
            [["The counter for the F1-U section is provided only for the CU-DU separation scenario."]],
            note_candidates[2]["rows"],
        )

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

    def test_table_text_from_rows_does_not_pad_cells_to_column_width(self) -> None:
        rows = [
            ["Parameter", "Description"],
            ["qci", "This parameter is the QoS Class Identifier(QCI)."],
        ]

        markdown = _table_text_from_rows(rows)

        self.assertEqual(
            "\n".join(
                [
                    "| Parameter | Description |",
                    "| --- | --- |",
                    "| qci | This parameter is the QoS Class Identifier(QCI). |",
                ]
            ),
            markdown,
        )

    def test_header_row_count_does_not_promote_first_data_row_to_header(self) -> None:
        rows = [
            ["Interface / Direction", "Sender", "Receiver"],
            ["S1-U / Downlink", "S-GW", "CU-UP"],
            ["F1-U / Uplink", "DU", "CU-UP"],
        ]

        self.assertEqual(1, _header_row_count(rows))

    def test_collapse_structural_triplet_columns_normalizes_ul_family_display_variants(self) -> None:
        table = [
            ["", "Family Displa", "y Name", "", "", "Type Name", "", "", "Type Description", ""],
            ["", "F1-U UL Inte", "rface per QCI", "", "", "F1UPacketLossCntUL_QC I", "", "", "Lost packets", ""],
            ["", "F1-U UL Inte UP per QCI", "rface collected in", "", "", "F1UPacketLossRateUL_Q CI", "", "", "Lost packets", ""],
            ["", "F1-U UL Inte UP per UP", "rface collected in", "", "", "F1UPacketLossCntUL", "", "", "Lost packets", ""],
        ]

        self.assertEqual(
            [
                ["Family Display Name", "Type Name", "Type Description"],
                ["F1-U UL Interface per QCI", "F1UPacketLossCntUL_QCI", "Lost packets"],
                ["F1-U UL Interface collected in UP per QCI", "F1UPacketLossRateUL_QCI", "Lost packets"],
                ["F1-U UL Interface collected in UP per UP", "F1UPacketLossCntUL", "Lost packets"],
            ],
            _collapse_structural_triplet_columns(table),
        )

    def test_collapse_structural_triplet_columns_repairs_shifted_family_type_rows(self) -> None:
        table = [
            ["", "Family Display Name", "", "", "Type Name", "", "", "Type Description", ""],
            ["", "F1-U DL Interface per QCI F1-U DL Interface per PRC per QCI", "", "", "F1UPacketLossCntDL_QC I", "", "", "Lost packets", ""],
            ["", "", "F1UPacketOosCntDL_QCI", "OOS packets"],
            ["", "", "F1UPacketLossRateDL_Q CI", "Loss-rate packets"],
        ]

        self.assertEqual(
            [
                ["Family Display Name", "Type Name", "Type Description"],
                ["F1-U DL Interface per QCI\nF1-U DL Interface per PRC per QCI", "F1UPacketLossCntDL_QCI", "Lost packets"],
                ["", "F1UPacketOosCntDL_QCI", "OOS packets"],
                ["", "F1UPacketLossRateDL_QCI", "Loss-rate packets"],
            ],
            _collapse_structural_triplet_columns(table),
        )

    def test_collapse_structural_triplet_columns_normalizes_dl_du_family_display_name(self) -> None:
        table = [
            ["", "Family Displa", "y Name", "", "", "Type Name", "", "", "Type Description", ""],
            ["", "F1-U DL Inte F1-U DL Inte DU", "rface per DU rface per PRC per", "", "", "F1UPacketLossCntDL", "", "", "Lost packets", ""],
        ]

        self.assertEqual(
            [
                ["Family Display Name", "Type Name", "Type Description"],
                ["F1-U DL Interface per DU\nF1-U DL Interface per PRC per DU", "F1UPacketLossCntDL", "Lost packets"],
            ],
            _collapse_structural_triplet_columns(table),
        )

    def test_table_text_from_rows_preserves_single_column_table_without_note_reclassification(self) -> None:
        rows = [
            ["F1-U path is not present in the integrated CU-DU shape. Hence, the counters for"],
            ["F1-U are not provided in this shape."],
        ]

        markdown = _table_text_from_rows(rows)

        self.assertIn("| F1-U path is not present in the integrated CU-DU shape. Hence, the counters for |", markdown)
        self.assertIn("F1-U path is not present in the integrated CU-DU shape. Hence, the counters for", markdown)
        self.assertIn("F1-U are not provided in this shape.", markdown)
        self.assertIn("| --- |", markdown)

    def test_note_body_text_is_single_line(self) -> None:
        rows = [
            ["For F1-U/Xn-U interface, GTP SN marking & Loss/OOS counting is always"],
            ["activated."],
        ]
        self.assertEqual(
            "Note: For F1-U/Xn-U interface, GTP SN marking & Loss/OOS counting is always activated.",
            _note_body_text(rows),
        )

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
