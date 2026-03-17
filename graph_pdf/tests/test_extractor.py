from __future__ import annotations

import math
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from pathlib import Path

import pdfplumber

from extractor import (
    _build_body_blocks,
    _collapse_structural_triplet_columns,
    _detect_body_bounds,
    _char_rotation_degrees,
    _collect_rotated_text_debug,
    _collect_table_drawing_debug,
    _continuation_regions_should_merge,
    _extract_embedded_images,
    _extract_body_word_lines,
    _extract_tables,
    _is_gray_color,
    _is_non_watermark_obj,
    _looks_like_table,
    _normalize_list_block_lines,
    _normalize_body_lines,
    _table_rejection_reason,
    _merge_horizontal_band_segments,
    _merge_vertical_band_segments,
    _normalize_cell_lines,
    _parse_pages_spec,
    _should_try_table_continuation_merge,
    _table_regions,
    extract_pdf_to_outputs,
)
from verify import _extract_markdown_tables
from sample_fixture import load_demo_fixture
from sample_generator import create_demo_pdf


class TableExtractionFormattingTests(unittest.TestCase):
    def _build_pdf(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pdf_path = Path(tmp.name) / "sample.pdf"
        create_demo_pdf(pdf_path)
        return pdf_path

    def _extract_result(self) -> dict:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        pdf_path = root / "sample.pdf"
        md_dir = root / "md"
        image_dir = root / "images"
        create_demo_pdf(pdf_path)
        return extract_pdf_to_outputs(
            pdf_path=pdf_path,
            out_md_dir=md_dir,
            out_image_dir=image_dir,
            stem="sample",
        )

    def _extract_markdown(self) -> str:
        return self._extract_result()["markdown"]

    def _extract_table_markdown(self) -> str:
        return self._extract_result()["table_markdown"]

    def test_fixture_roundtrip_matches_expected_tables(self) -> None:
        fixture = load_demo_fixture()
        markdown = self._extract_table_markdown()
        extracted_tables = _extract_markdown_tables(markdown)
        extracted_by_index = {idx: rows for idx, rows in enumerate(extracted_tables)}

        for idx, table in enumerate(fixture["tables"]):
            self.assertIn(idx, extracted_by_index)
            self.assertEqual(table["rows"], extracted_by_index[idx], table["id"])

    def _table_blocks(self, markdown: str) -> list[str]:
        blocks: list[str] = []
        current: list[str] = []
        for line in markdown.splitlines():
            if line.startswith("### Page ") and " table " in line:
                if current:
                    blocks.append("\n".join(current))
                current = [line]
                continue
            if current:
                current.append(line)
        if current:
            blocks.append("\n".join(current))
        return blocks

    def test_table_output_uses_markdown_tables(self) -> None:
        markdown = self._extract_table_markdown()
        self.assertIn("### Page 1 table 1", markdown)
        self.assertIn("| Item | Qty | Price |", markdown)
        self.assertIn("| --- | --- | --- |", markdown)
        self.assertNotIn("- Row 1", markdown)
        self.assertIn("<br>", markdown)

    def test_watermark_fragments_do_not_remain_in_table_cells(self) -> None:
        fixture = load_demo_fixture()
        markdown = self._extract_table_markdown()
        self.assertNotIn(fixture["watermark_text"], markdown)

    def test_wrapped_cell_text_is_collapsed_but_bullets_remain_split(self) -> None:
        markdown = self._extract_table_markdown()
        self.assertIn("| Laptop<br>- line 1 | 12 | $120 |", markdown)
        self.assertIn(
            "Docking station compatibility review package for extended desktop deployment approval",
            markdown,
        )
        self.assertIn("| Docs | READY | Finalize<br>- sample<br>- archive |", markdown)
        self.assertIn(
            "Escalation owner confirmed.<br>Regional fallback documented.<br>Launch blackout window approved.",
            markdown,
        )

    def test_punctuation_ends_logical_cell_line(self) -> None:
        cell = (
            "First sentence ends here.\n"
            "Second sentence starts here.\n"
            "Wrapped continuation without punctuation\n"
            "still belongs together"
        )
        self.assertEqual(
            [
                "First sentence ends here.",
                "Second sentence starts here.",
                "Wrapped continuation without punctuation still belongs together",
            ],
            _normalize_cell_lines(cell),
        )

    def test_hyphen_ended_line_joins_next_line_without_space(self) -> None:
        cell = "cross-\nborder policy"
        self.assertEqual(["cross-border policy"], _normalize_cell_lines(cell))

    def test_hyphen_ended_line_does_not_absorb_next_bullet(self) -> None:
        cell = "review-\n- next item"
        self.assertEqual(["review-", "- next item"], _normalize_cell_lines(cell))

    def test_hollow_circle_like_o_is_treated_as_bullet(self) -> None:
        cell = "review-\no next item"
        self.assertEqual(["review-", "o next item"], _normalize_cell_lines(cell))
        self.assertEqual(
            ["Wrapped sentence line", "o next item"],
            _normalize_body_lines(["Wrapped sentence line", "o next item"]),
        )

    def test_unknown_glyph_like_question_mark_is_treated_as_bullet(self) -> None:
        cell = "review-\n? next item"
        self.assertEqual(["review-", "? next item"], _normalize_cell_lines(cell))
        self.assertEqual(
            ["Wrapped sentence line", "? next item"],
            _normalize_body_lines(["Wrapped sentence line", "? next item"]),
        )

    def test_diamond_bullet_starts_new_item_instead_of_continuation(self) -> None:
        cell = "review-\n◆ next item"
        self.assertEqual(["review-", "◆ next item"], _normalize_cell_lines(cell))
        self.assertEqual(
            ["Wrapped sentence line", "◆ next item"],
            _normalize_body_lines(["Wrapped sentence line", "◆ next item"]),
        )

    def test_normalize_body_lines_joins_wrapped_sentence_lines(self) -> None:
        lines = [
            "This paragraph starts on one visual line and",
            "continues on the next extracted line without punctuation",
            "- bullet item should stay separate",
            "Chapter 2: Heading stays separate",
            "The next paragraph also starts cleanly.",
        ]

        self.assertEqual(
            [
                "This paragraph starts on one visual line and continues on the next extracted line without punctuation",
                "- bullet item should stay separate",
                "Chapter 2: Heading stays separate",
                "The next paragraph also starts cleanly.",
            ],
            _normalize_body_lines(lines),
        )

    def test_build_body_blocks_splits_heading_paragraph_and_list(self) -> None:
        lines = [
            {"text": "Chapter 1: Deep Structure Verification", "x0": 36.0, "x1": 260.0, "top": 90.0, "bottom": 102.0, "size": 14.0, "fontname": "Helvetica-Bold", "color": (0.0, 0.0, 0.0), "is_bold": True, "is_italic": False},
            {"text": "This paragraph starts on one extracted line", "x0": 36.0, "x1": 320.0, "top": 118.0, "bottom": 130.0, "size": 11.0, "fontname": "Helvetica", "color": (0.0, 0.0, 0.0), "is_bold": False, "is_italic": False},
            {"text": "and continues on the next extracted line", "x0": 36.0, "x1": 310.0, "top": 132.0, "bottom": 144.0, "size": 11.0, "fontname": "Helvetica", "color": (0.0, 0.0, 0.0), "is_bold": False, "is_italic": False},
            {"text": "- bullet item", "x0": 48.0, "x1": 120.0, "top": 160.0, "bottom": 172.0, "size": 11.0, "fontname": "Helvetica", "color": (0.0, 0.0, 0.0), "is_bold": False, "is_italic": False},
        ]

        blocks = _build_body_blocks(lines)

        self.assertEqual(
            [
                {"kind": "heading", "lines": ["Chapter 1: Deep Structure Verification"]},
                {"kind": "paragraph", "lines": ["This paragraph starts on one extracted line", "and continues on the next extracted line"]},
                {"kind": "list", "lines": ["- bullet item"]},
            ],
            [{"kind": block["kind"], "lines": [line["text"] for line in block["lines"]]} for block in blocks],
        )

    def test_build_body_blocks_splits_when_color_changes_between_lines(self) -> None:
        lines = [
            {"text": "First paragraph line", "x0": 36.0, "x1": 220.0, "top": 120.0, "bottom": 132.0, "size": 11.0, "fontname": "Helvetica", "color": (0.0, 0.0, 0.0), "is_bold": False, "is_italic": False},
            {"text": "Second line with different color", "x0": 36.0, "x1": 250.0, "top": 134.0, "bottom": 146.0, "size": 11.0, "fontname": "Helvetica", "color": (0.4, 0.4, 0.4), "is_bold": False, "is_italic": False},
        ]

        blocks = _build_body_blocks(lines)

        self.assertEqual(2, len(blocks))
        self.assertEqual(["First paragraph line"], [line["text"] for line in blocks[0]["lines"]])
        self.assertEqual(["Second line with different color"], [line["text"] for line in blocks[1]["lines"]])

    def test_build_body_blocks_keeps_paragraph_together_despite_style_change_when_sentence_continues(self) -> None:
        lines = [
            {"text": "This line introduces the uncommon term", "x0": 36.0, "x1": 260.0, "top": 120.0, "bottom": 132.0, "size": 11.0, "fontname": "Helvetica", "color": (0.0, 0.0, 0.0), "is_bold": False, "is_italic": False, "word_count": 6, "has_mixed_styles": False, "first_word_style_signature": ("Helvetica", False, False, None)},
            {"text": "ProtoLexeme expands into explanation", "x0": 36.0, "x1": 220.0, "top": 134.0, "bottom": 146.0, "size": 11.0, "fontname": "Helvetica", "color": (0.2, 0.2, 0.7), "is_bold": False, "is_italic": False, "word_count": 4, "has_mixed_styles": True, "first_word_style_signature": ("Helvetica-Bold", True, False, None)},
        ]

        blocks = _build_body_blocks(lines)

        self.assertEqual(1, len(blocks))
        self.assertEqual("paragraph", blocks[0]["kind"])
        self.assertEqual(
            ["This line introduces the uncommon term", "ProtoLexeme expands into explanation"],
            [line["text"] for line in blocks[0]["lines"]],
        )

    def test_extract_body_word_lines_marks_marker_candidate_and_text_start(self) -> None:
        filtered_page = SimpleNamespace(
            extract_words=lambda **kwargs: [
                {"text": "?", "x0": 48.0, "x1": 54.0, "top": 120.0, "bottom": 132.0, "size": 11.0, "fontname": "Symbol"},
                {"text": "bullet", "x0": 64.0, "x1": 92.0, "top": 120.0, "bottom": 132.0, "size": 11.0, "fontname": "Helvetica"},
                {"text": "text", "x0": 96.0, "x1": 118.0, "top": 120.0, "bottom": 132.0, "size": 11.0, "fontname": "Helvetica"},
            ]
        )
        page = SimpleNamespace()

        with patch("extractor._filter_page_for_extraction", return_value=filtered_page), patch(
            "extractor._detect_body_bounds", return_value=(40.0, 700.0)
        ):
            lines = _extract_body_word_lines(page, header_margin=90.0, footer_margin=40.0)

        self.assertEqual(1, len(lines))
        self.assertTrue(lines[0]["marker_candidate"])
        self.assertEqual(64.0, lines[0]["text_start_x"])

    def test_build_body_blocks_splits_when_bold_changes_between_lines(self) -> None:
        lines = [
            {"text": "Regular line", "x0": 36.0, "x1": 140.0, "top": 120.0, "bottom": 132.0, "size": 11.0, "fontname": "Helvetica", "color": (0.0, 0.0, 0.0), "is_bold": False, "is_italic": False},
            {"text": "Bold line", "x0": 36.0, "x1": 120.0, "top": 134.0, "bottom": 146.0, "size": 11.0, "fontname": "Helvetica-Bold", "color": (0.0, 0.0, 0.0), "is_bold": True, "is_italic": False},
        ]

        blocks = _build_body_blocks(lines)

        self.assertEqual(2, len(blocks))
        self.assertEqual(["Regular line"], [line["text"] for line in blocks[0]["lines"]])
        self.assertEqual(["Bold line"], [line["text"] for line in blocks[1]["lines"]])

    def test_build_body_blocks_keeps_bullet_continuation_aligned_with_text_start(self) -> None:
        lines = [
            {"text": "- bullet item starts here", "x0": 48.0, "x1": 220.0, "top": 120.0, "bottom": 132.0, "size": 11.0, "fontname": "Helvetica", "color": (0.0, 0.0, 0.0), "is_bold": False, "is_italic": False, "text_start_x": 64.0},
            {"text": "and continues aligned with item text", "x0": 64.0, "x1": 260.0, "top": 134.0, "bottom": 146.0, "size": 11.0, "fontname": "Helvetica", "color": (0.0, 0.0, 0.0), "is_bold": False, "is_italic": False, "text_start_x": 64.0},
        ]

        blocks = _build_body_blocks(lines)

        self.assertEqual(1, len(blocks))
        self.assertEqual("list", blocks[0]["kind"])
        self.assertEqual(
            ["- bullet item starts here", "and continues aligned with item text"],
            [line["text"] for line in blocks[0]["lines"]],
        )

    def test_build_body_blocks_keeps_bullet_continuation_when_further_indented(self) -> None:
        lines = [
            {"text": "- bullet item starts here", "x0": 48.0, "x1": 220.0, "top": 120.0, "bottom": 132.0, "size": 11.0, "fontname": "Helvetica", "color": (0.0, 0.0, 0.0), "is_bold": False, "is_italic": False, "text_start_x": 64.0},
            {"text": "continuation under the bullet body", "x0": 72.0, "x1": 260.0, "top": 134.0, "bottom": 146.0, "size": 11.0, "fontname": "Helvetica", "color": (0.0, 0.0, 0.0), "is_bold": False, "is_italic": False, "text_start_x": 72.0},
        ]

        blocks = _build_body_blocks(lines)

        self.assertEqual(1, len(blocks))
        self.assertEqual("list", blocks[0]["kind"])

    def test_normalize_list_block_lines_merges_continuation_lines_into_item(self) -> None:
        lines = [
            {"text": "- bullet item starts here", "x0": 48.0, "x1": 220.0, "top": 120.0, "bottom": 132.0, "size": 11.0, "fontname": "Helvetica", "color": (0.0, 0.0, 0.0), "is_bold": False, "is_italic": False, "text_start_x": 64.0},
            {"text": "and continues aligned with item text", "x0": 64.0, "x1": 260.0, "top": 134.0, "bottom": 146.0, "size": 11.0, "fontname": "Helvetica", "color": (0.0, 0.0, 0.0), "is_bold": False, "is_italic": False, "text_start_x": 64.0},
            {"text": "- next bullet", "x0": 48.0, "x1": 130.0, "top": 160.0, "bottom": 172.0, "size": 11.0, "fontname": "Helvetica", "color": (0.0, 0.0, 0.0), "is_bold": False, "is_italic": False, "text_start_x": 64.0},
        ]

        self.assertEqual(
            [
                "- bullet item starts here and continues aligned with item text",
                "- next bullet",
            ],
            _normalize_list_block_lines(lines),
        )

    def test_parse_pages_spec_supports_ranges_and_lists(self) -> None:
        self.assertEqual([1, 3, 4, 5, 8], _parse_pages_spec("1,3-5,8"))

    def test_collect_rotated_text_debug_keeps_non_zero_angle_chars(self) -> None:
        page = SimpleNamespace(
            chars=[
                {"text": "A", "matrix": (1.0, 0.0, 0.0, 1.0, 0.0, 0.0), "top": 10.0, "bottom": 20.0},
                {"text": "B", "matrix": (0.573576, 0.819152, -0.819152, 0.573576, 1.0, 1.0), "top": 30.0, "bottom": 40.0},
            ]
        )

        entries = _collect_rotated_text_debug(page, page_no=2)

        self.assertEqual(1, len(entries))
        self.assertEqual("B", entries[0]["text"])
        self.assertAlmostEqual(55.0, entries[0]["rotation"], places=1)

    def test_looks_like_table_tolerates_none_cells(self) -> None:
        table = [
            [None, "Status", "Notes"],
            ["Docs", None, "Ready"],
        ]
        self.assertTrue(_looks_like_table(table))

    def test_looks_like_table_allows_single_column_when_other_checks_pass(self) -> None:
        table = [
            ["Status"],
            ["Ready"],
        ]
        self.assertTrue(_looks_like_table(table))

    def test_table_rejection_reason_no_longer_rejects_large_row_count_by_size_only(self) -> None:
        table = [["Value"] for _ in range(81)]
        self.assertIsNone(_table_rejection_reason(table))

    def test_table_rejection_reason_allows_sparse_tables(self) -> None:
        table = [
            ["Status", "", ""],
            ["Ready", "", ""],
        ]
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
            [
                ["", "Area", "", "Status", "Action"],
                ["note", "Docs", "", "READY", "Finalize"],
            ],
            _collapse_structural_triplet_columns(table),
        )

    def test_gray_text_between_53_and_57_degrees_is_treated_as_watermark(self) -> None:
        char = {
            "object_type": "char",
            "matrix": (0.588, 0.809, -0.809, 0.588, 0.0, 0.0),
            "non_stroking_color": (0.92, 0.92, 0.92),
        }
        self.assertFalse(_is_non_watermark_obj(char))

    def test_light_neutral_gray_range_is_detected(self) -> None:
        self.assertTrue(_is_gray_color((0.92, 0.92, 0.92)))
        self.assertFalse(_is_gray_color((0.92, 0.86, 0.92)))
        self.assertFalse(_is_gray_color((0.72, 0.72, 0.72)))

    def test_detect_body_bounds_uses_long_top_and_bottom_rules_when_present(self) -> None:
        page = SimpleNamespace(
            width=600.0,
            height=800.0,
            horizontal_edges=[
                {"x0": 40.0, "x1": 560.0, "top": 52.0},
                {"x0": 60.0, "x1": 220.0, "top": 180.0},
                {"x0": 35.0, "x1": 565.0, "top": 742.0},
            ],
        )

        body_top, body_bottom = _detect_body_bounds(page, header_margin=90.0, footer_margin=40.0)

        self.assertEqual(52.0, body_top)
        self.assertEqual(742.0, body_bottom)

    def test_detect_body_bounds_falls_back_when_no_dividers_exist(self) -> None:
        page = SimpleNamespace(width=600.0, height=800.0, horizontal_edges=[])

        body_top, body_bottom = _detect_body_bounds(page, header_margin=90.0, footer_margin=40.0)

        self.assertEqual(40.0, body_top)
        self.assertEqual(710.0, body_bottom)

    def test_detect_body_bounds_uses_large_chapter_line_when_no_top_divider_exists(self) -> None:
        page = SimpleNamespace(
            width=600.0,
            height=800.0,
            horizontal_edges=[
                {"x0": 35.0, "x1": 565.0, "top": 742.0},
            ],
            extract_words=lambda **kwargs: [
                {
                    "text": "Chapter",
                    "x0": 36.0,
                    "x1": 110.0,
                    "top": 96.0,
                    "bottom": 114.0,
                    "size": 20.0,
                    "fontname": "Helvetica-Bold",
                },
                {
                    "text": "3:",
                    "x0": 118.0,
                    "x1": 140.0,
                    "top": 96.0,
                    "bottom": 114.0,
                    "size": 20.0,
                    "fontname": "Helvetica-Bold",
                },
                {
                    "text": "New",
                    "x0": 148.0,
                    "x1": 190.0,
                    "top": 96.0,
                    "bottom": 114.0,
                    "size": 20.0,
                    "fontname": "Helvetica-Bold",
                },
                {
                    "text": "Section",
                    "x0": 198.0,
                    "x1": 260.0,
                    "top": 96.0,
                    "bottom": 114.0,
                    "size": 20.0,
                    "fontname": "Helvetica-Bold",
                },
            ],
        )

        body_top, body_bottom = _detect_body_bounds(page, header_margin=90.0, footer_margin=40.0)

        self.assertEqual(96.0, body_top)
        self.assertEqual(742.0, body_bottom)

    def test_continuation_regions_merge_when_near_footer_header_with_shared_axes(self) -> None:
        prev_bbox = (100.0, 620.0, 400.0, 705.0)
        curr_bbox = (102.0, 88.0, 402.0, 190.0)
        prev_axes = [180.0, 280.0]
        curr_axes = [180.4, 279.8]

        should_merge = _continuation_regions_should_merge(
            prev_bbox=prev_bbox,
            curr_bbox=curr_bbox,
            prev_axes=prev_axes,
            curr_axes=curr_axes,
            body_top=72.0,
            body_bottom=722.0,
            gap_text_boxes=[],
        )

        self.assertTrue(should_merge)

    def test_continuation_regions_merge_without_boundary_proximity_when_gap_has_no_body_content(self) -> None:
        prev_bbox = (100.0, 540.0, 400.0, 610.0)
        curr_bbox = (102.0, 140.0, 402.0, 220.0)
        prev_axes = [180.0, 280.0]
        curr_axes = [180.4, 279.8]

        should_merge = _continuation_regions_should_merge(
            prev_bbox=prev_bbox,
            curr_bbox=curr_bbox,
            prev_axes=prev_axes,
            curr_axes=curr_axes,
            body_top=72.0,
            body_bottom=722.0,
            gap_text_boxes=[],
        )

        self.assertTrue(should_merge)

    def test_continuation_regions_do_not_merge_when_gap_has_other_content(self) -> None:
        prev_bbox = (100.0, 620.0, 400.0, 705.0)
        curr_bbox = (102.0, 88.0, 402.0, 190.0)
        prev_axes = [180.0, 280.0]
        curr_axes = [180.4, 279.8]

        should_merge = _continuation_regions_should_merge(
            prev_bbox=prev_bbox,
            curr_bbox=curr_bbox,
            prev_axes=prev_axes,
            curr_axes=curr_axes,
            body_top=72.0,
            body_bottom=722.0,
            gap_text_boxes=[(40.0, 730.0, 80.0, 742.0)],
        )

        self.assertFalse(should_merge)

    def test_same_page_tables_do_not_trigger_continuation_merge(self) -> None:
        self.assertFalse(_should_try_table_continuation_merge(pending_page=2, current_page=2))
        self.assertTrue(_should_try_table_continuation_merge(pending_page=2, current_page=3))

    def test_table_regions_split_when_components_have_different_vertical_edges(self) -> None:
        page = SimpleNamespace(
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

        groups = _table_regions(page)

        self.assertEqual(2, len(groups))

    def test_table_regions_merge_when_components_share_vertical_edges(self) -> None:
        page = SimpleNamespace(
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

        groups = _table_regions(page)

        self.assertEqual(1, len(groups))

    def test_merge_horizontal_band_segments_handles_contained_and_overlapping_lines(self) -> None:
        segments = [
            {"x0": 10.0, "x1": 50.0, "top": 100.0, "bottom": 100.0},
            {"x0": 20.0, "x1": 40.0, "top": 100.0, "bottom": 100.0},
            {"x0": 49.5, "x1": 80.0, "top": 100.0, "bottom": 100.0},
            {"x0": 80.5, "x1": 100.0, "top": 100.0, "bottom": 100.0},
        ]

        merged = _merge_horizontal_band_segments(segments, tolerance=1.0)

        self.assertEqual(
            [{"x0": 10.0, "x1": 100.0, "top": 100.0, "bottom": 100.0}],
            merged,
        )

    def test_merge_horizontal_band_segments_is_order_independent_with_contained_lines(self) -> None:
        segments = [
            {"x0": 50.4, "x1": 80.0, "top": 100.0, "bottom": 100.2},
            {"x0": 20.0, "x1": 40.0, "top": 99.9, "bottom": 100.1},
            {"x0": 10.0, "x1": 50.0, "top": 100.0, "bottom": 100.0},
            {"x0": 80.4, "x1": 100.0, "top": 100.0, "bottom": 100.0},
        ]

        merged = _merge_horizontal_band_segments(segments, tolerance=1.0)

        self.assertEqual(
            [{"x0": 10.0, "x1": 100.0, "top": 99.9, "bottom": 100.2}],
            merged,
        )

    def test_merge_horizontal_band_segments_keeps_gaps_larger_than_tolerance(self) -> None:
        segments = [
            {"x0": 10.0, "x1": 50.0, "top": 100.0, "bottom": 100.0},
            {"x0": 20.0, "x1": 40.0, "top": 100.0, "bottom": 100.0},
            {"x0": 55.1, "x1": 80.0, "top": 100.0, "bottom": 100.0},
        ]

        merged = _merge_horizontal_band_segments(segments, tolerance=1.0)

        self.assertEqual(
            [
                {"x0": 10.0, "x1": 50.0, "top": 100.0, "bottom": 100.0},
                {"x0": 55.1, "x1": 80.0, "top": 100.0, "bottom": 100.0},
            ],
            merged,
        )

    def test_merge_vertical_band_segments_handles_contained_and_overlapping_lines(self) -> None:
        segments = [
            {"x0": 200.0, "x1": 200.0, "top": 10.0, "bottom": 50.0},
            {"x0": 200.0, "x1": 200.0, "top": 20.0, "bottom": 40.0},
            {"x0": 200.0, "x1": 200.0, "top": 49.5, "bottom": 80.0},
            {"x0": 200.0, "x1": 200.0, "top": 80.5, "bottom": 100.0},
        ]

        merged = _merge_vertical_band_segments(segments, tolerance=1.0)

        self.assertEqual(
            [{"x0": 200.0, "x1": 200.0, "top": 10.0, "bottom": 100.0}],
            merged,
        )

    def test_merge_vertical_band_segments_is_order_independent_with_contained_lines(self) -> None:
        segments = [
            {"x0": 199.8, "x1": 200.0, "top": 50.4, "bottom": 80.0},
            {"x0": 200.0, "x1": 200.1, "top": 20.0, "bottom": 40.0},
            {"x0": 200.0, "x1": 200.0, "top": 10.0, "bottom": 50.0},
            {"x0": 200.0, "x1": 200.0, "top": 80.4, "bottom": 100.0},
        ]

        merged = _merge_vertical_band_segments(segments, tolerance=1.0)

        self.assertEqual(
            [{"x0": 199.8, "x1": 200.1, "top": 10.0, "bottom": 100.0}],
            merged,
        )

    def test_merge_vertical_band_segments_keeps_gaps_larger_than_tolerance(self) -> None:
        segments = [
            {"x0": 200.0, "x1": 200.0, "top": 10.0, "bottom": 50.0},
            {"x0": 200.0, "x1": 200.0, "top": 20.0, "bottom": 40.0},
            {"x0": 200.0, "x1": 200.0, "top": 55.2, "bottom": 80.0},
        ]

        merged = _merge_vertical_band_segments(segments, tolerance=1.0)

        self.assertEqual(
            [
                {"x0": 200.0, "x1": 200.0, "top": 10.0, "bottom": 50.0},
                {"x0": 200.0, "x1": 200.0, "top": 55.2, "bottom": 80.0},
            ],
            merged,
        )

    def test_merge_horizontal_band_segments_preserves_full_vertical_span(self) -> None:
        segments = [
            {"x0": 10.0, "x1": 50.0, "top": 100.0, "bottom": 100.6},
            {"x0": 50.4, "x1": 80.0, "top": 99.6, "bottom": 100.0},
        ]

        merged = _merge_horizontal_band_segments(segments, tolerance=1.0)

        self.assertEqual(
            [{"x0": 10.0, "x1": 80.0, "top": 99.6, "bottom": 100.6}],
            merged,
        )

    def test_merge_vertical_band_segments_preserves_full_horizontal_span(self) -> None:
        segments = [
            {"x0": 199.6, "x1": 200.0, "top": 10.0, "bottom": 50.0},
            {"x0": 200.0, "x1": 200.6, "top": 49.5, "bottom": 80.0},
        ]

        merged = _merge_vertical_band_segments(segments, tolerance=1.0)

        self.assertEqual(
            [{"x0": 199.6, "x1": 200.6, "top": 10.0, "bottom": 80.0}],
            merged,
        )

    def test_extract_can_limit_to_selected_pages(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        pdf_path = root / "sample.pdf"
        create_demo_pdf(pdf_path)

        result = extract_pdf_to_outputs(
            pdf_path=pdf_path,
            out_md_dir=root / "md",
            out_image_dir=root / "images",
            stem="sample",
            pages=[3],
        )

        self.assertNotIn("### Page 1", result["markdown"])
        self.assertIn("### Page 3", result["markdown"])
        self.assertNotIn("### Page 1 table 1", result["table_markdown"])
        self.assertIn("### Page 3 table 1", result["table_markdown"])
        self.assertIn("### Page 3 table 2", result["table_markdown"])
        self.assertEqual(2, result["summary"]["table_count"])
        self.assertEqual(1, len(result["image_files"]))
        self.assertTrue(result["image_files"][0].name.startswith("sample_page_03_image_"))

    def test_extract_tables_skips_page_wide_fallback_by_default(self) -> None:
        page = SimpleNamespace(
            width=600.0,
            height=800.0,
            horizontal_edges=[],
            vertical_edges=[],
            extract_tables=MagicMock(return_value=[[["A", "B"], ["1", "2"]]]),
            filter=lambda fn: page,
        )

        self.assertEqual([], _extract_tables(page))
        page.extract_tables.assert_not_called()

    def test_extract_embedded_images_respects_selected_pages(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out_dir = Path(tmp.name) / "images"
        fake_reader = SimpleNamespace(
            pages=[
                SimpleNamespace(images=[SimpleNamespace(name="p1.png", data=b"one")]),
                SimpleNamespace(images=[SimpleNamespace(name="p2.png", data=b"two")]),
                SimpleNamespace(images=[SimpleNamespace(name="p3.png", data=b"three")]),
            ]
        )
        fake_plumber_pdf = SimpleNamespace(
            pages=[
                SimpleNamespace(
                    width=600.0,
                    height=800.0,
                    horizontal_edges=[],
                    images=[{"name": "p1", "top": 100.0, "bottom": 120.0}],
                ),
                SimpleNamespace(
                    width=600.0,
                    height=800.0,
                    horizontal_edges=[],
                    images=[{"name": "p2", "top": 100.0, "bottom": 120.0}],
                ),
                SimpleNamespace(
                    width=600.0,
                    height=800.0,
                    horizontal_edges=[],
                    images=[{"name": "p3", "top": 100.0, "bottom": 120.0}],
                ),
            ],
        )
        fake_plumber_open = MagicMock()
        fake_plumber_open.__enter__.return_value = fake_plumber_pdf
        fake_plumber_open.__exit__.return_value = False

        with patch("extractor.PdfReader", return_value=fake_reader), patch(
            "extractor.pdfplumber.open", return_value=fake_plumber_open
        ):
            image_files = _extract_embedded_images(
                pdf_path=Path("ignored.pdf"),
                out_image_dir=out_dir,
                stem="sample",
                pages=[2],
            )

        self.assertEqual(1, len(image_files))
        self.assertEqual("sample_page_02_image_01.png", image_files[0].name)

    def test_spanning_stage_table_merges_into_one_block(self) -> None:
        markdown = self._extract_table_markdown()
        blocks = self._table_blocks(markdown)
        self.assertEqual(3, len(blocks))
        stage_block = next((block for block in blocks if "Phase A" in block), "")
        self.assertTrue(stage_block)
        self.assertIn("### Page 1 table 2", markdown)
        self.assertIn("Release Notes", stage_block)
        self.assertIn("Phase C", stage_block)
        self.assertIn("Finance", stage_block)
        self.assertNotIn("### Page 2 table 3", markdown)

    def test_demo_markdown_contains_only_body_text(self) -> None:
        result = self._extract_result()
        markdown = result["markdown"]
        self.assertIn("Chapter 1: Deep Structure Verification", markdown)
        self.assertIn(
            "This intentionally verbose sentence exists to force a visual wrap in the sample PDF while still representing single logical sentence for downstream language-model parsing and retrieval quality checks across extracted body text.",
            markdown,
        )
        self.assertNotIn("### Page 1 table 1", markdown)
        self.assertNotIn("| Item | Qty | Price |", markdown)
        self.assertNotIn("Phase A", markdown)
        self.assertNotIn("Finalize<br>- sample<br>- archive", markdown)
        self.assertEqual(markdown, result["md_file"].read_text(encoding="utf-8"))

    def test_demo_table_markdown_is_written_to_separate_file(self) -> None:
        result = self._extract_result()
        table_markdown = result["table_markdown"]
        table_md_file = result["table_md_file"]
        self.assertEqual(table_markdown, table_md_file.read_text(encoding="utf-8"))
        self.assertTrue(table_md_file.name.endswith("_table.md"))
        self.assertIn("### Page 1 table 1", table_markdown)
        self.assertEqual(2, len(result["image_files"]))

    def test_header_images_are_not_exported_as_body_images(self) -> None:
        pdf_path = self._build_pdf()
        with pdfplumber.open(str(pdf_path)) as pdf:
            raw_image_count = sum(len(page.images) for page in pdf.pages)

        result = self._extract_result()
        self.assertEqual(5, raw_image_count)
        self.assertEqual(2, len(result["image_files"]))

    def test_debug_watermark_writes_rotated_text_log(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        pdf_path = root / "sample.pdf"
        create_demo_pdf(pdf_path)

        result = extract_pdf_to_outputs(
            pdf_path=pdf_path,
            out_md_dir=root / "md",
            out_image_dir=root / "images",
            stem="sample",
            debug_watermark=True,
        )

        debug_file = result["debug_watermark_file"]
        self.assertIsNotNone(debug_file)
        payload = json.loads(debug_file.read_text(encoding="utf-8"))
        self.assertTrue(payload)
        self.assertTrue(any(abs(entry["rotation"]) > 0.1 for entry in payload))

    def test_collect_table_drawing_debug_reports_expected_page1_grid(self) -> None:
        pdf_path = self._build_pdf()
        with pdfplumber.open(str(pdf_path)) as pdf:
            payload = _collect_table_drawing_debug(pdf.pages[0], page_no=1)

        self.assertEqual(1, payload["page"])
        self.assertEqual(2, len(payload["tables"]))
        self.assertEqual(6, payload["tables"][0]["row_count"])
        self.assertEqual(3, payload["tables"][0]["col_count"])
        self.assertEqual(3, payload["tables"][1]["row_count"])
        self.assertEqual(3, payload["tables"][1]["col_count"])
        self.assertTrue(payload["tables"][0]["horizontal_segments"])
        self.assertTrue(payload["tables"][0]["vertical_segments"])
        self.assertEqual(
            {"x0", "x1", "top", "bottom", "in_body_bounds"},
            set(payload["tables"][0]["horizontal_segments"][0].keys()),
        )
        self.assertEqual(
            {"x0", "x1", "top", "bottom", "in_body_bounds"},
            set(payload["tables"][0]["vertical_segments"][0].keys()),
        )
        self.assertTrue(payload["tables"][0]["horizontal_groups"])
        self.assertTrue(payload["tables"][0]["vertical_groups"])
        self.assertEqual(
            {"axis", "segments", "merged_segments"},
            set(payload["tables"][0]["horizontal_groups"][0].keys()),
        )
        self.assertEqual(
            {"axis", "segments", "merged_segments"},
            set(payload["tables"][0]["vertical_groups"][0].keys()),
        )

    def test_debug_writes_table_drawing_log(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        pdf_path = root / "sample.pdf"
        create_demo_pdf(pdf_path)

        result = extract_pdf_to_outputs(
            pdf_path=pdf_path,
            out_md_dir=root / "md",
            out_image_dir=root / "images",
            stem="sample",
            debug=True,
        )

        debug_file = result["debug_file"]
        self.assertIsNotNone(debug_file)
        payload = json.loads(debug_file.read_text(encoding="utf-8"))
        with pdfplumber.open(str(pdf_path)) as pdf:
            expected_page_count = len(pdf.pages)
        self.assertEqual(expected_page_count, len(payload["pages"]))
        self.assertEqual(2, payload["pages"][0]["table_count"])
        self.assertEqual(6, payload["pages"][0]["tables"][0]["row_count"])
        self.assertIn("text_debug", payload["pages"][0])
        self.assertIn("raw_lines", payload["pages"][0]["text_debug"])
        self.assertIn("normalized_lines", payload["pages"][0]["text_debug"])
        self.assertGreaterEqual(
            len(payload["pages"][0]["text_debug"]["raw_lines"]),
            len(payload["pages"][0]["text_debug"]["normalized_lines"]),
        )
        edge_debug_file = result["debug_edges_file"]
        self.assertIsNotNone(edge_debug_file)
        edge_payload = json.loads(edge_debug_file.read_text(encoding="utf-8"))
        self.assertEqual(expected_page_count, len(edge_payload["pages"]))
        self.assertIn("all_horizontal_edges", edge_payload["pages"][0])
        self.assertIn("selected_horizontal_edges", edge_payload["pages"][0])
        self.assertIn("all_vertical_edges", edge_payload["pages"][0])
        self.assertIn("selected_vertical_edges", edge_payload["pages"][0])

    def test_third_table_uses_structural_triplet_columns_in_sample_pdf(self) -> None:
        pdf_path = self._build_pdf()
        with pdfplumber.open(str(pdf_path)) as pdf:
            payload = _collect_table_drawing_debug(pdf.pages[2], page_no=3)

        self.assertTrue(any(table["col_count"] == 9 for table in payload["tables"]))

    def test_stage_table_repeats_header_after_page_break(self) -> None:
        pdf_path = self._build_pdf()
        with pdfplumber.open(str(pdf_path)) as pdf:
            texts = [(page.extract_text() or "") for page in pdf.pages]
        header_hits = sum("Stage Team Notes" in text for text in texts)
        self.assertGreaterEqual(header_hits, 2)

    def test_phase_a_label_only_appears_on_first_stage_fragment(self) -> None:
        pdf_path = self._build_pdf()
        with pdfplumber.open(str(pdf_path)) as pdf:
            texts = [(page.extract_text() or "") for page in pdf.pages]
        self.assertIn("Phase A", texts[0])
        self.assertNotIn("Phase A", texts[1])

    def test_stage_table_has_no_left_vertical_border_for_merged_column(self) -> None:
        pdf_path = self._build_pdf()
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages[:4]:
                stage_left_lines = [
                    line
                    for line in page.lines
                    if abs(line["x0"] - 36.0) < 0.2
                    and abs(line["x1"] - 36.0) < 0.2
                    and line["top"] < 720
                    and line["bottom"] > 80
                ]
                self.assertFalse(stage_left_lines, f"unexpected left border on page {page.page_number}")

    def test_long_legal_notes_row_spans_two_pages(self) -> None:
        pdf_path = self._build_pdf()
        with pdfplumber.open(str(pdf_path)) as pdf:
            texts = [(page.extract_text() or "") for page in pdf.pages]

        joined = "\n".join(texts)
        self.assertEqual(3, len(texts))
        self.assertIn("Legal", joined)
        self.assertIn("consent language review", texts[1].lower())
        self.assertIn("policy exception register", texts[2].lower())
        self.assertIn("sign-off archive retention", texts[2].lower())

    def test_demo_pdf_has_rotated_gray_watermark_on_every_page(self) -> None:
        pdf_path = self._build_pdf()
        with pdfplumber.open(str(pdf_path)) as pdf:
            self.assertEqual(len(pdf.pages), 3)
            for page in pdf.pages:
                watermark_chars = [
                    char
                    for char in page.chars
                    if isinstance(char.get("matrix"), tuple)
                    and len(char["matrix"]) >= 2
                    and 53.0 <= math.degrees(math.atan2(char["matrix"][1], char["matrix"][0])) <= 57.0
                    and isinstance(char.get("non_stroking_color"), tuple)
                    and len(char["non_stroking_color"]) >= 3
                    and all(abs(float(c) - 0.92) <= 0.03 for c in char["non_stroking_color"][:3])
                ]
                self.assertTrue(watermark_chars)


if __name__ == "__main__":
    unittest.main()
