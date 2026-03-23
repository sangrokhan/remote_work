from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from extractor.shared import _parse_pages_spec
from extractor.text import (
    _build_body_blocks,
    _detect_body_bounds,
    _extract_body_word_lines,
    _extract_drawing_image_bboxes,
    _is_gray_color,
    _is_non_watermark_obj,
    _is_shape_text_line,
    _normalize_cell_lines,
    _should_merge_paragraph_lines,
)


class TextModuleTests(unittest.TestCase):
    def test_parse_pages_spec_supports_ranges_and_lists(self) -> None:
        self.assertEqual([1, 3, 4, 5, 8], _parse_pages_spec("1,3-5,8"))

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

    def test_should_merge_paragraph_lines_uses_fixed_gap_threshold(self) -> None:
        previous = {"bottom": 132.0}
        close_line = {"top": 136.0}
        far_line = {"top": 139.0}
        self.assertTrue(_should_merge_paragraph_lines(previous, close_line))
        self.assertFalse(_should_merge_paragraph_lines(previous, far_line))

    def test_build_body_blocks_groups_adjacent_paragraph_lines(self) -> None:
        lines = [
            {"text": "Chapter 1: Deep Structure Verification", "top": 90.0, "bottom": 102.0},
            {"text": "This line introduces the uncommon term", "top": 120.0, "bottom": 132.0},
            {"text": "ProtoLexeme expands into explanation", "top": 134.0, "bottom": 146.0},
        ]

        blocks = _build_body_blocks(lines)

        self.assertEqual(["heading", "paragraph"], [block["kind"] for block in blocks])
        self.assertEqual(
            ["This line introduces the uncommon term", "ProtoLexeme expands into explanation"],
            [line["text"] for line in blocks[1]["lines"]],
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

        with patch("extractor.text._filter_page_for_extraction", return_value=filtered_page), patch(
            "extractor.text._detect_body_bounds", return_value=(40.0, 700.0)
        ):
            lines = _extract_body_word_lines(page, header_margin=90.0, footer_margin=40.0)

        self.assertEqual(1, len(lines))
        self.assertTrue(lines[0]["marker_candidate"])
        self.assertEqual(64.0, lines[0]["text_start_x"])

    def test_extract_body_word_lines_marks_shape_text_when_line_overlaps_shape_region(self) -> None:
        filtered_page = SimpleNamespace(
            extract_words=lambda **kwargs: [
                {
                    "text": "Callout",
                    "x0": 48.0,
                    "x1": 92.0,
                    "top": 120.0,
                    "bottom": 132.0,
                    "size": 11.0,
                    "fontname": "Helvetica",
                },
                {
                    "text": "summary",
                    "x0": 96.0,
                    "x1": 144.0,
                    "top": 120.0,
                    "bottom": 132.0,
                    "size": 11.0,
                    "fontname": "Helvetica",
                },
            ],
            chars=[],
        )
        page = SimpleNamespace()

        with patch("extractor.text._filter_page_for_extraction", return_value=filtered_page), patch(
            "extractor.text._detect_body_bounds", return_value=(40.0, 700.0)
        ), patch(
            "extractor.text._shape_text_regions", return_value=[(40.0, 110.0, 540.0, 150.0)]
        ):
            lines = _extract_body_word_lines(page, header_margin=90.0, footer_margin=40.0)

        self.assertEqual(1, len(lines))
        self.assertTrue(lines[0]["is_shape_text"])

    def test_extract_drawing_image_bboxes_selects_largest_curve_as_region(self) -> None:
        page = SimpleNamespace(
            curves=[
                {"object_type": "curve", "x0": 20.0, "top": 60.0, "x1": 220.0, "bottom": 190.0},
                {"object_type": "curve", "x0": 40.0, "top": 80.0, "x1": 180.0, "bottom": 170.0},
            ],
            lines=[
                {"object_type": "line", "x0": 20.0, "top": 50.0, "x1": 220.0, "bottom": 55.0},
                {"object_type": "line", "x0": 20.0, "top": 190.0, "x1": 220.0, "bottom": 195.0},
            ],
            rects=[
                {"object_type": "rect", "x0": 18.0, "top": 58.0, "x1": 222.0, "bottom": 192.0},
            ],
        )

        with patch("extractor.text._detect_body_bounds", return_value=(40.0, 700.0)):
            regions = _extract_drawing_image_bboxes(page=page, header_margin=90.0, footer_margin=40.0)

        self.assertEqual(1, len(regions))
        self.assertEqual((20.0, 60.0, 220.0, 190.0), regions[0])

    def test_extract_drawing_image_bboxes_uses_excluded_bboxes(self) -> None:
        page = SimpleNamespace(
            curves=[
                {"object_type": "curve", "x0": 20.0, "top": 60.0, "x1": 120.0, "bottom": 150.0},
            ],
            lines=[
                {"object_type": "line", "x0": 20.0, "top": 55.0, "x1": 120.0, "bottom": 60.0},
            ],
            rects=[],
        )
        with patch("extractor.text._detect_body_bounds", return_value=(40.0, 700.0)):
            regions = _extract_drawing_image_bboxes(
                page=page,
                header_margin=90.0,
                footer_margin=40.0,
                excluded_bboxes=[(15.0, 55.0, 130.0, 160.0)],
            )

        self.assertEqual([], regions)

    def test_extract_drawing_image_bboxes_empty_when_no_curve(self) -> None:
        page = SimpleNamespace(
            curves=[],
            lines=[
                {"object_type": "line", "x0": 20.0, "top": 60.0, "x1": 220.0, "bottom": 190.0},
            ],
            rects=[
                {"object_type": "rect", "x0": 20.0, "top": 60.0, "x1": 220.0, "bottom": 190.0},
            ],
        )
        with patch("extractor.text._detect_body_bounds", return_value=(40.0, 700.0)):
            regions = _extract_drawing_image_bboxes(page=page, header_margin=90.0, footer_margin=40.0)

        self.assertEqual([], regions)

    def test_is_shape_text_line_returns_false_when_line_is_outside_shape_regions(self) -> None:
        self.assertFalse(
            _is_shape_text_line(
                {"x0": 48.0, "x1": 144.0, "top": 120.0, "bottom": 132.0},
                [(200.0, 200.0, 300.0, 260.0)],
            )
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
            horizontal_edges=[{"x0": 35.0, "x1": 565.0, "top": 742.0}],
            extract_words=lambda **kwargs: [
                {"text": "Chapter", "x0": 36.0, "x1": 110.0, "top": 96.0, "bottom": 114.0, "size": 20.0, "fontname": "Helvetica-Bold"},
                {"text": "3:", "x0": 118.0, "x1": 140.0, "top": 96.0, "bottom": 114.0, "size": 20.0, "fontname": "Helvetica-Bold"},
                {"text": "New", "x0": 148.0, "x1": 190.0, "top": 96.0, "bottom": 114.0, "size": 20.0, "fontname": "Helvetica-Bold"},
                {"text": "Section", "x0": 198.0, "x1": 260.0, "top": 96.0, "bottom": 114.0, "size": 20.0, "fontname": "Helvetica-Bold"},
            ],
        )

        body_top, body_bottom = _detect_body_bounds(page, header_margin=90.0, footer_margin=40.0)

        self.assertEqual(96.0, body_top)
        self.assertEqual(742.0, body_bottom)
