from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from extractor import extract_pdf_to_outputs
from extractor.pipeline import (
    _CrossPageTableCandidate,
    _DocumentOutputState,
    _is_cross_page_continuation_candidate,
    _load_cross_page_candidate,
    _pick_cross_page_anchor,
    _rows_for_continuation_matching,
    _strip_repeated_headers_by_chunk,
    _table_shapes_compatible,
    _to_cross_page_candidate,
)
from extractor.debug import _collect_rotated_text_debug
from extractor.images import _extract_embedded_images
from extractor.tables import _append_output_table


def _build_flow_diagram_pdf(pdf_path: Path) -> tuple[str, tuple[str, ...]]:
    outside_text = "Flow diagram paragraph should remain visible in extracted markdown."
    hidden_text = (
        "HIDDEN_START_NODE",
        "HIDDEN_DECISION",
        "HIDDEN_PROCESS",
    )

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.setFont("Helvetica", 11)
    c.drawString(72, 700, outside_text)
    c.drawString(72, 680, "Visible context text before the embedded flow drawing.")

    c.setLineWidth(1.0)
    c.rect(140.0, 260.0, 300.0, 240.0)
    c.line(160.0, 280.0, 340.0, 480.0)
    c.line(160.0, 480.0, 420.0, 280.0)
    c.bezier(140.0, 390.0, 140.0, 250.0, 440.0, 470.0, 440.0, 270.0)

    c.setFont("Helvetica", 10)
    c.drawString(170.0, 392.0, hidden_text[0])
    c.drawString(300.0, 392.0, hidden_text[1])
    c.drawString(260.0, 352.0, hidden_text[2])
    c.save()

    return outside_text, hidden_text


def _build_heading_pdf(pdf_path: Path) -> tuple[str, str]:
    title = "Sized Heading Title"
    paragraph = "This paragraph should remain normal body text."

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, 700, title)
    c.setFont("Helvetica", 11)
    c.drawString(72, 672, paragraph)
    c.drawString(72, 656, "Continuation line for the body paragraph.")
    c.save()

    return title, paragraph


class PipelineExtractionTests(unittest.TestCase):
    def _table_blocks(self, markdown: str) -> list[str]:
        blocks: list[str] = []
        current: list[str] = []
        for line in markdown.splitlines():
            if line.startswith("[") and line.endswith("]") and "_tables.md - Table " in line:
                if current:
                    blocks.append("\n".join(current))
                current = [line]
                continue
            if current:
                current.append(line)
        if current:
            blocks.append("\n".join(current))
        return blocks

    def test_table_shapes_allow_header_only_continuation_chunk(self) -> None:
        self.assertTrue(_table_shapes_compatible((2, 9), (3, 4)))

    def test_rows_for_continuation_matching_collapses_only_globally_empty_columns(self) -> None:
        rows = [
            ["Column A", "", "Column B", "", "Column C"],
            ["Value A", "", "Value B", "", "Value C"],
        ]

        self.assertEqual(
            [
                ["Column A", "Column B", "Column C"],
                ["Value A", "Value B", "Value C"],
            ],
            _rows_for_continuation_matching(rows),
        )

    def test_cross_page_continuation_candidate_allows_moderate_bottom_gap(self) -> None:
        self.assertTrue(
            _is_cross_page_continuation_candidate(
                bbox=(72.02, 416.49, 525.57, 676.86),
                body_top=66.24,
                body_bottom=730.42,
                continuation_gap=39.8508,
            )
        )

    def test_cross_page_continuation_candidate_ignores_table_top_position_when_bottom_gap_is_within_tolerance(self) -> None:
        self.assertTrue(
            _is_cross_page_continuation_candidate(
                bbox=(72.02, 153.90, 525.57, 674.82),
                body_top=66.24,
                body_bottom=730.42,
                continuation_gap=39.8508,
            )
        )

    def test_pick_cross_page_anchor_rejects_when_vertical_axes_do_not_overlap(self) -> None:
        anchor = _CrossPageTableCandidate(
            table_no=8,
            start_page=2,
            last_page=2,
            bbox=(40.0, 640.0, 540.0, 700.0),
            rows=[["A", "B"], ["one", "two"]],
            shape_signature=(2, 2),
            axes=[120.0, 320.0],
            has_gap_text=False,
            page_height=792.0,
        )

        selected = _pick_cross_page_anchor(
            current_bbox=(560.0, 72.0, 760.0, 128.0),
            current_axes=[580.0, 720.0],
            current_rows=[["X", "Y"], ["three", "four"]],
            current_shape=(2, 2),
            body_top=70.0,
            body_bottom=700.0,
            continuation_gap=40.0,
            region_map={
                2: {"body_top": 70.0, "body_bottom": 700.0, "tables": [], "text": [], "images": [], "notes": []},
                3: {"body_top": 70.0, "body_bottom": 700.0, "tables": [], "text": [], "images": [], "notes": []},
            },
            anchors=[anchor],
            current_page=3,
        )

        self.assertIsNone(selected)

    def test_pick_cross_page_anchor_rejects_x_overlap_without_vertical_axis_overlap(self) -> None:
        anchor = _CrossPageTableCandidate(
            table_no=12,
            start_page=2,
            last_page=2,
            bbox=(40.0, 640.0, 540.0, 700.0),
            rows=[["A", "B"], ["one", "two"]],
            shape_signature=(2, 2),
            axes=[120.0, 320.0],
            has_gap_text=False,
            page_height=792.0,
        )

        with patch("extractor.pipeline._has_cross_page_gap_blocked", return_value=False), patch(
            "extractor.pipeline._continuation_regions_should_merge",
            return_value=True,
        ):
            selected = _pick_cross_page_anchor(
                current_bbox=(60.0, 72.0, 520.0, 128.0),
                current_axes=[180.0, 420.0],
                current_rows=[["X", "Y"], ["three", "four"]],
                current_shape=(2, 2),
                body_top=70.0,
                body_bottom=700.0,
                continuation_gap=40.0,
                region_map={
                    2: {"body_top": 70.0, "body_bottom": 700.0, "tables": [], "text": [], "images": [], "notes": []},
                    3: {"body_top": 70.0, "body_bottom": 700.0, "tables": [], "text": [], "images": [], "notes": []},
                },
                anchors=[anchor],
                current_page=3,
            )

        self.assertIsNone(selected)

    def test_pick_cross_page_anchor_rejects_when_intervening_region_blocks_gap(self) -> None:
        anchor = _CrossPageTableCandidate(
            table_no=9,
            start_page=3,
            last_page=3,
            bbox=(40.0, 640.0, 540.0, 700.0),
            rows=[["A", "B"], ["old", "row"]],
            shape_signature=(2, 2),
            axes=[120.0, 320.0],
            has_gap_text=False,
            page_height=792.0,
        )

        with patch("extractor.pipeline._has_cross_page_gap_blocked", return_value=True), patch(
            "extractor.pipeline._continuation_regions_should_merge",
            return_value=True,
        ):
            selected = _pick_cross_page_anchor(
                current_bbox=(40.0, 72.0, 540.0, 128.0),
                current_axes=[120.0, 320.0],
                current_rows=[["X", "Y"], ["new", "row"]],
                current_shape=(2, 2),
                body_top=70.0,
                body_bottom=700.0,
                continuation_gap=40.0,
                region_map={
                    3: {"body_top": 70.0, "body_bottom": 700.0, "tables": [], "text": [], "images": [], "notes": []},
                    4: {"body_top": 70.0, "body_bottom": 700.0, "tables": [], "text": [], "images": [], "notes": []},
                },
                anchors=[anchor],
                current_page=4,
            )

        self.assertIsNone(selected)

    def test_pipeline_excludes_only_note_anchor_regions_from_image_export(self) -> None:
        pages = [SimpleNamespace(width=612.0, height=792.0, page_index=1, vertical_edges=[], horizontal_edges=[])]
        fake_pdf = SimpleNamespace(pages=pages)
        fake_pdf_context = MagicMock()
        fake_pdf_context.__enter__.return_value = fake_pdf
        fake_pdf_context.__exit__.return_value = False
        captured: dict[str, object] = {}

        note_bbox = (120.0, 200.0, 520.0, 320.0)
        note_anchor = (128.0, 212.0, 146.0, 230.0)

        def fake_extract_embedded_images(**kwargs):
            captured["excluded_regions_by_page"] = kwargs.get("excluded_regions_by_page")
            return []

        with patch("extractor.pipeline.pdfplumber.open", return_value=fake_pdf_context), patch(
            "extractor.pipeline._extract_heading_preview_markdown", return_value=""
        ), patch(
            "extractor.pipeline._collect_note_candidates",
            return_value=[
                {
                    "bbox": note_bbox,
                    "rows": [["A note sentence"]],
                    "is_white_content": False,
                    "is_note_like": True,
                    "note_anchor": note_anchor,
                }
            ],
        ), patch(
            "extractor.pipeline._extract_tables",
            return_value=[],
        ), patch(
            "extractor.pipeline._extract_body_text",
            return_value="",
        ), patch(
            "extractor.pipeline._collect_embedded_image_refs",
            return_value={},
        ), patch(
            "extractor.pipeline._extract_embedded_images",
            side_effect=fake_extract_embedded_images,
        ), patch(
            "extractor.pipeline._extract_drawing_image_bboxes",
            return_value=[],
        ), patch(
            "extractor.pipeline._detect_body_bounds",
            return_value=(70.0, 700.0),
        ), patch(
            "extractor.pipeline._body_text_boxes",
            return_value=[],
        ), patch(
            "extractor.pipeline._vertical_axes_for_bbox",
            return_value=[127.22, 524.85],
        ), patch(
            "extractor.pipeline._has_gap_text_before_bbox",
            return_value=False,
        ), patch(
            "extractor.pipeline._has_gap_text_after_bbox",
            return_value=False,
        ), patch(
            "extractor.pipeline._continuation_regions_should_merge",
            return_value=False,
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                extract_pdf_to_outputs(
                    pdf_path=Path("unused.pdf"),
                    out_md_dir=root / "md",
                    out_image_dir=root / "images",
                    stem="note-anchor-exclusion",
                    page_write=True,
                )

        self.assertEqual({1: [note_anchor]}, captured["excluded_regions_by_page"])

    def test_pipeline_inserts_drawing_image_reference_into_body_flow(self) -> None:
        pages = [SimpleNamespace(width=612.0, height=792.0, page_index=1)]
        fake_pdf = SimpleNamespace(pages=pages)
        fake_pdf_context = MagicMock()
        fake_pdf_context.__enter__.return_value = fake_pdf
        fake_pdf_context.__exit__.return_value = False
        captured_refs: list[str] = []

        def fake_extract_body_text(page, header_margin, footer_margin, excluded_bboxes=(), reference_lines=(), heading_levels=None):
            captured_refs.extend(str(entry.get("text") or "") for entry in reference_lines)
            return ""

        with patch("extractor.pipeline.pdfplumber.open", return_value=fake_pdf_context), patch(
            "extractor.pipeline._extract_heading_preview_markdown", return_value=""
        ), patch(
            "extractor.pipeline._collect_note_candidates",
            return_value=[],
        ), patch(
            "extractor.pipeline._extract_tables",
            return_value=[],
        ), patch(
            "extractor.pipeline._extract_body_text",
            side_effect=fake_extract_body_text,
        ), patch(
            "extractor.pipeline._collect_embedded_image_refs",
            return_value={},
        ), patch(
            "extractor.pipeline._extract_embedded_images",
            return_value=[],
        ), patch(
            "extractor.pipeline._extract_drawing_image_bboxes",
            return_value=[(160.0, 220.0, 420.0, 380.0)],
        ), patch(
            "extractor.pipeline._detect_body_bounds",
            return_value=(70.0, 700.0),
        ), patch(
            "extractor.pipeline._body_text_boxes",
            return_value=[],
        ), patch(
            "extractor.pipeline._vertical_axes_for_bbox",
            return_value=[127.22, 524.85],
        ), patch(
            "extractor.pipeline._has_gap_text_before_bbox",
            return_value=False,
        ), patch(
            "extractor.pipeline._has_gap_text_after_bbox",
            return_value=False,
        ), patch(
            "extractor.pipeline._continuation_regions_should_merge",
            return_value=False,
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                extract_pdf_to_outputs(
                    pdf_path=Path("unused.pdf"),
                    out_md_dir=root / "md",
                    out_image_dir=root / "images",
                    stem="drawing-image-ref",
                    page_write=True,
                )

        self.assertIn("[output_image_1.png]", captured_refs)

    def test_pipeline_collects_note_groups_before_extracting_tables(self) -> None:
        pages = [SimpleNamespace(width=612.0, height=792.0, page_index=1)]
        fake_pdf = SimpleNamespace(pages=pages)
        fake_pdf_context = MagicMock()
        fake_pdf_context.__enter__.return_value = fake_pdf
        fake_pdf_context.__exit__.return_value = False

        call_order: list[str] = []

        def fake_collect_note_candidates(page: object):
            call_order.append("notes")
            return []

        def fake_extract_tables(page: object, force_table: bool = False, strategy_debug=None, **kwargs):
            call_order.append("tables")
            return []

        with patch("extractor.pipeline.pdfplumber.open", return_value=fake_pdf_context), patch(
            "extractor.pipeline._extract_heading_preview_markdown",
            return_value="",
        ), patch(
            "extractor.pipeline._collect_note_candidates",
            side_effect=fake_collect_note_candidates,
        ), patch(
            "extractor.pipeline._extract_tables",
            side_effect=fake_extract_tables,
        ), patch(
            "extractor.pipeline._extract_body_text",
            return_value="",
        ), patch(
            "extractor.pipeline._collect_embedded_image_refs",
            return_value={},
        ), patch(
            "extractor.pipeline._extract_embedded_images",
            return_value=[],
        ), patch(
            "extractor.pipeline._extract_drawing_image_bboxes",
            return_value=[],
        ), patch(
            "extractor.pipeline._detect_body_bounds",
            return_value=(70.0, 700.0),
        ), patch(
            "extractor.pipeline._body_text_boxes",
            return_value=[],
        ), patch(
            "extractor.pipeline._vertical_axes_for_bbox",
            return_value=[127.22, 524.85],
        ), patch(
            "extractor.pipeline._has_gap_text_before_bbox",
            return_value=False,
        ), patch(
            "extractor.pipeline._has_gap_text_after_bbox",
            return_value=False,
        ), patch(
            "extractor.pipeline._continuation_regions_should_merge",
            return_value=False,
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                extract_pdf_to_outputs(
                    pdf_path=Path("unused.pdf"),
                    out_md_dir=root / "md",
                    out_image_dir=root / "images",
                    stem="note-first",
                )

        self.assertEqual(["notes", "tables"], call_order)

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
                SimpleNamespace(width=600.0, height=800.0, horizontal_edges=[], images=[{"name": "p1", "top": 100.0, "bottom": 120.0}]),
                SimpleNamespace(width=600.0, height=800.0, horizontal_edges=[], images=[{"name": "p2", "top": 100.0, "bottom": 120.0}]),
                SimpleNamespace(width=600.0, height=800.0, horizontal_edges=[], images=[{"name": "p3", "top": 100.0, "bottom": 120.0}]),
            ],
        )
        fake_plumber_open = MagicMock()
        fake_plumber_open.__enter__.return_value = fake_plumber_pdf
        fake_plumber_open.__exit__.return_value = False

        with patch("extractor.images.PdfReader", return_value=fake_reader), patch(
            "extractor.images.pdfplumber.open", return_value=fake_plumber_open
        ):
            image_files = _extract_embedded_images(
                pdf_path=Path("ignored.pdf"),
                out_image_dir=out_dir,
                stem="sample",
                pages=[2],
            )

        self.assertEqual(1, len(image_files))
        self.assertEqual("sample_image_1.png", image_files[0].name)

    def test_flow_diagram_text_is_filtered_and_rendered_as_drawing(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        pdf_path = root / "flow_diagram.pdf"
        outside_text, hidden_labels = _build_flow_diagram_pdf(pdf_path)

        result = extract_pdf_to_outputs(
            pdf_path=pdf_path,
            out_md_dir=root / "md",
            out_image_dir=root / "images",
            stem="flow_diagram",
        )

        self.assertIn(outside_text, result["markdown"])
        for hidden_label in hidden_labels:
            self.assertNotIn(hidden_label, result["markdown"])

        image_files = [Path(path) for path in result["image_files"]]
        self.assertEqual(1, len(image_files))
        self.assertTrue(image_files[0].exists())

    def test_add_heading_uses_external_font_size_mapping_for_markdown(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        pdf_path = root / "heading.pdf"
        title, paragraph = _build_heading_pdf(pdf_path)
        heading_json = root / "heading.json"
        heading_json.write_text(
            json.dumps(
                {
                    "heading_rules": [
                        {
                            "match": {"font_size": 20.0},
                            "assign": {"tag": "h1"},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        result = extract_pdf_to_outputs(
            pdf_path=pdf_path,
            out_md_dir=root / "md",
            out_image_dir=root / "images",
            stem="heading",
            add_heading=heading_json,
        )

        self.assertIn(f"# {title}", result["markdown"])
        self.assertIn(paragraph, result["markdown"])
        self.assertNotIn(f"# {paragraph}", result["markdown"])

    def test_missing_h2_uses_output_as_default_document_name(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        pdf_path = root / "no_h2.pdf"
        _title, _paragraph = _build_heading_pdf(pdf_path)
        heading_json = root / "heading.json"
        heading_json.write_text(
            json.dumps(
                {
                    "heading_rules": [
                        {
                            "match": {"font_size": 20.0},
                            "assign": {"tag": "h1"},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        result = extract_pdf_to_outputs(
            pdf_path=pdf_path,
            out_md_dir=root / "md",
            out_image_dir=root / "images",
            stem="ignored-stem",
            add_heading=heading_json,
        )

        self.assertEqual("output.md", Path(result["md_file"]).name)
        self.assertEqual("output_tables.md", Path(result["table_md_file"]).name)

    def test_strip_repeated_headers_by_chunk(self) -> None:
        chunks = [
            [["Col1", "Col2"], ["A", "1"], ["B", "2"]],
            [["Col1", "Col2"], ["C", "3"]],
            [["D", "E"], ["F", "G"]],
            [["D", "E"], ["H", "I"]],
        ]
        self.assertEqual(
            [
                ["Col1", "Col2"],
                ["A", "1"],
                ["B", "2"],
                ["C", "3"],
                ["D", "E"],
                ["F", "G"],
                ["H", "I"],
            ],
            _strip_repeated_headers_by_chunk(chunks),
        )

    def test_pick_cross_page_anchor_prefers_latest_lower_anchor_on_score_tie(self) -> None:
        anchors = [
            _CrossPageTableCandidate(
                table_no=10,
                start_page=2,
                last_page=2,
                bbox=(40.0, 610.0, 540.0, 660.0),
                rows=[["Col1", "Col2"], ["A", "1"]],
                shape_signature=(2, 2),
                axes=[40.0, 540.0],
                has_gap_text=False,
                page_height=792.0,
            ),
            _CrossPageTableCandidate(
                table_no=11,
                start_page=2,
                last_page=2,
                bbox=(40.0, 640.0, 540.0, 690.0),
                rows=[["Col1", "Col2"], ["B", "2"]],
                shape_signature=(2, 2),
                axes=[40.0, 540.0],
                has_gap_text=False,
                page_height=792.0,
            ),
        ]

        with patch("extractor.pipeline._has_cross_page_gap_blocked", return_value=False), patch(
            "extractor.pipeline._continuation_regions_should_merge",
            return_value=True,
        ):
            selected = _pick_cross_page_anchor(
                current_bbox=(40.0, 70.0, 540.0, 120.0),
                current_axes=[40.0, 540.0],
                current_rows=[["Col1", "Col2"], ["C", "3"]],
                current_shape=(2, 2),
                body_top=70.0,
                body_bottom=700.0,
                continuation_gap=40.0,
                region_map={
                    2: {"body_top": 70.0, "body_bottom": 700.0, "tables": [], "text": [], "images": [], "notes": []},
                    3: {"body_top": 70.0, "body_bottom": 700.0, "tables": [], "text": [], "images": [], "notes": []},
                },
                anchors=anchors,
                current_page=3,
            )

        self.assertIsNotNone(selected)
        self.assertEqual(11, selected.table_no)

    def test_cross_page_merge_only_targets_first_table_on_next_page(self) -> None:
        pages = [SimpleNamespace(width=612.0, height=792.0, page_index=1), SimpleNamespace(width=612.0, height=792.0, page_index=2)]
        fake_pdf = SimpleNamespace(pages=pages)
        fake_pdf_context = MagicMock()
        fake_pdf_context.__enter__.return_value = fake_pdf
        fake_pdf_context.__exit__.return_value = False

        def fake_extract_tables(page: object, force_table: bool = False, strategy_debug=None, **kwargs):
            if page.page_index == 1:
                return [
                    ([["Col1", "Col2"], ["A", "1"], ["B", "2"]], (40.0, 650.0, 540.0, 700.0)),
                ]
            return [
                ([["Col1", "Col2"], ["C", "3"]], (40.0, 30.0, 540.0, 75.0)),
                ([["Other", "Value"], ["X", "Y"]], (40.0, 220.0, 540.0, 260.0)),
            ]

        with patch("extractor.pipeline.pdfplumber.open", return_value=fake_pdf_context), patch(
            "extractor.pipeline._extract_tables", side_effect=fake_extract_tables
        ), patch("extractor.pipeline._detect_body_bounds", return_value=(70.0, 700.0)), patch(
            "extractor.pipeline._extract_drawing_image_bboxes", return_value=[]
        ), patch("extractor.pipeline._body_text_boxes", return_value=[]), patch(
            "extractor.pipeline._has_gap_text_before_bbox", return_value=False
        ), patch("extractor.pipeline._has_gap_text_after_bbox", return_value=False), patch(
            "extractor.pipeline._vertical_axes_for_bbox", return_value=[40.0, 540.0]
        ), patch("extractor.pipeline._continuation_regions_should_merge", return_value=True), patch(
            "extractor.pipeline._collect_embedded_image_refs", return_value={}
        ), patch(
            "extractor.pipeline._extract_body_text", return_value=""
        ), patch(
            "extractor.pipeline._extract_embedded_images",
            return_value=[],
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                result = extract_pdf_to_outputs(
                    pdf_path=Path("unused.pdf"),
                    out_md_dir=root / "md",
                    out_image_dir=root / "images",
                    stem="first-table-only",
                    page_write=True,
                )

        table_markdown = result["table_markdown"]
        blocks = self._table_blocks(table_markdown)
        self.assertEqual(2, len(blocks), table_markdown)
        self.assertIsNotNone(re.search(r"^\[[A-Za-z0-9._-]+_tables\.md - Table 1\]$", table_markdown, flags=re.MULTILINE))
        self.assertIsNotNone(re.search(r"^\[[A-Za-z0-9._-]+_tables\.md - Table 2\]$", table_markdown, flags=re.MULTILINE))
        self.assertIsNone(re.search(r"^\[[A-Za-z0-9._-]+_tables\.md - Table 3\]$", table_markdown, flags=re.MULTILINE))
        self.assertEqual(1, blocks[0].count("| Col1 | Col2 |"))

    def test_cross_page_merge_blocks_when_intervening_region_exists(self) -> None:
        pages = [SimpleNamespace(width=612.0, height=792.0, page_index=1), SimpleNamespace(width=612.0, height=792.0, page_index=2)]
        fake_pdf = SimpleNamespace(pages=pages)
        fake_pdf_context = MagicMock()
        fake_pdf_context.__enter__.return_value = fake_pdf
        fake_pdf_context.__exit__.return_value = False

        def fake_extract_tables(page: object, force_table: bool = False, strategy_debug=None, **kwargs):
            if page.page_index == 1:
                return [
                    ([["Col1", "Col2"], ["A", "1"]], (40.0, 650.0, 540.0, 680.0)),
                ]
            return [
                ([["Col1", "Col2"], ["B", "2"]], (40.0, 90.0, 540.0, 135.0)),
            ]

        def fake_drawing_image_bboxes(
            page: object,
            header_margin: float,
            footer_margin: float,
            excluded_bboxes=(),
        ):
            if page.page_index == 1:
                return [(40.0, 688.0, 540.0, 698.0)]
            return []

        with patch("extractor.pipeline.pdfplumber.open", return_value=fake_pdf_context), patch(
            "extractor.pipeline._extract_tables", side_effect=fake_extract_tables
        ), patch("extractor.pipeline._detect_body_bounds", return_value=(70.0, 700.0)), patch(
            "extractor.pipeline._extract_drawing_image_bboxes", side_effect=fake_drawing_image_bboxes
        ), patch("extractor.pipeline._body_text_boxes", return_value=[]), patch(
            "extractor.pipeline._has_gap_text_before_bbox", return_value=False
        ), patch("extractor.pipeline._has_gap_text_after_bbox", return_value=False), patch(
            "extractor.pipeline._vertical_axes_for_bbox", return_value=[40.0, 540.0]
        ), patch("extractor.pipeline._continuation_regions_should_merge", return_value=True), patch(
            "extractor.pipeline._collect_embedded_image_refs", return_value={}
        ), patch(
            "extractor.pipeline._extract_body_text", return_value=""
        ), patch(
            "extractor.pipeline._extract_embedded_images",
            return_value=[],
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                result = extract_pdf_to_outputs(
                    pdf_path=Path("unused.pdf"),
                    out_md_dir=root / "md",
                    out_image_dir=root / "images",
                    stem="first-table-gap-blocked",
                    page_write=True,
                )

        table_markdown = result["table_markdown"]
        blocks = self._table_blocks(table_markdown)
        self.assertEqual(2, len(blocks), table_markdown)
        self.assertIsNotNone(re.search(r"^\[[A-Za-z0-9._-]+_tables\.md - Table 1\]$", table_markdown, flags=re.MULTILINE))
        self.assertIsNotNone(re.search(r"^\[[A-Za-z0-9._-]+_tables\.md - Table 2\]$", table_markdown, flags=re.MULTILINE))

    def test_cross_page_merge_does_not_reuse_stale_mid_page_anchor(self) -> None:
        pages = [
            SimpleNamespace(width=612.0, height=792.0, page_index=1),
            SimpleNamespace(width=612.0, height=792.0, page_index=2),
            SimpleNamespace(width=612.0, height=792.0, page_index=3),
        ]
        fake_pdf = SimpleNamespace(pages=pages)
        fake_pdf_context = MagicMock()
        fake_pdf_context.__enter__.return_value = fake_pdf
        fake_pdf_context.__exit__.return_value = False

        def fake_extract_tables(page: object, force_table: bool = False, strategy_debug=None, **kwargs):
            if page.page_index == 1:
                return [
                    ([["Col1", "Col2"], ["A1", "1"]], (40.0, 650.0, 540.0, 700.0)),
                ]
            if page.page_index == 2:
                return [
                    ([["Col1", "Col2"], ["A2", "2"]], (40.0, 30.0, 540.0, 75.0)),
                    ([["Col1", "Col2"], ["B1", "10"]], (40.0, 620.0, 540.0, 700.0)),
                ]
            return [
                ([["Col1", "Col2"], ["B2", "11"]], (40.0, 30.0, 540.0, 75.0)),
            ]

        with patch("extractor.pipeline.pdfplumber.open", return_value=fake_pdf_context), patch(
            "extractor.pipeline._extract_tables", side_effect=fake_extract_tables
        ), patch("extractor.pipeline._detect_body_bounds", return_value=(70.0, 700.0)), patch(
            "extractor.pipeline._extract_drawing_image_bboxes", return_value=[]
        ), patch("extractor.pipeline._body_text_boxes", return_value=[]), patch(
            "extractor.pipeline._has_gap_text_before_bbox", return_value=False
        ), patch("extractor.pipeline._has_gap_text_after_bbox", return_value=False), patch(
            "extractor.pipeline._vertical_axes_for_bbox", return_value=[40.0, 540.0]
        ), patch("extractor.pipeline._continuation_regions_should_merge", return_value=True), patch(
            "extractor.pipeline._collect_embedded_image_refs", return_value={}
        ), patch(
            "extractor.pipeline._extract_body_text", return_value=""
        ), patch(
            "extractor.pipeline._extract_embedded_images",
            return_value=[],
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                result = extract_pdf_to_outputs(
                    pdf_path=Path("unused.pdf"),
                    out_md_dir=root / "md",
                    out_image_dir=root / "images",
                    stem="stale-anchor",
                    page_write=True,
                )

        table_markdown = result["table_markdown"]
        blocks = self._table_blocks(table_markdown)
        self.assertEqual(2, len(blocks), table_markdown)
        self.assertIsNotNone(re.search(r"^\[[A-Za-z0-9._-]+_tables\.md - Table 1\]$", table_markdown, flags=re.MULTILINE))
        self.assertIsNotNone(re.search(r"^\[[A-Za-z0-9._-]+_tables\.md - Table 2\]$", table_markdown, flags=re.MULTILINE))
        self.assertEqual(
            1,
            len(re.findall(r"^\[[A-Za-z0-9._-]+_tables\.md - Table 1\]$", table_markdown, flags=re.MULTILINE)),
            table_markdown,
        )
        self.assertNotIn("| B2 | 11 |", blocks[0], table_markdown)
        self.assertIn("| B2 | 11 |", blocks[1], table_markdown)

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
