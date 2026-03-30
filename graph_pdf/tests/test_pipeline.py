from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from extractor import extract_pdf_to_outputs
from extractor.pipeline import (
    _CrossPageTableCandidate,
    _DocumentOutputState,
    _is_cross_page_continuation_candidate,
    _looks_like_cross_page_continuation_rows,
    _load_cross_page_candidate,
    _strip_repeated_headers_by_chunk,
    _table_shapes_compatible,
)
from extractor.debug import _collect_rotated_text_debug, _collect_table_drawing_debug
from extractor.images import _extract_embedded_images
from extractor.tables import _append_output_table
from sample_fixture import load_demo_fixture
from sample_generator import create_demo_pdf
from verify import _extract_markdown_tables


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


def _build_document_id_heading_pdf(pdf_path: Path) -> tuple[str, str]:
    title = "FGR-TEST01, Example Feature"
    paragraph = "Document id heading should drive output naming."

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, 700, title)
    c.setFont("Helvetica", 11)
    c.drawString(72, 672, paragraph)
    c.save()

    return title, paragraph


class PipelineExtractionTests(unittest.TestCase):
    def _build_pdf(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pdf_path = Path(tmp.name) / "sample.pdf"
        create_demo_pdf(pdf_path)
        return pdf_path

    def _extract_result(self, *, page_write: bool = False) -> dict:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        pdf_path = root / "sample.pdf"
        create_demo_pdf(pdf_path)
        return extract_pdf_to_outputs(
            pdf_path=pdf_path,
            out_md_dir=root / "md",
            out_image_dir=root / "images",
            stem="sample",
            page_write=page_write,
        )

    def _extract_table_markdown(self) -> str:
        return self._extract_result()["table_markdown"]

    def _table_blocks(self, markdown: str) -> list[str]:
        blocks: list[str] = []
        current: list[str] = []
        for line in markdown.splitlines():
            if line.startswith("[//]: # (") and " - Table " in line and line.endswith(")"):
                if current:
                    blocks.append("\n".join(current))
                current = [line]
                continue
            if current:
                current.append(line)
        if current:
            blocks.append("\n".join(current))
        return blocks

    def test_fixture_roundtrip_matches_expected_tables(self) -> None:
        fixture = load_demo_fixture()
        markdown = self._extract_table_markdown()
        extracted_tables = _extract_markdown_tables(markdown)
        extracted_by_index = {idx: rows for idx, rows in enumerate(extracted_tables)}
        expected_tables = [table for table in fixture["tables"] if table.get("id") != "callout"]

        for idx, table in enumerate(expected_tables):
            self.assertIn(idx, extracted_by_index)
            self.assertEqual(table["rows"], extracted_by_index[idx], table["id"])

    def test_table_shapes_allow_header_only_continuation_chunk(self) -> None:
        self.assertTrue(_table_shapes_compatible((2, 9), (3, 4)))

    def test_cross_page_continuation_candidate_allows_moderate_bottom_gap(self) -> None:
        self.assertTrue(
            _is_cross_page_continuation_candidate(
                bbox=(72.02, 416.49, 525.57, 676.86),
                body_top=66.24,
                body_bottom=730.42,
                continuation_gap=39.8508,
            )
        )

    def test_cross_page_continuation_candidate_rejects_high_on_page_table(self) -> None:
        self.assertFalse(
            _is_cross_page_continuation_candidate(
                bbox=(72.02, 153.90, 525.57, 674.82),
                body_top=66.24,
                body_bottom=730.42,
                continuation_gap=39.8508,
            )
        )

    def test_cross_page_continuation_rows_accept_type_description_only_chunk(self) -> None:
        previous_rows = [
            ["", "Family Displa", "y Name", "", "", "Type Name", "", "", "Type Description", ""],
            ["", "F1-U UL Inte UP per UP", "rface collected in", "", "", "F1UPacketLossCntUL", "", "", "desc", ""],
        ]
        current_rows = [
            ["", "", "F1UPacketOosCntDL_QCI", "desc"],
            ["", "", "F1UPacketCntDL_QCI", "desc"],
        ]
        self.assertTrue(_looks_like_cross_page_continuation_rows(previous_rows, current_rows))

    def test_cross_page_continuation_rows_reject_new_family_section_chunk(self) -> None:
        previous_rows = [
            ["", "Family Display Name", "", "", "Type Name", "", "", "Type Description", ""],
            ["", "S1-U Interface collected in UP per sGW IP per QCI", "", "", "S1URxPacketLossCnt", "", "", "desc", ""],
        ]
        current_rows = [
            ["", "Family Displa", "y Name", "", "", "Type Name", "", "", "Type Description", ""],
            ["", "N3 Interface", "per UPF IP", "", "", "N3RxPacketLossCnt", "", "", "desc", ""],
        ]
        self.assertFalse(_looks_like_cross_page_continuation_rows(previous_rows, current_rows))

    def test_cross_page_continuation_rows_accept_repeated_header_collected_family_chunk(self) -> None:
        previous_rows = [
            ["", "Family Display Name", "", "", "Type Name", "", "", "Type Description", ""],
            ["", "S1-U Interface collected in UP per sGW IP per QCI", "", "", "S1URxPacketLossCnt", "", "", "desc", ""],
        ]
        current_rows = [
            ["", "Family Displa", "y Name", "", "", "Type Name", "", "", "Type Description", ""],
            ["", "F1-U Interfa per gNB-DU", "ce collected in UP per QCI", "", "", "F1URxPacketLossCnt", "", "", "desc", ""],
        ]
        self.assertTrue(_looks_like_cross_page_continuation_rows(previous_rows, current_rows))

    def test_table_output_uses_markdown_tables(self) -> None:
        markdown = self._extract_table_markdown()
        self.assertIsNotNone(
            re.search(r"^\[//\]: # \(.+ - Table 1\)$", markdown, flags=re.MULTILINE),
            "table header is missing",
        )
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
        self.assertIn("| Docs | READY | Finalize<br>- sample<br>- archive |", markdown)

    def test_stage_table_uses_collapsed_multirow_header_in_markdown_output(self) -> None:
        markdown = self._extract_table_markdown()
        self.assertIn(
            "| Stage<br>Group | Team<br>Function | Notes<br>Deliverable |",
            markdown,
        )

    def test_orphan_table_header_line_is_owned_by_table_not_body_text(self) -> None:
        pages = [SimpleNamespace(width=612.0, height=792.0, page_index=1)]
        fake_pdf = SimpleNamespace(pages=pages)
        fake_pdf_context = MagicMock()
        fake_pdf_context.__enter__.return_value = fake_pdf
        fake_pdf_context.__exit__.return_value = False

        orphan_header = "Family Display Name Type Name Type Description"

        def fake_extract_body_text(page, header_margin, footer_margin, excluded_bboxes=(), reference_lines=(), heading_levels=None):
            return f"{orphan_header}\nVisible body line"

        with patch("extractor.pipeline.pdfplumber.open", return_value=fake_pdf_context), patch(
            "extractor.pipeline._extract_heading_preview_markdown",
            return_value="",
        ), patch(
            "extractor.pipeline._collect_note_candidates",
            return_value=[],
        ), patch(
            "extractor.pipeline._extract_tables",
            return_value=[
                (
                    [[orphan_header], ["Counter row"]],
                    (40.0, 120.0, 540.0, 180.0),
                )
            ],
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
            return_value=[],
        ), patch(
            "extractor.pipeline._detect_body_bounds",
            return_value=(70.0, 700.0),
        ), patch(
            "extractor.pipeline._body_text_boxes",
            return_value=[],
        ), patch(
            "extractor.pipeline._vertical_axes_for_bbox",
            return_value=[40.0, 540.0],
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
                result = extract_pdf_to_outputs(
                    pdf_path=Path("unused.pdf"),
                    out_md_dir=root / "md",
                    out_image_dir=root / "images",
                    stem="orphan-header",
                )

        self.assertIn(orphan_header, result["table_markdown"])
        self.assertNotIn(orphan_header, result["markdown"])

    def test_pipeline_excludes_only_note_anchor_regions_from_image_export(self) -> None:
        pages = [SimpleNamespace(width=612.0, height=792.0, page_index=1)]
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
            pages=[1],
            page_write=True,
        )

        self.assertIn("[//]: # (Page 1)", result["markdown"])
        self.assertNotIn("[//]: # (Page 3)", result["markdown"])
        self.assertIsNotNone(re.search(r"^\[//\]: # \(.+ - Table 1\)$", result["table_markdown"], flags=re.MULTILINE))
        self.assertEqual(2, result["summary"]["table_count"])
        self.assertEqual(1, len(result["image_files"]))

    def test_markdown_includes_table_references_for_detected_tables(self) -> None:
        markdown = self._extract_result(page_write=True)["markdown"]

        self.assertRegex(markdown, r"\[[A-Za-z0-9._-]+_tables\.md - Table 1\]")
        self.assertEqual(1, len(re.findall(r"\[[A-Za-z0-9._-]+_tables\.md - Table 2\]", markdown)))
        self.assertRegex(markdown, r"\[[A-Za-z0-9._-]+_tables\.md - Table 3\]")
        self.assertIn("[//]: # (Page 1)", markdown)
        self.assertIn("[//]: # (Page 4)", markdown)

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

    def test_default_heading_profile_applies_to_direct_pipeline_calls(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        pdf_path = root / "docid.pdf"
        title, paragraph = _build_document_id_heading_pdf(pdf_path)
        heading_json = root / "heading.json"
        heading_json.write_text(
            json.dumps(
                {
                    "heading_rules": [
                        {
                            "match": {"font_size": 20.0},
                            "assign": {"tag": "h2"},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        with patch("extractor.pipeline.DEFAULT_ADD_HEADING", heading_json):
            result = extract_pdf_to_outputs(
                pdf_path=pdf_path,
                out_md_dir=root / "md",
                out_image_dir=root / "images",
                stem="docid",
            )

        self.assertEqual("FGR-TEST01.md", Path(result["md_file"]).name)
        self.assertEqual("FGR-TEST01_tables.md", Path(result["table_md_file"]).name)
        self.assertFalse((root / "md" / "docid.md").exists())
        self.assertIn(f"## {title}", result["markdown"])
        self.assertIn(paragraph, result["markdown"])

    def test_h2_document_id_heading_drives_output_file_names_without_stem_fallback(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        pdf_path = root / "docid.pdf"
        title, paragraph = _build_document_id_heading_pdf(pdf_path)
        heading_json = root / "heading.json"
        heading_json.write_text(
            json.dumps(
                {
                    "heading_rules": [
                        {
                            "match": {"font_size": 20.0},
                            "assign": {"tag": "h2"},
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
            stem="docid",
            add_heading=heading_json,
        )

        self.assertEqual("FGR-TEST01.md", Path(result["md_file"]).name)
        self.assertEqual("FGR-TEST01_tables.md", Path(result["table_md_file"]).name)
        self.assertFalse((root / "md" / "docid.md").exists())
        self.assertIn(f"## {title}", result["markdown"])
        self.assertIn(paragraph, result["markdown"])

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

    def test_pre_h2_content_stays_in_output_until_h2_is_actually_seen(self) -> None:
        pages = [
            SimpleNamespace(width=612.0, height=792.0, page_index=1),
            SimpleNamespace(width=612.0, height=792.0, page_index=2),
        ]
        fake_pdf = SimpleNamespace(pages=pages)
        fake_pdf_context = MagicMock()
        fake_pdf_context.__enter__.return_value = fake_pdf
        fake_pdf_context.__exit__.return_value = False

        def fake_extract_heading_preview_markdown(page, header_margin, footer_margin, heading_levels=None):
            if page.page_index == 1:
                return ""
            return "## FGR-DOC002 Second document"

        def fake_extract_body_text(page, header_margin, footer_margin, excluded_bboxes=(), reference_lines=(), heading_levels=None):
            if page.page_index == 1:
                return "Prelude body before first h2"
            return "## FGR-DOC002 Second document\nDoc 2 body"

        with patch("extractor.pipeline.pdfplumber.open", return_value=fake_pdf_context), patch(
            "extractor.pipeline._extract_heading_preview_markdown", side_effect=fake_extract_heading_preview_markdown
        ), patch(
            "extractor.pipeline._extract_body_text", side_effect=fake_extract_body_text
        ), patch("extractor.pipeline._extract_tables", return_value=[]), patch(
            "extractor.pipeline._collect_embedded_image_refs", return_value={}
        ), patch(
            "extractor.pipeline._extract_embedded_images", return_value=[]
        ), patch(
            "extractor.pipeline._extract_drawing_image_bboxes", return_value=[]
        ), patch(
            "extractor.pipeline._detect_body_bounds", return_value=(70.0, 700.0)
        ), patch(
            "extractor.pipeline._body_text_boxes", return_value=[]
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                result = extract_pdf_to_outputs(
                    pdf_path=Path("unused.pdf"),
                    out_md_dir=root / "md",
                    out_image_dir=root / "images",
                    stem="pre-h2",
                )

        documents = result["documents"]
        self.assertEqual(["output", "FGR-DOC002"], [doc["document_id"] for doc in documents])
        self.assertIn("Prelude body before first h2", documents[0]["markdown"])
        self.assertNotIn("Prelude body before first h2", documents[1]["markdown"])
        self.assertEqual("output.md", Path(documents[0]["md_file"]).name)
        self.assertEqual("FGR-DOC002.md", Path(documents[1]["md_file"]).name)

    def test_document_images_are_written_under_doc_id_images_directory(self) -> None:
        result = self._extract_result()
        image_paths = [Path(path) for path in result["image_files"]]

        self.assertTrue(image_paths)
        for image_path in image_paths:
            self.assertEqual(f"{image_path.stem.split('_image_')[0]}_images", image_path.parent.name)

    def test_chapter_page_is_moved_to_next_document_start(self) -> None:
        pages = [
            SimpleNamespace(width=612.0, height=792.0, page_index=1),
            SimpleNamespace(width=612.0, height=792.0, page_index=2),
            SimpleNamespace(width=612.0, height=792.0, page_index=3),
        ]
        fake_pdf = SimpleNamespace(pages=pages)
        fake_pdf_context = MagicMock()
        fake_pdf_context.__enter__.return_value = fake_pdf
        fake_pdf_context.__exit__.return_value = False

        def fake_extract_body_text(page, header_margin, footer_margin, excluded_bboxes=(), reference_lines=(), heading_levels=None):
            if page.page_index == 1:
                return "## FGR-DOC001 First document\nDoc 1 body"
            if page.page_index == 2:
                return "# Chapter 2\nChapter bridge text"
            return "## FGR-DOC002 Second document\nDoc 2 body"

        with patch("extractor.pipeline.pdfplumber.open", return_value=fake_pdf_context), patch(
            "extractor.pipeline._extract_body_text", side_effect=fake_extract_body_text
        ), patch("extractor.pipeline._extract_tables", return_value=[]), patch(
            "extractor.pipeline._collect_embedded_image_refs", return_value={}
        ), patch(
            "extractor.pipeline._extract_embedded_images", return_value=[]
        ), patch(
            "extractor.pipeline._extract_drawing_image_bboxes", return_value=[]
        ), patch(
            "extractor.pipeline._detect_body_bounds", return_value=(70.0, 700.0)
        ), patch(
            "extractor.pipeline._body_text_boxes", return_value=[]
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                result = extract_pdf_to_outputs(
                    pdf_path=Path("unused.pdf"),
                    out_md_dir=root / "md",
                    out_image_dir=root / "images",
                    stem="chapter-shift",
                )

        documents = result["documents"]
        self.assertEqual(2, len(documents))
        self.assertNotIn("Chapter bridge text", documents[0]["markdown"])
        self.assertIn("Chapter bridge text", documents[1]["markdown"])

    def test_pipeline_does_not_use_body_text_preview_pass_for_document_switching(self) -> None:
        pages = [SimpleNamespace(width=612.0, height=792.0, page_index=1)]
        fake_pdf = SimpleNamespace(pages=pages)
        fake_pdf_context = MagicMock()
        fake_pdf_context.__enter__.return_value = fake_pdf
        fake_pdf_context.__exit__.return_value = False

        extract_body_text_calls: list[int] = []

        def fake_extract_body_text(page, header_margin, footer_margin, excluded_bboxes=(), reference_lines=(), heading_levels=None):
            extract_body_text_calls.append(page.page_index)
            return "Plain body text"

        with patch("extractor.pipeline.pdfplumber.open", return_value=fake_pdf_context), patch(
            "extractor.pipeline._extract_body_text", side_effect=fake_extract_body_text
        ), patch(
            "extractor.pipeline._extract_body_text_lines",
            return_value=(["FGR-DOC001 First document"], ["## FGR-DOC001 First document"]),
        ), patch("extractor.pipeline._extract_tables", return_value=[]), patch(
            "extractor.pipeline._collect_embedded_image_refs", return_value={}
        ), patch(
            "extractor.pipeline._extract_embedded_images", return_value=[]
        ), patch(
            "extractor.pipeline._extract_drawing_image_bboxes", return_value=[]
        ), patch(
            "extractor.pipeline._detect_body_bounds", return_value=(70.0, 700.0)
        ), patch(
            "extractor.pipeline._body_text_boxes", return_value=[]
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                extract_pdf_to_outputs(
                    pdf_path=Path("unused.pdf"),
                    out_md_dir=root / "md",
                    out_image_dir=root / "images",
                    stem="single-pass",
                )

        self.assertEqual([1], extract_body_text_calls)

    def test_spanning_stage_table_merges_into_one_block(self) -> None:
        markdown = self._extract_table_markdown()
        blocks = self._table_blocks(markdown)
        self.assertEqual(3, len(blocks))
        stage_block = next((block for block in blocks if "Phase A" in block), "")
        self.assertTrue(stage_block)
        self.assertIn("Release Notes", stage_block)
        self.assertIn("Phase C", stage_block)

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

    def test_strip_repeated_headers_by_chunk_normalizes_split_header_variants(self) -> None:
        chunks = [
            [
                ["", "Family Display Name", "", "", "Type Name", "", "", "Type Description", ""],
                ["", "S1-U Interface collected in UP per sGW IP per QCI", "", "", "S1URxPacketLossCnt", "", "", "desc", ""],
            ],
            [
                ["", "Family Displa", "y Name", "", "", "Type Name", "", "", "Type Description", ""],
                ["", "F1-U Interfa per gNB-DU", "ce collected in UP per QCI", "", "", "F1URxPacketLossCnt", "", "", "desc", ""],
            ],
        ]
        self.assertEqual(
            [
                ["", "Family Display Name", "", "", "Type Name", "", "", "Type Description", ""],
                ["", "S1-U Interface collected in UP per sGW IP per QCI", "", "", "S1URxPacketLossCnt", "", "", "desc", ""],
                ["", "F1-U Interfa per gNB-DU", "ce collected in UP per QCI", "", "", "F1URxPacketLossCnt", "", "", "desc", ""],
            ],
            _strip_repeated_headers_by_chunk(chunks),
        )

    def test_load_cross_page_candidate_preserves_chunk_nesting(self) -> None:
        state = _DocumentOutputState(document_id="demo")
        candidate = _CrossPageTableCandidate(
            table_no=14,
            start_page=12,
            last_page=13,
            bbox=(72.0, 80.0, 525.0, 249.0),
            rows=[
                ["Family Display Name", "Type Name", "Type Description"],
                ["S1-U Interface", "S1URxPacketLossCnt", "Lost packets"],
            ],
            shape_signature=(2, 3),
            axes=[72.0, 240.0, 525.0],
            has_gap_text=False,
            page_height=792.0,
        )

        _load_cross_page_candidate(state, candidate)

        self.assertEqual([candidate.rows], state.pending_table_state.chunks)
        self.assertEqual(candidate.rows, state.pending_table_state.flattened_rows())

    def test_reloaded_cross_page_candidate_outputs_normal_table_rows(self) -> None:
        state = _DocumentOutputState(document_id="demo")
        candidate = _CrossPageTableCandidate(
            table_no=14,
            start_page=12,
            last_page=13,
            bbox=(72.0, 80.0, 525.0, 249.0),
            rows=[
                ["Family Display Name", "Type Name", "Type Description"],
                ["S1-U Interface", "S1URxPacketLossCnt", "Lost packets"],
            ],
            shape_signature=(2, 3),
            axes=[72.0, 240.0, 525.0],
            has_gap_text=False,
            page_height=792.0,
        )

        _load_cross_page_candidate(state, candidate)
        output_tables: list[str] = []
        _append_output_table(
            output_tables,
            state.document_id,
            int(state.pending_table_state.table_no or 0),
            state.pending_table_state.flattened_rows(),
        )

        table_markdown = "\n".join(output_tables)
        self.assertIn("| Family Display Name | Type Name | Type Description |", table_markdown)
        self.assertIn("| S1-U Interface | S1URxPacketLossCnt | Lost packets |", table_markdown)
        self.assertNotIn("| F | a | m |", table_markdown)

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
        self.assertIsNotNone(re.search(r"^\[//\]: # \(.+ - Table 1\)$", table_markdown, flags=re.MULTILINE))
        self.assertIsNotNone(re.search(r"^\[//\]: # \(.+ - Table 2\)$", table_markdown, flags=re.MULTILINE))
        self.assertIsNone(re.search(r"^\[//\]: # \(.+ - Table 3\)$", table_markdown, flags=re.MULTILINE))
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
        self.assertIsNotNone(re.search(r"^\[//\]: # \(.+ - Table 1\)$", table_markdown, flags=re.MULTILINE))
        self.assertIsNotNone(re.search(r"^\[//\]: # \(.+ - Table 2\)$", table_markdown, flags=re.MULTILINE))

    def test_pipeline_excludes_table_owned_orphan_header_lines_from_body_text(self) -> None:
        pages = [SimpleNamespace(width=612.0, height=792.0, page_index=1)]
        fake_pdf = SimpleNamespace(pages=pages)
        fake_pdf_context = MagicMock()
        fake_pdf_context.__enter__.return_value = fake_pdf
        fake_pdf_context.__exit__.return_value = False
        orphan_bbox = (77.42, 652.9, 405.88, 661.9)

        def fake_extract_body_text(page, header_margin, footer_margin, excluded_bboxes=(), reference_lines=(), heading_levels=None):
            self.assertIn(orphan_bbox, excluded_bboxes)
            return ""

        with patch("extractor.pipeline.pdfplumber.open", return_value=fake_pdf_context), patch(
            "extractor.pipeline._extract_heading_preview_markdown", return_value=""
        ), patch(
            "extractor.pipeline._extract_tables",
            return_value=[
                ([["Family Display Name", "Type Name", "Type Description"], ["Docs", "READY", "Finalize"]], (72.0, 668.0, 525.0, 720.0)),
            ],
        ), patch("extractor.pipeline._detect_body_bounds", return_value=(70.0, 700.0)), patch(
            "extractor.pipeline._extract_drawing_image_bboxes", return_value=[]
        ), patch("extractor.pipeline._body_text_boxes", return_value=[]), patch(
            "extractor.pipeline._table_owned_body_line_bboxes",
            return_value=[orphan_bbox],
        ), patch("extractor.pipeline._has_gap_text_before_bbox", return_value=False), patch(
            "extractor.pipeline._has_gap_text_after_bbox", return_value=False
        ), patch("extractor.pipeline._vertical_axes_for_bbox", return_value=[40.0, 540.0]), patch(
            "extractor.pipeline._collect_embedded_image_refs", return_value={}
        ), patch(
            "extractor.pipeline._extract_body_text", side_effect=fake_extract_body_text
        ), patch(
            "extractor.pipeline._extract_embedded_images",
            return_value=[],
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                extract_pdf_to_outputs(
                    pdf_path=Path("unused.pdf"),
                    out_md_dir=root / "md",
                    out_image_dir=root / "images",
                    stem="orphan-header-owned",
                    page_write=True,
                )

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
        self.assertIsNotNone(re.search(r"^\[//\]: # \(.+ - Table 1\)$", table_markdown, flags=re.MULTILINE))
        self.assertIsNotNone(re.search(r"^\[//\]: # \(.+ - Table 2\)$", table_markdown, flags=re.MULTILINE))
        self.assertEqual(
            1,
            len(re.findall(r"^\[//\]: # \(.+ - Table 1\)$", table_markdown, flags=re.MULTILINE)),
            table_markdown,
        )
        self.assertNotIn("| B2 | 11 |", blocks[0], table_markdown)
        self.assertIn("| B2 | 11 |", blocks[1], table_markdown)

    def test_demo_table_markdown_is_written_to_separate_file(self) -> None:
        result = self._extract_result()
        self.assertEqual(result["table_markdown"], result["table_md_file"].read_text(encoding="utf-8"))
        self.assertTrue(result["table_md_file"].name.endswith("_tables.md"))
        self.assertEqual(2, len(result["image_files"]))

    def test_header_images_are_not_exported_as_body_images(self) -> None:
        pdf_path = self._build_pdf()
        with pdfplumber.open(str(pdf_path)) as pdf:
            raw_image_count = sum(len(page.images) for page in pdf.pages)

        result = self._extract_result()
        self.assertEqual(6, raw_image_count)
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
        self.assertTrue(any(abs(entry["rotation"]) > 0.1 for entry in payload))

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

    def test_collect_table_drawing_debug_reports_expected_page1_grid(self) -> None:
        pdf_path = self._build_pdf()
        with pdfplumber.open(str(pdf_path)) as pdf:
            payload = _collect_table_drawing_debug(pdf.pages[0], page_no=1)

        self.assertEqual(2, len(payload["tables"]))
        self.assertEqual(6, payload["tables"][0]["row_count"])
        self.assertEqual(3, payload["tables"][0]["col_count"])
        self.assertTrue(payload["tables"][0]["horizontal_groups"])
        self.assertTrue(payload["tables"][0]["vertical_groups"])
        self.assertIn("stroking_color", payload["tables"][0]["horizontal_segments"][0])
        self.assertIn("linewidth", payload["tables"][0]["horizontal_segments"][0])
        self.assertIn("stroke", payload["tables"][0]["horizontal_segments"][0])
        self.assertIn("source_drawings", payload)
        self.assertIn("lines", payload["source_drawings"])
        self.assertIn("rects", payload["source_drawings"])
        self.assertIn("curves", payload["source_drawings"])
        self.assertTrue(payload["source_drawings"]["lines"])
        self.assertIn("table_objects", payload)
        self.assertIn("image_objects", payload)
        self.assertIn("text_objects", payload)
        self.assertIn("profile", payload["text_debug"])
        self.assertIn("dominant_font_size", payload["text_debug"]["profile"])
        self.assertIn("font_size_histogram", payload["text_debug"]["profile"])
        self.assertIn("font_color_histogram", payload["text_debug"]["profile"])
        self.assertIn("font_size_candidates", payload["text_debug"]["raw_line_boxes"][0])
        self.assertIn("dominant_font_size", payload["text_debug"]["raw_line_boxes"][0])
        self.assertIn("font_color", payload["text_debug"]["raw_line_boxes"][0])
        self.assertNotEqual("", payload["text_debug"]["raw_line_boxes"][0]["font_color"])

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

        payload = json.loads(result["debug_file"].read_text(encoding="utf-8"))
        edge_payload = json.loads(result["debug_edges_file"].read_text(encoding="utf-8"))
        self.assertEqual(4, len(payload["pages"]))
        self.assertEqual(4, len(edge_payload["pages"]))
        self.assertIn("text_debug", payload["pages"][0])
        self.assertIn("stroking_color", payload["pages"][0]["tables"][0]["horizontal_segments"][0])
        self.assertIn("document_text_profile", payload)
        self.assertIn("font_size_histogram", payload["document_text_profile"])
        self.assertIn("font_color_histogram", payload["pages"][0]["text_debug"]["profile"])
        self.assertIn("source_drawings", payload["pages"][0])
        self.assertIn("lines", payload["pages"][0]["source_drawings"])
        self.assertIn("profile", payload["pages"][0]["text_debug"])
        self.assertIn("linewidth", edge_payload["pages"][0]["all_horizontal_edges"][0])

    def test_long_legal_notes_row_spans_two_pages(self) -> None:
        markdown = self._extract_table_markdown().lower()
        self.assertIn("policy exception register", markdown)
        self.assertIn("sign-off archive retention", markdown)

    def test_demo_pdf_has_rotated_gray_watermark_on_every_page(self) -> None:
        pdf_path = self._build_pdf()
        with pdfplumber.open(str(pdf_path)) as pdf:
            self.assertEqual(len(pdf.pages), 4)
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
