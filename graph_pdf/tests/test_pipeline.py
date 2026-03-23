from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from extractor import extract_pdf_to_outputs
from extractor.debug import _collect_rotated_text_debug, _collect_table_drawing_debug
from extractor.images import _extract_embedded_images
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


class PipelineExtractionTests(unittest.TestCase):
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
        create_demo_pdf(pdf_path)
        return extract_pdf_to_outputs(
            pdf_path=pdf_path,
            out_md_dir=root / "md",
            out_image_dir=root / "images",
            stem="sample",
        )

    def _extract_table_markdown(self) -> str:
        return self._extract_result()["table_markdown"]

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

    def test_fixture_roundtrip_matches_expected_tables(self) -> None:
        fixture = load_demo_fixture()
        markdown = self._extract_table_markdown()
        extracted_tables = _extract_markdown_tables(markdown)
        extracted_by_index = {idx: rows for idx, rows in enumerate(extracted_tables)}

        for idx, table in enumerate(fixture["tables"]):
            self.assertIn(idx, extracted_by_index)
            self.assertEqual(table["rows"], extracted_by_index[idx], table["id"])

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
        self.assertIn("| Docs | READY | Finalize<br>- sample<br>- archive |", markdown)

    def test_stage_table_uses_collapsed_multirow_header_in_markdown_output(self) -> None:
        markdown = self._extract_table_markdown()
        self.assertIn(
            "| Stage<br>Group | Team<br>Function | Notes<br>Deliverable |",
            markdown,
        )

    def test_single_column_box_like_region_is_emitted_as_one_cell_table(self) -> None:
        markdown = self._extract_table_markdown()
        self.assertIn("| Column 1 |", markdown)
        self.assertIn(
            "| Escalation lane summary Owner confirmed for regional review and exception routing.<br>Backup approver stays on the same visual box and must not become a second table row. |",
            markdown,
        )
        self.assertNotIn("| Escalation lane summary |", markdown)
        self.assertNotIn("| Owner confirmed for regional review and exception routing. |", markdown)

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
        )

        self.assertIn("### Page 1", result["markdown"])
        self.assertNotIn("### Page 3", result["markdown"])
        self.assertIn("### Page 1 table 1", result["table_markdown"])
        self.assertEqual(2, result["summary"]["table_count"])
        self.assertEqual(1, len(result["image_files"]))

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
        self.assertEqual("sample_page_02_image_01.png", image_files[0].name)

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

        drawing_images = [Path(path) for path in result["image_files"] if "_drawing_" in Path(path).name]
        self.assertEqual(1, len(drawing_images))
        self.assertTrue(drawing_images[0].exists())

    def test_spanning_stage_table_merges_into_one_block(self) -> None:
        markdown = self._extract_table_markdown()
        blocks = self._table_blocks(markdown)
        self.assertEqual(4, len(blocks))
        stage_block = next((block for block in blocks if "Phase A" in block), "")
        self.assertTrue(stage_block)
        self.assertIn("Release Notes", stage_block)
        self.assertIn("Phase C", stage_block)
        self.assertNotIn("### Page 2 table 3", markdown)

    def test_demo_table_markdown_is_written_to_separate_file(self) -> None:
        result = self._extract_result()
        self.assertEqual(result["table_markdown"], result["table_md_file"].read_text(encoding="utf-8"))
        self.assertTrue(result["table_md_file"].name.endswith("_table.md"))
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
