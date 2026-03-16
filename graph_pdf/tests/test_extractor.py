from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pdfplumber

from extractor import extract_pdf_to_outputs
from sample_generator import create_demo_pdf


class TableExtractionFormattingTests(unittest.TestCase):
    def _build_pdf(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pdf_path = Path(tmp.name) / "sample.pdf"
        create_demo_pdf(pdf_path)
        return pdf_path

    def _extract_result(self) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_path = root / "sample.pdf"
            md_dir = root / "md"
            image_dir = root / "images"
            create_demo_pdf(pdf_path)
            result = extract_pdf_to_outputs(
                pdf_path=pdf_path,
                out_md_dir=md_dir,
                out_image_dir=image_dir,
                stem="sample",
            )
            return result

    def _extract_markdown(self) -> str:
        return self._extract_result()["markdown"]

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
        markdown = self._extract_markdown()
        self.assertIn("### Page 1 table 1", markdown)
        self.assertIn("| Item | Qty | Price |", markdown)
        self.assertIn("| --- | --- | --- |", markdown)
        self.assertNotIn("- Row 1", markdown)
        self.assertIn("<br>", markdown)

    def test_watermark_fragments_do_not_remain_in_table_cells(self) -> None:
        markdown = self._extract_markdown()
        self.assertNotIn("FID", markdown)
        self.assertNotIn("I Qty", markdown)
        self.assertNotIn("N <br>", markdown)
        self.assertNotIn("O <br>", markdown)

    def test_wrapped_cell_text_is_collapsed_but_bullets_remain_split(self) -> None:
        markdown = self._extract_markdown()
        self.assertIn("| Laptop<br>- line 1 | 12 | $120 |", markdown)
        self.assertIn(
            "Docking station compatibility review package for extended desktop deployment approval",
            markdown,
        )
        self.assertIn("| Docs | READY | Finalize<br>- sample<br>- archive |", markdown)

    def test_spanning_stage_table_merges_into_one_block(self) -> None:
        markdown = self._extract_markdown()
        blocks = self._table_blocks(markdown)
        self.assertEqual(3, len(blocks))
        stage_block = next((block for block in blocks if "Phase A" in block), "")
        self.assertTrue(stage_block)
        self.assertIn("Release Notes", stage_block)
        self.assertIn("Phase C", stage_block)
        self.assertIn("Finance", stage_block)
        self.assertNotIn("### Page 2 table 3", markdown)

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
        self.assertGreaterEqual(len(texts), 4)
        self.assertIn("Legal", joined)
        self.assertIn("policy exception register", texts[2].lower())
        self.assertIn("sign-off archive retention", texts[3].lower())

    def test_demo_pdf_has_confidential_watermark_on_every_page(self) -> None:
        pdf_path = self._build_pdf()
        with pdfplumber.open(str(pdf_path)) as pdf:
            self.assertGreaterEqual(len(pdf.pages), 4)
            for page in pdf.pages:
                watermark_chars = [
                    char
                    for char in page.chars
                    if "CONFIDENTIAL".startswith(char.get("text", "").upper())
                    and float(char.get("size", 0)) >= 40
                ]
                self.assertTrue(watermark_chars)


if __name__ == "__main__":
    unittest.main()
