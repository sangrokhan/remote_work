from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pdfplumber

from extractor import extract_pdf_to_outputs
from sample_generator import create_demo_pdf


class TableExtractionFormattingTests(unittest.TestCase):
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

    def test_table_output_uses_row_blocks_not_markdown_table(self) -> None:
        markdown = self._extract_markdown()
        self.assertIn("### Page 1 table 1", markdown)
        self.assertIn("- Row 1", markdown)
        self.assertNotIn("| Item |", markdown)
        self.assertNotIn("<br>", markdown)

    def test_watermark_fragments_do_not_remain_in_table_cells(self) -> None:
        markdown = self._extract_markdown()
        self.assertNotIn("FID", markdown)
        self.assertNotIn("I Qty", markdown)
        self.assertNotIn("N <br>", markdown)
        self.assertNotIn("O <br>", markdown)

    def test_wrapped_cell_text_is_collapsed_but_bullets_remain_split(self) -> None:
        markdown = self._extract_markdown()
        self.assertIn("Item: Laptop", markdown)
        self.assertIn("Qty: 12", markdown)
        self.assertIn("Price: $120", markdown)
        self.assertIn(
            "Item: Docking station compatibility review package for extended desktop deployment approval",
            markdown,
        )
        self.assertIn("Area: Docs", markdown)
        self.assertIn("Action: Finalize", markdown)
        self.assertIn("  - sample", markdown)
        self.assertIn("  - archive", markdown)

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

    def test_demo_pdf_has_confidential_watermark_on_every_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "sample.pdf"
            create_demo_pdf(pdf_path)
            with pdfplumber.open(str(pdf_path)) as pdf:
                self.assertGreaterEqual(len(pdf.pages), 3)
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
