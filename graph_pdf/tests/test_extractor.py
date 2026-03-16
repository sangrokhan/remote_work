from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from extractor import extract_pdf_to_outputs
from sample_generator import create_demo_pdf


class TableExtractionFormattingTests(unittest.TestCase):
    def _extract_markdown(self) -> str:
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
            return result["markdown"]

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
        self.assertIn("Area: Docs", markdown)
        self.assertIn("Action: Finalize", markdown)
        self.assertIn("  - sample", markdown)
        self.assertIn("  - archive", markdown)


if __name__ == "__main__":
    unittest.main()
