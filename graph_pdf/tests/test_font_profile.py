from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from extractor import profile_pdf_fonts
from extractor.__main__ import main as cli_main
from sample_generator import create_demo_pdf


class FontProfileTests(unittest.TestCase):
    def _build_pdf(self) -> tuple[Path, Path]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        pdf_path = root / "sample.pdf"
        create_demo_pdf(pdf_path)
        return root, pdf_path

    def test_profile_pdf_fonts_writes_sample_style_summary(self) -> None:
        root, pdf_path = self._build_pdf()

        result = profile_pdf_fonts(
            pdf_path=pdf_path,
            out_dir=root / "md",
            stem="sample",
        )

        payload = json.loads(result["json_file"].read_text(encoding="utf-8"))
        self.assertEqual(str(pdf_path), payload["pdf"])
        self.assertEqual(4, payload["page_count"])
        self.assertTrue(payload["styles"])

        styles = {
            (entry["font_size"], entry["font_color"]): entry
            for entry in payload["styles"]
        }
        self.assertIn((20.0, "0.000,0.000,0.000"), styles)
        self.assertIn((11.0, "0.000,0.300,0.700"), styles)
        self.assertIn(
            "Chapter 1: Deep Structure Verification",
            styles[(20.0, "0.000,0.000,0.000")]["sample_texts"],
        )
        self.assertIn(
            "Blue accent line marks a separate style bucket for font profile review.",
            styles[(11.0, "0.000,0.300,0.700")]["sample_texts"],
        )

        with result["csv_file"].open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertTrue(rows)
        self.assertEqual(
            {"font_size", "font_color", "line_count", "page_count", "sample_pages", "sample_texts"},
            set(rows[0].keys()),
        )

    def test_cli_profile_fonts_option_writes_profile_outputs(self) -> None:
        root, pdf_path = self._build_pdf()

        argv = [
            "extractor",
            str(pdf_path),
            "--out-md-dir",
            str(root / "md"),
            "--stem",
            "sample",
            "--profile-fonts",
        ]
        with patch.object(sys, "argv", argv):
            cli_main()

        self.assertTrue((root / "md" / "sample_font_profile.json").exists())
        self.assertTrue((root / "md" / "sample_font_profile.csv").exists())


if __name__ == "__main__":
    unittest.main()
