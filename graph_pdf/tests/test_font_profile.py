from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
        self.assertEqual(1, styles[(11.0, "0.000,0.300,0.700")]["sample_page"])
        self.assertGreaterEqual(styles[(11.0, "0.000,0.300,0.700")]["page_count"], 1)
        self.assertNotIn("Laptop 12 $120", styles[(11.0, "0.000,0.000,0.000")]["sample_texts"])

        with result["csv_file"].open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertTrue(rows)
        self.assertEqual(
            {"font_size", "font_color", "line_count", "page_count", "sample_page", "sample_texts"},
            set(rows[0].keys()),
        )

    def test_profile_pdf_fonts_skips_lines_marked_as_shape_text(self) -> None:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        pdf_path = root / "sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        fake_pdf = SimpleNamespace(pages=[SimpleNamespace()])
        fake_open = MagicMock()
        fake_open.__enter__.return_value = fake_pdf
        fake_open.__exit__.return_value = None

        with patch("extractor.font_profile.pdfplumber.open", return_value=fake_open), patch(
            "extractor.font_profile._extract_tables", return_value=[]
        ), patch(
            "extractor.font_profile._extract_body_word_lines",
            return_value=[
                {
                    "text": "Visible body line",
                    "dominant_font_size": 11.0,
                    "color": [0.0, 0.0, 0.0],
                    "is_shape_text": False,
                },
                {
                    "text": "Diagram label",
                    "dominant_font_size": 11.0,
                    "color": [0.0, 0.0, 0.0],
                    "is_shape_text": True,
                },
            ],
        ):
            result = profile_pdf_fonts(
                pdf_path=pdf_path,
                out_dir=root / "md",
                stem="sample",
            )

        payload = json.loads(result["json_file"].read_text(encoding="utf-8"))
        self.assertEqual(1, payload["styles"][0]["line_count"])
        self.assertEqual(["Visible body line"], payload["styles"][0]["sample_texts"])

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

    def test_cli_add_heading_option_passes_heading_json_to_pipeline(self) -> None:
        root, pdf_path = self._build_pdf()
        heading_json = root / "heading.json"
        heading_json.write_text(
            json.dumps({"heading_rules": [{"match": {"font_size": 20.0}, "assign": {"tag": "h1"}}]}),
            encoding="utf-8",
        )

        argv = [
            "extractor",
            str(pdf_path),
            "--out-md-dir",
            str(root / "md"),
            "--out-image-dir",
            str(root / "images"),
            "--stem",
            "sample",
            "--add-heading",
            str(heading_json),
        ]
        with patch.object(sys, "argv", argv), patch("extractor.__main__.extract_pdf_to_outputs") as mock_extract:
            cli_main()

        self.assertEqual(heading_json, mock_extract.call_args.kwargs["add_heading"])


if __name__ == "__main__":
    unittest.main()
