from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from extractor.__main__ import main as cli_main
from extractor.pipeline import extract_pdf_to_outputs
from sample_generator import create_demo_pdf


class RawDumpTests(unittest.TestCase):
    def _build_pdf(self) -> tuple[Path, Path]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        pdf_path = root / "sample.pdf"
        create_demo_pdf(pdf_path)
        return root, pdf_path

    def test_dump_pdf_to_raw_file_writes_minimal_payload(self) -> None:
        from extractor.raw import dump_pdf_to_raw_file

        root, pdf_path = self._build_pdf()
        raw_path = root / "sample.raw.dump"

        dump_pdf_to_raw_file(pdf_path=pdf_path, raw_path=raw_path, pages=[1, 2])

        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        self.assertEqual("1.1", payload["schema_version"])
        self.assertTrue(payload["document_pdf_base64"])
        self.assertEqual(2, len(payload))

    def test_extract_pdf_to_outputs_from_raw_matches_pdf_text_outputs(self) -> None:
        from extractor.raw import dump_pdf_to_raw_file

        root, pdf_path = self._build_pdf()
        raw_path = root / "sample.raw.dump"
        dump_pdf_to_raw_file(pdf_path=pdf_path, raw_path=raw_path)

        pdf_result = extract_pdf_to_outputs(
            pdf_path=pdf_path,
            out_md_dir=root / "pdf_md",
            out_image_dir=root / "pdf_images",
            stem="pdf_sample",
        )
        raw_result = extract_pdf_to_outputs(
            pdf_path=pdf_path,
            out_md_dir=root / "raw_md",
            out_image_dir=root / "raw_images",
            stem="raw_sample",
            from_raw=raw_path,
        )

        self.assertEqual(pdf_result["markdown"], raw_result["markdown"])
        self.assertEqual(pdf_result["table_markdown"], raw_result["table_markdown"])
        self.assertEqual(len(pdf_result["image_files"]), len(raw_result["image_files"]))

    def test_cli_raw_and_from_raw_options_work_end_to_end(self) -> None:
        root, pdf_path = self._build_pdf()
        raw_path = root / "sample.raw.dump"

        export_argv = [
            "extractor",
            str(pdf_path),
            "--raw",
            str(raw_path),
            "--pages",
            "1-2",
        ]
        with patch.object(sys, "argv", export_argv):
            cli_main()

        self.assertTrue(raw_path.exists())

        import_argv = [
            "extractor",
            str(pdf_path),
            "--from-raw",
            str(raw_path),
            "--out-md-dir",
            str(root / "md"),
            "--out-image-dir",
            str(root / "images"),
        ]
        with patch.object(sys, "argv", import_argv):
            cli_main()

        self.assertTrue((root / "md" / "document.md").exists())
        self.assertTrue((root / "md" / "output_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
