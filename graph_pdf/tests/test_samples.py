from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from extractor.pipeline import extract_pdf_to_outputs
from extractor.raw import materialize_raw_dump


class SampleRawDumpTests(unittest.TestCase):
    def _sample_raw_paths(self) -> list[Path]:
        samples_dir = Path(__file__).resolve().parents[1] / "samples"
        return sorted(samples_dir.glob("*.dump"))

    def test_sample_raw_dumps_parse_via_from_raw(self) -> None:
        samples = self._sample_raw_paths()
        if not samples:
            self.skipTest("No sample raw dump files found")

        for raw_path in samples:
            with self.subTest(sample=raw_path.name):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    result_from_raw = extract_pdf_to_outputs(
                        pdf_path=None,
                        out_md_dir=root / "from_raw" / "md",
                        out_image_dir=root / "from_raw" / "images",
                        stem=f"{raw_path.stem}_from_raw",
                        from_raw=raw_path,
                    )

                    with materialize_raw_dump(raw_path) as (pdf_path, _payload):
                        direct_result = extract_pdf_to_outputs(
                            pdf_path=pdf_path,
                            out_md_dir=root / "direct" / "md",
                            out_image_dir=root / "direct" / "images",
                            stem=f"{raw_path.stem}_direct",
                        )

                    self.assertTrue(result_from_raw["text_file"].exists())
                    self.assertTrue(result_from_raw["table_md_file"].exists())
                    self.assertEqual(
                        result_from_raw["summary"]["table_count"],
                        direct_result["summary"]["table_count"],
                        raw_path.name,
                    )
                    self.assertEqual(
                        result_from_raw["table_markdown"],
                        direct_result["table_markdown"],
                        raw_path.name,
                    )
                    self.assertEqual(
                        result_from_raw["markdown"],
                        direct_result["markdown"],
                        raw_path.name,
                    )


if __name__ == "__main__":
    unittest.main()
