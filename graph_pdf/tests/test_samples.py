from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from extractor.pipeline import extract_pdf_to_outputs
from extractor.raw import materialize_raw_dump


class SampleRawDumpTests(unittest.TestCase):
    _ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "sample_visuals" / "parse"

    def _sample_raw_paths(self) -> list[Path]:
        samples_dir = Path(__file__).resolve().parents[1] / "samples"
        return sorted(samples_dir.glob("*.dump"))

    def _load_artifact_summary(self, raw_path: Path, mode: str) -> dict[str, Any]:
        stem = raw_path.stem
        expected_stem = f"{stem}_{mode}" if mode == "from_raw" else stem
        summary_path = self._ARTIFACT_ROOT / stem / mode / "md" / f"{expected_stem}_summary.json"
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def _assert_output_snapshot_matches(self, result: dict[str, Any], raw_path: Path, mode: str) -> None:
        artifact = self._load_artifact_summary(raw_path=raw_path, mode=mode)

        artifact_md = Path(artifact["md_file"]).read_text(encoding="utf-8")
        artifact_txt = Path(artifact["text_file"]).read_text(encoding="utf-8")
        artifact_table = Path(artifact["table_md_file"]).read_text(encoding="utf-8")
        artifact_images = sorted(Path(p).name for p in artifact["images"])

        artifact_md_file = Path(artifact["md_file"])
        artifact_table_file = Path(artifact["table_md_file"])
        artifact_txt_file = Path(artifact["text_file"])

        self.assertTrue(artifact_md_file.exists(), f"missing artifact md fixture: {artifact_md_file}")
        self.assertTrue(artifact_table_file.exists(), f"missing artifact table md fixture: {artifact_table_file}")
        self.assertTrue(artifact_txt_file.exists(), f"missing artifact txt fixture: {artifact_txt_file}")

        actual_md = result["markdown"]
        actual_table_md = result["table_markdown"]
        actual_txt = Path(result["text_file"]).read_text(encoding="utf-8")
        actual_images = sorted(Path(p).name for p in result["summary"]["images"])
        self.assertEqual(artifact["table_count"], result["summary"]["table_count"], raw_path.name)
        self.assertEqual(len(artifact["images"]), len(actual_images), raw_path.name)
        self.assertEqual(artifact_images, actual_images, raw_path.name)
        self.assertMultiLineEqual(artifact_md, actual_md, raw_path.name)
        self.assertMultiLineEqual(artifact_table, actual_table_md, raw_path.name)
        self.assertMultiLineEqual(artifact_txt, actual_txt, raw_path.name)

        for image_name in artifact_images:
            artifact_image_path = self._ARTIFACT_ROOT / raw_path.stem / mode / "images" / image_name
            actual_image_path = next((Path(path) for path in result["summary"]["images"] if Path(path).name == image_name), None)
            self.assertIsNotNone(actual_image_path, f"missing image in current output: {image_name}")

            artifact_digest = hashlib.sha256(artifact_image_path.read_bytes()).hexdigest()
            actual_digest = hashlib.sha256(actual_image_path.read_bytes()).hexdigest()
            self.assertEqual(artifact_digest, actual_digest, f"image changed: {image_name} ({raw_path.name}, {mode})")

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
                            stem=raw_path.stem,
                        )

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
                    self.assertMultiLineEqual(
                        Path(result_from_raw["text_file"]).read_text(encoding="utf-8"),
                        Path(direct_result["text_file"]).read_text(encoding="utf-8"),
                        raw_path.name,
                    )

                    self._assert_output_snapshot_matches(result_from_raw, raw_path=raw_path, mode="from_raw")
                    self._assert_output_snapshot_matches(direct_result, raw_path=raw_path, mode="pdf_path")
                    self.assertTrue(Path(direct_result["text_file"]).exists())
                    self.assertTrue(Path(direct_result["md_file"]).exists())
                    self.assertTrue(Path(direct_result["table_md_file"]).exists())


if __name__ == "__main__":
    unittest.main()
