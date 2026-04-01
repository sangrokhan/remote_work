from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from extractor.pipeline import extract_pdf_to_outputs
from extractor.raw import materialize_raw_dump
from scripts.replay_samples import (
    _count_table_references,
    _count_table_sections,
    _extract_table_references,
    _extract_table_sections,
)


class SampleRawDumpTests(unittest.TestCase):
    _ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "samples" / "gold"

    def _assert_summary_table_counts_match(self, summary: dict[str, Any], summary_name: str) -> None:
        table_md_file = Path(summary["table_md_file"])
        table_markdown = table_md_file.read_text(encoding="utf-8")
        actual_table_count = _count_table_sections(table_markdown)
        self.assertEqual(actual_table_count, summary["table_count"], summary_name)

    def _sample_raw_paths(self) -> list[Path]:
        samples_dir = Path(__file__).resolve().parents[1] / "samples"
        return sorted(samples_dir.glob("*.dump"))

    def _load_artifact_summary(self, raw_path: Path, mode: str) -> dict[str, Any]:
        del mode
        stem = raw_path.stem
        summary_path = self._ARTIFACT_ROOT / stem / "md" / f"{stem}_summary.json"
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def _assert_output_snapshot_matches(self, result: dict[str, Any], raw_path: Path, mode: str) -> None:
        artifact = self._load_artifact_summary(raw_path=raw_path, mode=mode)
        artifact_md = Path(artifact["md_file"]).read_text(encoding="utf-8")
        artifact_table_markdown = Path(artifact["table_md_file"]).read_text(encoding="utf-8")
        artifact_table_count = _count_table_sections(artifact_table_markdown)
        artifact_txt = Path(artifact["text_file"]).read_text(encoding="utf-8")
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
        self.assertEqual(artifact_table_count, result["summary"]["table_count"], raw_path.name)
        self.assertEqual(len(artifact["images"]), len(actual_images), raw_path.name)
        self.assertEqual(artifact_images, actual_images, raw_path.name)
        self.assertMultiLineEqual(artifact_md, actual_md, raw_path.name)
        self.assertMultiLineEqual(artifact_table_markdown, actual_table_md, raw_path.name)
        self.assertMultiLineEqual(artifact_txt, actual_txt, raw_path.name)

        artifact_image_paths = {Path(path).name: Path(path) for path in artifact["images"]}
        for image_name in artifact_images:
            artifact_image_path = artifact_image_paths.get(image_name)
            actual_image_path = next((Path(path) for path in result["summary"]["images"] if Path(path).name == image_name), None)
            self.assertIsNotNone(artifact_image_path, f"missing artifact image: {image_name}")
            self.assertIsNotNone(actual_image_path, f"missing image in current output: {image_name}")

            artifact_digest = hashlib.sha256(Path(artifact_image_path).read_bytes()).hexdigest()
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

    def test_replay_samples_uses_current_table_reference_contract(self) -> None:
        markdown = "Body text\n\n[FGR-BC0401_tables.md - Table 1]\n"
        table_markdown = "[FGR-BC0401_tables.md - Table 1]\n| Col |\n| --- |\n| Value |\n"

        self.assertEqual(1, _count_table_references(markdown))
        self.assertEqual(1, _count_table_sections(table_markdown))
        self.assertEqual(
            ["FGR-BC0401_tables.md - Table 1"],
            _extract_table_references(markdown),
        )
        self.assertEqual(
            ["FGR-BC0401_tables.md - Table 1"],
            _extract_table_sections(table_markdown),
        )

    def test_gold_summaries_match_table_markdown_counts(self) -> None:
        for summary_path in sorted(self._ARTIFACT_ROOT.glob("raw-*/md/*_summary.json")):
            with self.subTest(summary=summary_path.name):
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                self._assert_summary_table_counts_match(summary, summary_path.name)
                for index, document in enumerate(summary.get("documents", []), start=1):
                    self._assert_summary_table_counts_match(document, f"{summary_path.name} documents[{index}]")

    def test_raw_93_114_table_count_matches_golden_table_file(self) -> None:
        raw_path = Path(__file__).resolve().parents[1] / "samples" / "raw-93-114.dump"
        artifact = self._load_artifact_summary(raw_path=raw_path, mode="from_raw")
        expected_table_count = _count_table_sections(
            Path(artifact["table_md_file"]).read_text(encoding="utf-8")
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = extract_pdf_to_outputs(
                pdf_path=None,
                out_md_dir=root / "from_raw" / "md",
                out_image_dir=root / "from_raw" / "images",
                stem="raw-93-114_from_raw",
                from_raw=raw_path,
            )

        self.assertEqual(expected_table_count, result["summary"]["table_count"])

    def test_raw_93_114_table_19_matches_golden_when_header_starts_on_previous_page(self) -> None:
        raw_path = Path(__file__).resolve().parents[1] / "samples" / "raw-93-114.dump"
        artifact = self._load_artifact_summary(raw_path=raw_path, mode="from_raw")
        expected_blocks = Path(artifact["table_md_file"]).read_text(encoding="utf-8").strip().split("\n\n")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = extract_pdf_to_outputs(
                pdf_path=None,
                out_md_dir=root / "from_raw" / "md",
                out_image_dir=root / "from_raw" / "images",
                stem="raw-93-114_from_raw",
                from_raw=raw_path,
            )

        actual_blocks = result["table_markdown"].strip().split("\n\n")
        self.assertGreaterEqual(len(actual_blocks), 19)
        self.assertGreaterEqual(len(expected_blocks), 19)
        self.assertEqual(expected_blocks[18], actual_blocks[18])


if __name__ == "__main__":
    unittest.main()
