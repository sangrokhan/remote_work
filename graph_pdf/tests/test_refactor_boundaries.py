from __future__ import annotations

import unittest

from extractor import extract_pdf_to_outputs
from extractor.text import _normalize_cell_lines


class RefactorBoundaryTests(unittest.TestCase):
    def test_public_entrypoint_remains_importable(self) -> None:
        self.assertTrue(callable(extract_pdf_to_outputs))

    def test_text_helpers_move_into_text_module(self) -> None:
        self.assertEqual(
            ["First sentence.", "Wrapped continuation line stays together"],
            _normalize_cell_lines("First sentence.\nWrapped continuation line\nstays together"),
        )
