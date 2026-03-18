from __future__ import annotations

import unittest

import extractor


class ExtractorPublicApiTests(unittest.TestCase):
    def test_legacy_helpers_are_not_exposed(self) -> None:
        self.assertFalse(hasattr(extractor, "_normalize_body_lines"))
        self.assertFalse(hasattr(extractor, "_normalize_list_block_lines"))
        self.assertFalse(hasattr(extractor, "_looks_like_table"))
        self.assertFalse(hasattr(extractor, "_is_list_continuation_line"))
        self.assertFalse(hasattr(extractor, "_looks_like_inline_term_continuation"))
        self.assertFalse(hasattr(extractor, "_has_room_for_next_line_start"))


if __name__ == "__main__":
    unittest.main()
