from __future__ import annotations

from pathlib import Path

import pytest
import docx as _docx

from spar.parsers.docx_config import DocxParseConfig


class TestDocxParseConfig:
    def test_defaults(self) -> None:
        cfg = DocxParseConfig()
        assert cfg.heading_depth == 2
        assert cfg.output_dir == Path("output")
        assert cfg.slugify_max_len == 30

    def test_custom_values(self) -> None:
        cfg = DocxParseConfig(heading_depth=3, output_dir=Path("/tmp/out"), slugify_max_len=20)
        assert cfg.heading_depth == 3
        assert cfg.output_dir == Path("/tmp/out")
        assert cfg.slugify_max_len == 20
