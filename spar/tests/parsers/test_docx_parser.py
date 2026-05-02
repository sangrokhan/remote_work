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


from spar.parsers.docx_parser import _slugify


class TestSlugify:
    def test_spaces_to_dash(self) -> None:
        assert _slugify("System Overview", 30) == "System-Overview"

    def test_truncates(self) -> None:
        result = _slugify("A Very Long Section Title Here", 15)
        assert len(result) <= 15

    def test_strips_special_chars(self) -> None:
        assert _slugify("Section (v2.0)!", 30) == "Section-v2.0"

    def test_empty_string(self) -> None:
        assert _slugify("", 30) == "unnamed"


from spar.parsers.docx_parser import ParseResult, ExtractedTable, ExtractedImage


class TestParseResult:
    def test_parse_result_fields(self, tmp_path: Path) -> None:
        table = ExtractedTable(
            path=tmp_path / "Table_Intro_1.csv",
            section_path=["Introduction"],
            seq=1,
        )
        image = ExtractedImage(
            path=tmp_path / "Fig_Intro_1.png",
            section_path=["Introduction"],
            seq=1,
            ext="png",
        )
        result = ParseResult(markdown="# Hello", tables=[table], images=[image])
        assert result.markdown == "# Hello"
        assert len(result.tables) == 1
        assert len(result.images) == 1
        assert result.tables[0].seq == 1
        assert result.images[0].ext == "png"
