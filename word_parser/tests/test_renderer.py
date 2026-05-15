import re
import pytest
from core.models import ParagraphElement, TableElement, Run, Chunk
from core.renderer import render_chunk, slugify


def normal_para(text):
    return ParagraphElement(
        text=text, style_name="Normal",
        runs=[Run(text=text, font_size=None, bold=False)],
        page_approx=1,
    )


def table_elem(rows):
    return TableElement(
        rows=rows, col_count=len(rows[0]),
        page_approx=1, preceded_by_page_break=False,
    )


def test_slugify_basic():
    assert slugify("3.2 Configuration Details") == "3_2_configuration_details"


def test_slugify_special_chars():
    assert slugify("Hello, World!") == "hello_world"


def test_render_heading():
    chunk = Chunk(heading_text="Introduction", heading_depth=2, tag="intro", elements=[], index=0)
    content_md, _ = render_chunk(chunk)
    assert content_md.startswith("## Introduction")


def test_render_paragraph_body():
    chunk = Chunk(
        heading_text="Section", heading_depth=1, tag="sec",
        elements=[normal_para("Some body text")], index=0,
    )
    content_md, _ = render_chunk(chunk)
    assert "Some body text" in content_md


def test_render_table_gfm():
    rows = [["Name", "Value"], ["foo", "bar"], ["baz", "qux"]]
    chunk = Chunk(
        heading_text="Config", heading_depth=1, tag="config",
        elements=[table_elem(rows)], index=0,
    )
    _, table_md = render_chunk(chunk)
    assert "| Name | Value |" in table_md
    assert "| --- | --- |" in table_md
    assert "| foo | bar |" in table_md


def test_render_table_id_comment():
    rows = [["A", "B"], ["1", "2"]]
    chunk = Chunk(
        heading_text="Config", heading_depth=1, tag="config",
        elements=[table_elem(rows)], index=0,
    )
    _, table_md = render_chunk(chunk)
    assert "<!-- table-id: config_table_1 -->" in table_md


def test_render_second_table_increments_id():
    rows = [["A"], ["1"]]
    chunk = Chunk(
        heading_text="Config", heading_depth=1, tag="config",
        elements=[table_elem(rows), table_elem(rows)], index=0,
    )
    _, table_md = render_chunk(chunk)
    assert "<!-- table-id: config_table_1 -->" in table_md
    assert "<!-- table-id: config_table_2 -->" in table_md


def test_render_empty_heading_no_heading_line():
    chunk = Chunk(heading_text="", heading_depth=0, tag="preamble",
                  elements=[normal_para("preamble")], index=0)
    content_md, _ = render_chunk(chunk)
    assert not content_md.startswith("#")
    assert "preamble" in content_md
