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
    chunk = Chunk(heading_text="Introduction", heading_depth=2, elements=[], index=0)
    content_md, _ = render_chunk(chunk)
    assert content_md.startswith("## Introduction")


def test_render_paragraph_body():
    chunk = Chunk(
        heading_text="Section", heading_depth=1,
        elements=[normal_para("Some body text")], index=0,
    )
    content_md, _ = render_chunk(chunk)
    assert "Some body text" in content_md


def test_render_table_gfm():
    rows = [["Name", "Value"], ["foo", "bar"], ["baz", "qux"]]
    chunk = Chunk(
        heading_text="Config", heading_depth=1,
        elements=[table_elem(rows)], index=0,
    )
    _, table_md = render_chunk(chunk)
    assert "| Name | Value |" in table_md
    assert "| --- | --- |" in table_md
    assert "| foo | bar |" in table_md


def test_render_table_id_comment():
    rows = [["A", "B"], ["1", "2"]]
    chunk = Chunk(
        heading_text="Config", heading_depth=1,
        elements=[table_elem(rows)], index=0,
    )
    _, table_md = render_chunk(chunk)
    assert "<!-- table-id: config_table_1 -->" in table_md


def test_render_second_table_increments_id():
    rows = [["A"], ["1"]]
    chunk = Chunk(
        heading_text="Config", heading_depth=1,
        elements=[table_elem(rows), table_elem(rows)], index=0,
    )
    _, table_md = render_chunk(chunk)
    assert "<!-- table-id: config_table_1 -->" in table_md
    assert "<!-- table-id: config_table_2 -->" in table_md


def test_render_empty_heading_no_heading_line():
    chunk = Chunk(heading_text="", heading_depth=0,
                  elements=[normal_para("preamble")], index=0)
    content_md, _ = render_chunk(chunk)
    assert not content_md.startswith("#")
    assert "preamble" in content_md


def test_render_table_cell_newlines_become_br():
    rows = [["항목", "설명"], ["기능 A", "첫 번째 줄\n두 번째 줄\n세 번째 줄"], ["기능 B", "단일 줄"]]
    chunk = Chunk(
        heading_text="Spec", heading_depth=1,
        elements=[table_elem(rows)], index=0,
    )
    _, table_md = render_chunk(chunk)
    assert "첫 번째 줄<br>두 번째 줄<br>세 번째 줄" in table_md
    lines = table_md.splitlines()
    data_lines = [l for l in lines if l.startswith("|") and "---" not in l]
    assert len(data_lines) == 3  # header + 2 data rows, no broken lines


def test_render_sub_heading_as_markdown_header():
    sub = ParagraphElement(
        text="Sub Section", style_name="Heading 3",
        runs=[Run(text="Sub Section", font_size=None, bold=False)],
        page_approx=1,
        heading_depth=3,
    )
    chunk = Chunk(heading_text="Section", heading_depth=2, elements=[sub], index=0)
    content_md, _ = render_chunk(chunk)
    assert "### Sub Section" in content_md
