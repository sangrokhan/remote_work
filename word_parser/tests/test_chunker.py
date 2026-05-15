import pytest
from core.models import ParagraphElement, TableElement, Run, Chunk
from core.config import Config
from core.chunker import build_chunks


def cfg(**kwargs):
    defaults = dict(
        heading_styles={"Heading 1": 1, "Heading 2": 2},
        font_size_map={},
        body_styles=[],
        table_merge_enabled=True,
        output_dir="output",
        log_level="INFO",
    )
    defaults.update(kwargs)
    return Config(**defaults)


def heading_para(text, style="Heading 1"):
    return ParagraphElement(
        text=text, style_name=style,
        runs=[Run(text=text, font_size=None, bold=False)],
        page_approx=1,
    )


def normal_para(text="body"):
    return ParagraphElement(
        text=text, style_name="Normal",
        runs=[Run(text=text, font_size=None, bold=False)],
        page_approx=1,
    )


def test_single_chunk_from_heading():
    elements = [heading_para("Introduction"), normal_para("Some text")]
    chunks = build_chunks(elements, cfg(), logger=None)
    assert len(chunks) == 1
    assert chunks[0].heading_text == "Introduction"
    assert chunks[0].heading_depth == 1


def test_two_chunks_from_two_headings():
    elements = [
        heading_para("Section A"),
        normal_para("body A"),
        heading_para("Section B"),
        normal_para("body B"),
    ]
    chunks = build_chunks(elements, cfg(), logger=None)
    assert len(chunks) == 2
    assert chunks[0].heading_text == "Section A"
    assert chunks[1].heading_text == "Section B"


def test_preamble_chunk_for_content_before_first_heading():
    elements = [normal_para("preamble text"), heading_para("Section A")]
    chunks = build_chunks(elements, cfg(), logger=None)
    assert chunks[0].heading_text == ""
    assert chunks[0].heading_depth == 0
    assert chunks[1].heading_text == "Section A"


def test_chunk_elements_contain_body_paragraphs():
    body = normal_para("body text")
    elements = [heading_para("Section A"), body]
    chunks = build_chunks(elements, cfg(), logger=None)
    assert body in chunks[0].elements


def test_chunk_index_sequential():
    elements = [
        heading_para("A"), normal_para(),
        heading_para("B"), normal_para(),
        heading_para("C"), normal_para(),
    ]
    chunks = build_chunks(elements, cfg(), logger=None)
    assert [c.index for c in chunks] == [0, 1, 2]
