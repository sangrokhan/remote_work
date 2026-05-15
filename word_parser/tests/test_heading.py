import logging
import pytest
from core.models import ParagraphElement, Run
from core.config import Config
from core.heading import resolve_heading_depth


def make_config(**kwargs):
    defaults = dict(
        heading_styles={"Heading 1": 1, "Heading 2": 2},
        font_size_map={24: 1, 18: 2},
        body_styles=[],
        table_merge_enabled=True,
        output_dir="output",
        log_level="INFO",
    )
    defaults.update(kwargs)
    return Config(**defaults)


def make_para(text="text", style="Normal", font_size=None, bold=False):
    run = Run(text=text, font_size=font_size, bold=bold)
    return ParagraphElement(text=text, style_name=style, runs=[run], page_approx=1)


def test_heading_style_returns_depth():
    cfg = make_config()
    para = make_para(style="Heading 1")
    depth = resolve_heading_depth(para, cfg, logger=None)
    assert depth == 1


def test_heading_style_2():
    cfg = make_config()
    para = make_para(style="Heading 2")
    depth = resolve_heading_depth(para, cfg, logger=None)
    assert depth == 2


def test_font_size_fallback():
    cfg = make_config()
    para = make_para(style="Normal", font_size=24.0)
    depth = resolve_heading_depth(para, cfg, logger=None)
    assert depth == 1


def test_font_size_fallback_no_match_returns_none():
    cfg = make_config()
    para = make_para(style="Normal", font_size=12.0)
    depth = resolve_heading_depth(para, cfg, logger=None)
    assert depth is None


def test_normal_paragraph_returns_none():
    cfg = make_config()
    para = make_para(style="Normal", font_size=None)
    depth = resolve_heading_depth(para, cfg, logger=None)
    assert depth is None


def test_bold_large_no_match_returns_none():
    cfg = make_config()
    para = make_para(style="Normal", font_size=16.0, bold=True)
    depth = resolve_heading_depth(para, cfg, logger=None)
    assert depth is None
