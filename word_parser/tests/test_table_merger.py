import pytest
from core.models import ParagraphElement, TableElement, Run
from core.table_merger import merge_tables


def page_break_para():
    return ParagraphElement(
        text="", style_name="Normal", runs=[], page_approx=1, is_page_break=True
    )


def normal_para(text="text"):
    return ParagraphElement(
        text=text, style_name="Normal", runs=[Run(text=text, font_size=None, bold=False)],
        page_approx=1, is_page_break=False
    )


def table(rows, preceded=False):
    return TableElement(
        rows=rows, col_count=len(rows[0]) if rows else 0,
        page_approx=1, preceded_by_page_break=preceded
    )


def test_merges_two_tables_with_only_page_break():
    t1 = table([["A", "B"], ["1", "2"]])
    pb = page_break_para()
    t2 = TableElement(rows=[["A", "B"], ["3", "4"]], col_count=2, page_approx=2, preceded_by_page_break=True)
    elements = [t1, pb, t2]
    result = merge_tables(elements, logger=None)
    tables = [e for e in result if isinstance(e, TableElement)]
    assert len(tables) == 1
    assert len(tables[0].rows) == 3  # both tables merged, duplicate header dropped


def test_drops_repeated_header_on_merge():
    t1 = table([["A", "B"], ["1", "2"]])
    pb = page_break_para()
    t2 = TableElement(rows=[["A", "B"], ["3", "4"]], col_count=2, page_approx=2, preceded_by_page_break=True)
    elements = [t1, pb, t2]
    result = merge_tables(elements, logger=None)
    tables = [e for e in result if isinstance(e, TableElement)]
    # header ["A","B"] appears only once
    assert tables[0].rows.count(["A", "B"]) == 1


def test_no_merge_when_paragraph_between():
    t1 = table([["A", "B"], ["1", "2"]])
    mid = normal_para("some text")
    t2 = TableElement(rows=[["A", "B"], ["3", "4"]], col_count=2, page_approx=2, preceded_by_page_break=False)
    elements = [t1, mid, t2]
    result = merge_tables(elements, logger=None)
    tables = [e for e in result if isinstance(e, TableElement)]
    assert len(tables) == 2


def test_no_merge_different_col_count():
    t1 = table([["A", "B"], ["1", "2"]])
    pb = page_break_para()
    t2 = TableElement(rows=[["X", "Y", "Z"]], col_count=3, page_approx=2, preceded_by_page_break=True)
    elements = [t1, pb, t2]
    result = merge_tables(elements, logger=None)
    tables = [e for e in result if isinstance(e, TableElement)]
    assert len(tables) == 2


def test_page_break_elements_removed_after_merge():
    t1 = table([["A"], ["1"]])
    pb = page_break_para()
    t2 = TableElement(rows=[["A"], ["2"]], col_count=1, page_approx=2, preceded_by_page_break=True)
    elements = [t1, pb, t2]
    result = merge_tables(elements, logger=None)
    page_breaks = [e for e in result if isinstance(e, ParagraphElement) and e.is_page_break]
    assert len(page_breaks) == 0
