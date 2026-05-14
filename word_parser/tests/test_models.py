from core.models import Run, ParagraphElement, TableElement, ImageElement, Chunk


def test_paragraph_element_defaults():
    run = Run(text="hello", font_size=12.0, bold=False)
    para = ParagraphElement(
        text="hello",
        style_name="Normal",
        runs=[run],
        page_approx=1,
    )
    assert para.is_page_break is False


def test_table_element():
    tbl = TableElement(
        rows=[["A", "B"], ["1", "2"]],
        col_count=2,
        page_approx=1,
        preceded_by_page_break=False,
    )
    assert tbl.col_count == 2


def test_image_element():
    img = ImageElement(
        relationship_id="rId1",
        content_type="image/png",
        data=b"\x89PNG",
        page_approx=1,
    )
    assert img.content_type == "image/png"


def test_chunk_defaults():
    chunk = Chunk(heading_text="Intro", heading_depth=1, tag="intro", elements=[], index=0)
    assert chunk.table_counter == 0
    assert chunk.image_counter == 0
