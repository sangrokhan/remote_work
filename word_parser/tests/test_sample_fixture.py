"""Integration tests against tests/sample.docx fixture."""
import pytest
from pathlib import Path

from core.document import stream_elements, attach_captions
from core.models import ImageElement, TableElement, ParagraphElement
from core.chunker import build_chunks
from core.config import Config

SAMPLE = Path(__file__).parent / "sample.docx"


@pytest.fixture(scope="module")
def raw_elements():
    data = SAMPLE.read_bytes()
    return list(stream_elements(data))


@pytest.fixture(scope="module")
def elements(raw_elements):
    return attach_captions(raw_elements)


@pytest.fixture(scope="module")
def images(elements):
    return [e for e in elements if isinstance(e, ImageElement)]


@pytest.fixture(scope="module")
def tables(elements):
    return [e for e in elements if isinstance(e, TableElement)]


@pytest.fixture(scope="module")
def paragraphs(elements):
    return [e for e in elements if isinstance(e, ParagraphElement)]


# ── element counts ──────────────────────────────────────────────────────────

def test_image_count(images):
    assert len(images) == 3


def test_table_count(tables):
    assert len(tables) == 3


def test_paragraph_count_nonzero(paragraphs):
    assert len(paragraphs) > 0


# ── images ──────────────────────────────────────────────────────────────────

def test_all_images_are_png(images):
    for img in images:
        assert img.content_type == "image/png"


def test_all_images_have_data(images):
    for img in images:
        assert isinstance(img.data, bytes)
        assert len(img.data) > 0


def test_image_relationship_ids_are_unique(images):
    rids = [img.relationship_id for img in images]
    assert len(rids) == len(set(rids))


def test_image_page_approx_positive(images):
    for img in images:
        assert img.page_approx >= 1


# ── caption attachment ───────────────────────────────────────────────────────

def test_two_images_have_captions(images):
    captioned = [img for img in images if img.caption]
    assert len(captioned) == 2


def test_one_image_has_no_caption(images):
    uncaptioned = [img for img in images if not img.caption]
    assert len(uncaptioned) == 1


def test_caption_text_content(images):
    captions = {img.caption for img in images if img.caption}
    assert "그림 1. 시스템 구성도" in captions
    assert "그림 2. 청크 분리 예시" in captions


# ── tables ───────────────────────────────────────────────────────────────────

def test_table_dimensions(tables):
    dims = sorted((tbl.col_count, len(tbl.rows)) for tbl in tables)
    assert dims == [(2, 3), (2, 5), (3, 4)]


def test_tables_have_header_row(tables):
    for tbl in tables:
        assert len(tbl.rows) >= 2
        header = tbl.rows[0]
        assert any(cell.strip() for cell in header)


# ── heading structure ────────────────────────────────────────────────────────

def test_heading_styles_present(paragraphs):
    style_names = {p.style_name for p in paragraphs}
    assert "Heading 1" in style_names
    assert "Heading 2" in style_names
    assert "Heading 3" in style_names


def test_three_h1_headings(paragraphs):
    h1 = [p for p in paragraphs if p.style_name == "Heading 1"]
    assert len(h1) == 3


def test_h1_text_values(paragraphs):
    texts = [p.text for p in paragraphs if p.style_name == "Heading 1"]
    assert any("개요" in t for t in texts)
    assert any("기능" in t for t in texts)
    assert any("결론" in t for t in texts)


# ── chunking ─────────────────────────────────────────────────────────────────

def _make_config(split_depth: int) -> Config:
    return Config(
        heading_styles={"Heading 1": 1, "Heading 2": 2, "Heading 3": 3},
        font_size_map={},
        body_styles=["Normal", "Caption"],
        table_merge_enabled=False,
        output_dir="output",
        log_level="WARNING",
        chunk_split_depth=split_depth,
    )


def test_chunks_at_depth_3(elements):
    cfg = _make_config(split_depth=3)
    chunks = build_chunks(elements, cfg, logger=None)
    # 5 H3 file-level chunks + 3 preamble chunks (intro paras between H1→H2 folder transitions)
    assert len(chunks) == 8


def test_chunks_at_depth_2(elements):
    cfg = _make_config(split_depth=2)
    chunks = build_chunks(elements, cfg, logger=None)
    # 4 × H2 headings → 4 chunks
    assert len(chunks) == 4


def test_chunks_at_depth_1(elements):
    cfg = _make_config(split_depth=1)
    chunks = build_chunks(elements, cfg, logger=None)
    # 3 × H1 headings → 3 chunks
    assert len(chunks) == 3


def test_chunks_contain_images(elements):
    cfg = _make_config(split_depth=3)
    chunks = build_chunks(elements, cfg, logger=None)
    image_chunks = [c for c in chunks if any(isinstance(e, ImageElement) for e in c.elements)]
    assert len(image_chunks) == 3


def test_chunks_contain_tables(elements):
    cfg = _make_config(split_depth=3)
    chunks = build_chunks(elements, cfg, logger=None)
    table_chunks = [c for c in chunks if any(isinstance(e, TableElement) for e in c.elements)]
    # 3rd table (under H2 "3.1 향후 계획", no H3) is orphaned at split_depth=3 → dropped
    assert len(table_chunks) == 2
