import pytest
from spar.ingest.chunkers import chunk_markdown, chunk_fixed, chunk_3gpp_sections, dispatch


SAMPLE_MD = """# 1 Scope
This document specifies foo.

# 2 References
The following documents are referenced.

## 2.1 Normative
- Reference A
- Reference B

# 3 Definitions
Terms used in this spec.
"""


def test_markdown_chunker_splits_on_headers():
    chunks = chunk_markdown(SAMPLE_MD, source_doc="21101.md")
    # 4 헤더 → 4 청크 (전부 max_words 미만)
    assert len(chunks) == 4
    sections = [c["section"] for c in chunks]
    assert sections == ["1 Scope", "2 References", "2.1 Normative", "3 Definitions"]


def test_markdown_chunker_preserves_text_under_header():
    chunks = chunk_markdown(SAMPLE_MD, source_doc="21101.md")
    scope = next(c for c in chunks if c["section"] == "1 Scope")
    assert "specifies foo" in scope["text"]


def test_markdown_chunker_subsplits_long_section():
    long_text = "# Big\n" + "word " * 1200
    chunks = chunk_markdown(long_text, source_doc="big.md", max_words=500)
    # 1200 words / 500 → 3 청크
    assert len(chunks) == 3
    assert all(c["section"] == "Big" for c in chunks)


def test_chunk_ids_unique():
    chunks = chunk_markdown(SAMPLE_MD, source_doc="21101.md")
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_fixed_chunker_word_buckets():
    text = " ".join(["w"] * 1500)
    chunks = chunk_fixed(text, source_doc="x.txt", doc_type="mop", words=500)
    assert len(chunks) == 3
    assert all(c["doc_type"] == "mop" for c in chunks)


def test_dispatch_spec_uses_markdown():
    chunks = dispatch(SAMPLE_MD, source_doc="21101.md", doc_type="spec")
    assert len(chunks) == 4  # 헤더 기반


def test_dispatch_mop_uses_fixed():
    chunks = dispatch(SAMPLE_MD, source_doc="x.md", doc_type="mop")
    # fixed 청커는 헤더 무시 — 1 청크 (전체 < 500 words)
    assert len(chunks) == 1


def test_empty_text_returns_empty_list():
    assert chunk_markdown("", source_doc="x.md") == []
    assert chunk_fixed("", source_doc="x.md", doc_type="mop") == []


def test_markdown_chunker_skips_headers_inside_code_blocks():
    fence = "```"
    text = (
        "# Real Header\n"
        "Some text.\n"
        "\n"
        + fence + "\n"
        "# this is NOT a header\n"
        "# define FOO 1\n"
        + fence + "\n"
        "\n"
        "# Another Real Header\n"
        "More text.\n"
    )
    chunks = chunk_markdown(text, source_doc="x.md")
    sections = [c["section"] for c in chunks]
    assert sections == ["Real Header", "Another Real Header"]


def test_markdown_chunker_strips_atx_closing_hashes():
    text = "## Architecture ##\nbody\n## Another ###\nmore body\n"
    chunks = chunk_markdown(text, source_doc="x.md")
    sections = [c["section"] for c in chunks]
    assert sections == ["Architecture", "Another"]


def test_markdown_chunker_preserves_code_block_in_body():
    fence = "```"
    text = (
        "# Section\n"
        "Header text.\n"
        "\n"
        + fence + "\n"
        "def foo():\n"
        "    return 42\n"
        + fence + "\n"
        "\n"
        "After code.\n"
    )
    chunks = chunk_markdown(text, source_doc="x.md")
    assert len(chunks) == 1
    body = chunks[0]["text"]
    assert "def foo()" in body
    assert "return 42" in body
    assert "After code" in body


# --- chunk_3gpp_sections tests ---

_3GPP_MD = """\
### 4.2.1 Introduction
Overview text here.

### 4.2.2 Security requirements
Security content.

#### 4.2.2.1 NF discovery
NF discovery detail.

##### 4.2.2.1.1 Specific slice
Slice content.
"""


def test_3gpp_section_num_extracted():
    chunks = chunk_3gpp_sections(_3GPP_MD, source_doc="33518.md")
    nums = [c["section_num"] for c in chunks]
    assert nums == ["4.2.1", "4.2.2", "4.2.2.1", "4.2.2.1.1"]


def test_3gpp_section_title_extracted():
    chunks = chunk_3gpp_sections(_3GPP_MD, source_doc="33518.md")
    titles = [c["section_title"] for c in chunks]
    assert titles == ["Introduction", "Security requirements", "NF discovery", "Specific slice"]


def test_3gpp_section_depth():
    chunks = chunk_3gpp_sections(_3GPP_MD, source_doc="33518.md")
    depths = [c["section_depth"] for c in chunks]
    assert depths == [3, 3, 4, 5]


def test_3gpp_parent_sections():
    chunks = chunk_3gpp_sections(_3GPP_MD, source_doc="33518.md")
    by_num = {c["section_num"]: c for c in chunks}
    assert by_num["4.2.1"]["parent_sections"] == ["4", "4.2"]
    assert by_num["4.2.2.1"]["parent_sections"] == ["4", "4.2", "4.2.2"]
    assert by_num["4.2.1"]["parent_sections"] == ["4", "4.2"]


def test_3gpp_chunk_index_sequential():
    chunks = chunk_3gpp_sections(_3GPP_MD, source_doc="33518.md")
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_3gpp_chunk_index_in_section_zero_when_no_split():
    chunks = chunk_3gpp_sections(_3GPP_MD, source_doc="33518.md")
    assert all(c["chunk_index_in_section"] == 0 for c in chunks)


def test_3gpp_overlap_creates_window_chunks():
    # 600 words in one section → 2 windows with overlap=100, max_words=500
    text = "### 5.1 Big section\n" + "word " * 600
    chunks = chunk_3gpp_sections(text, source_doc="x.md", max_words=500, overlap=100)
    assert len(chunks) == 2
    assert chunks[0]["chunk_index_in_section"] == 0
    assert chunks[1]["chunk_index_in_section"] == 1


def test_3gpp_overlap_words_shared_at_boundary():
    # words 400-500 from chunk0 should appear as first 100 words of chunk1
    text = "### 5.1 Big\n" + " ".join(f"w{i}" for i in range(600))
    chunks = chunk_3gpp_sections(text, source_doc="x.md", max_words=500, overlap=100)
    assert len(chunks) == 2
    tail_of_first = chunks[0]["text"].split()[-100:]
    head_of_second = chunks[1]["text"].split()[:100]
    assert tail_of_first == head_of_second


def test_3gpp_section_field_contains_num_and_title():
    chunks = chunk_3gpp_sections(_3GPP_MD, source_doc="33518.md")
    c = next(c for c in chunks if c["section_num"] == "4.2.2.1")
    assert c["section"] == "4.2.2.1 NF discovery"


def test_3gpp_multiple_sections_chunk_index_continuous():
    text = (
        "### 6.1 First\n" + "a " * 600 + "\n"
        "### 6.2 Second\n" + "b " * 600 + "\n"
    )
    chunks = chunk_3gpp_sections(text, source_doc="x.md", max_words=500, overlap=50)
    # 6.1 → 2 sub-chunks, 6.2 → 2 sub-chunks = 4 total
    assert len(chunks) == 4
    assert [c["chunk_index"] for c in chunks] == [0, 1, 2, 3]
    assert chunks[0]["chunk_index_in_section"] == 0
    assert chunks[1]["chunk_index_in_section"] == 1
    assert chunks[2]["chunk_index_in_section"] == 0
    assert chunks[3]["chunk_index_in_section"] == 1
