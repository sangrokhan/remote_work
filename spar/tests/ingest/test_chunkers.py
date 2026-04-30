from spar.ingest.chunkers import chunk_markdown, chunk_fixed, dispatch


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
