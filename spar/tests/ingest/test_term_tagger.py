from __future__ import annotations

from spar.ingest.term_tagger import tag_chunk


def _chunk(text: str) -> dict:
    return {
        "chunk_id": "abc123",
        "doc_type": "spec",
        "source_doc": "doc.md",
        "text": text,
        "keywords": [],
    }


class TestTagChunk:
    def test_matches_present_keyword(self) -> None:
        chunk = _chunk("The NRCellDU object configures the cell.")
        result = tag_chunk(chunk, {"NRCellDU", "maxRetransmissions"})
        assert "NRCellDU" in result["keywords"]

    def test_no_match_returns_empty(self) -> None:
        chunk = _chunk("This is an unrelated paragraph.")
        result = tag_chunk(chunk, {"NRCellDU"})
        assert result["keywords"] == []

    def test_case_insensitive(self) -> None:
        chunk = _chunk("configure nrcelldu parameters")
        result = tag_chunk(chunk, {"NRCellDU"})
        assert "NRCellDU" in result["keywords"]

    def test_word_boundary_prevents_partial_match(self) -> None:
        chunk = _chunk("NRCellDUExtra is not the same as NRCellDU")
        result = tag_chunk(chunk, {"NRCellDU"})
        assert "NRCellDU" in result["keywords"]
        assert len(result["keywords"]) == 1

    def test_caps_at_50(self) -> None:
        keywords = {f"term{i}" for i in range(60)}
        text = " ".join(keywords)
        chunk = _chunk(text)
        result = tag_chunk(chunk, keywords)
        assert len(result["keywords"]) <= 50

    def test_returns_same_chunk_reference(self) -> None:
        chunk = _chunk("NRCellDU")
        result = tag_chunk(chunk, {"NRCellDU"})
        assert result is chunk

    def test_empty_keywords_set(self) -> None:
        chunk = _chunk("NRCellDU maxRetransmissions")
        result = tag_chunk(chunk, set())
        assert result["keywords"] == []
