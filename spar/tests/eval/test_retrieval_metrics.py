import pytest
from spar.eval.metrics import _is_hit, recall_at_k, reciprocal_rank, compute_metrics, hit_rank


GOLD = {
    "query_id": "Q0001",
    "query": "test query",
    "answer": "answer",
    "type": "definition",
    "section": "3.1",
    "source_doc": "29502-i40.md",
    "spec_number": "29.502",
    "release": "Rel-18",
}


def _chunk(source_doc: str, section_num: str) -> dict:
    return {"source_doc": source_doc, "section_num": section_num, "score": 1.0}


class TestIsHit:
    def test_exact_match(self):
        assert _is_hit(_chunk("29502-i40.md", "3.1"), GOLD)

    def test_subsection_match(self):
        assert _is_hit(_chunk("29502-i40.md", "3.1.2"), GOLD)

    def test_wrong_doc(self):
        assert not _is_hit(_chunk("29503-i40.md", "3.1"), GOLD)

    def test_wrong_section(self):
        assert not _is_hit(_chunk("29502-i40.md", "3.2"), GOLD)

    def test_md_suffix_ignored(self):
        assert _is_hit(_chunk("29502-i40", "3.1"), GOLD)


class TestRecallAtK:
    def test_hit_in_top1(self):
        retrieved = [_chunk("29502-i40.md", "3.1"), _chunk("other.md", "5.1")]
        assert recall_at_k(retrieved, GOLD, 1)

    def test_hit_at_rank5(self):
        misses = [_chunk("other.md", "1.0")] * 4
        retrieved = misses + [_chunk("29502-i40.md", "3.1")]
        assert not recall_at_k(retrieved, GOLD, 4)
        assert recall_at_k(retrieved, GOLD, 5)

    def test_no_hit(self):
        retrieved = [_chunk("other.md", "1.0")] * 10
        assert not recall_at_k(retrieved, GOLD, 10)


class TestReciprocalRank:
    def test_hit_at_rank1(self):
        retrieved = [_chunk("29502-i40.md", "3.1")]
        assert reciprocal_rank(retrieved, GOLD) == pytest.approx(1.0)

    def test_hit_at_rank3(self):
        retrieved = [_chunk("other.md", "1.0")] * 2 + [_chunk("29502-i40.md", "3.1")]
        assert reciprocal_rank(retrieved, GOLD) == pytest.approx(1 / 3)

    def test_no_hit(self):
        retrieved = [_chunk("other.md", "1.0")] * 5
        assert reciprocal_rank(retrieved, GOLD) == 0.0


class TestComputeMetrics:
    def _make_result(self, gold: dict, retrieved: list[dict]) -> dict:
        return {"gold": gold, "retrieved": retrieved}

    def test_perfect_retrieval(self):
        result = self._make_result(GOLD, [_chunk("29502-i40.md", "3.1")])
        metrics = compute_metrics([result])
        assert metrics["mrr"] == pytest.approx(1.0)
        assert metrics["recall_at_5"] == pytest.approx(1.0)
        assert metrics["n_queries"] == 1

    def test_no_hit(self):
        result = self._make_result(GOLD, [_chunk("other.md", "1.0")])
        metrics = compute_metrics([result])
        assert metrics["mrr"] == pytest.approx(0.0)
        assert metrics["recall_at_5"] == pytest.approx(0.0)

    def test_by_type_grouping(self):
        g2 = {**GOLD, "query_id": "Q0002", "type": "procedural"}
        results = [
            self._make_result(GOLD, [_chunk("29502-i40.md", "3.1")]),
            self._make_result(g2, [_chunk("other.md", "1.0")]),
        ]
        metrics = compute_metrics(results)
        assert "definition" in metrics["by_type"]
        assert "procedural" in metrics["by_type"]
        assert metrics["by_type"]["definition"]["recall_at_5"] == pytest.approx(1.0)
        assert metrics["by_type"]["procedural"]["recall_at_5"] == pytest.approx(0.0)

    def test_empty_results(self):
        metrics = compute_metrics([])
        assert metrics["n_queries"] == 0
        assert metrics["mrr"] == 0.0


class TestHitRank:
    def test_rank1(self):
        retrieved = [_chunk("29502-i40.md", "3.1")]
        assert hit_rank(retrieved, GOLD) == 1

    def test_rank3(self):
        retrieved = [_chunk("x.md", "1")] * 2 + [_chunk("29502-i40.md", "3.1")]
        assert hit_rank(retrieved, GOLD) == 3

    def test_none_when_missing(self):
        assert hit_rank([_chunk("x.md", "1")], GOLD) is None
