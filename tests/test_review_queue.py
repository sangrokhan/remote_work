import pytest
from causality_graph.extraction.review_queue import ReviewQueue, ReviewStatus

TRIPLE = {
    "from": "feature:CA",
    "to": "kpi:dl_throughput",
    "relation": "AFFECTS",
    "direction": "+",
    "magnitude": "high",
    "condition": "",
    "confidence": 0.92,
    "source_doc": "feature_CA.md",
}


@pytest.fixture
def queue(tmp_path):
    return ReviewQueue(path=tmp_path / "queue.jsonl")


def test_enqueue_and_list_pending(queue):
    queue.enqueue(TRIPLE)
    pending = queue.list_pending()
    assert len(pending) == 1
    assert pending[0]["from"] == "feature:CA"
    assert pending[0]["status"] == ReviewStatus.PENDING.value


def test_approve(queue):
    queue.enqueue(TRIPLE)
    item_id = queue.list_pending()[0]["id"]
    queue.approve(item_id)
    assert queue.list_pending() == []
    approved = queue.list_by_status(ReviewStatus.APPROVED)
    assert len(approved) == 1


def test_reject(queue):
    queue.enqueue(TRIPLE)
    item_id = queue.list_pending()[0]["id"]
    queue.reject(item_id, reason="incorrect direction")
    rejected = queue.list_by_status(ReviewStatus.REJECTED)
    assert rejected[0]["reject_reason"] == "incorrect direction"


def test_auto_approve_by_confidence(queue):
    queue.enqueue({**TRIPLE, "confidence": 0.97})
    queue.enqueue({**TRIPLE, "confidence": 0.60, "to": "kpi:other"})
    queue.auto_approve(threshold=0.95)
    pending = queue.list_pending()
    assert len(pending) == 1
    assert pending[0]["confidence"] == 0.60


def test_persists_across_instances(tmp_path):
    path = tmp_path / "queue.jsonl"
    q1 = ReviewQueue(path=path)
    q1.enqueue(TRIPLE)
    q2 = ReviewQueue(path=path)
    assert len(q2.list_pending()) == 1
