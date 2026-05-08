import json
import uuid
from enum import Enum
from pathlib import Path


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ReviewQueue:
    def __init__(self, path: Path):
        self._path = Path(path)
        self._path.touch()

    def _read_all(self) -> list[dict]:
        lines = self._path.read_text().splitlines()
        return [json.loads(l) for l in lines if l.strip()]

    def _write_all(self, items: list[dict]) -> None:
        self._path.write_text("\n".join(json.dumps(i) for i in items) + "\n" if items else "")

    def enqueue(self, triple: dict) -> str:
        item_id = str(uuid.uuid4())
        item = {**triple, "id": item_id, "status": ReviewStatus.PENDING.value, "reject_reason": ""}
        items = self._read_all()
        items.append(item)
        self._write_all(items)
        return item_id

    def list_pending(self) -> list[dict]:
        return [i for i in self._read_all() if i["status"] == ReviewStatus.PENDING.value]

    def list_by_status(self, status: ReviewStatus) -> list[dict]:
        return [i for i in self._read_all() if i["status"] == status.value]

    def approve(self, item_id: str) -> None:
        items = self._read_all()
        for item in items:
            if item["id"] == item_id:
                item["status"] = ReviewStatus.APPROVED.value
        self._write_all(items)

    def reject(self, item_id: str, reason: str = "") -> None:
        items = self._read_all()
        for item in items:
            if item["id"] == item_id:
                item["status"] = ReviewStatus.REJECTED.value
                item["reject_reason"] = reason
        self._write_all(items)

    def auto_approve(self, threshold: float = 0.9) -> int:
        items = self._read_all()
        count = 0
        for item in items:
            if item["status"] == ReviewStatus.PENDING.value and item.get("confidence", 0) >= threshold:
                item["status"] = ReviewStatus.APPROVED.value
                count += 1
        self._write_all(items)
        return count
