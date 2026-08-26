"""In-memory recent-run store (Phase 1 of the web UI -- DESIGN.md 5 point 5).

TODO(persistence): this keeps only the most recent ``MAX_RUNS`` runs in a
process-local dict. File/DB persistence (mirroring token_traffic's
``core/store.py`` retention-pruned JSON-on-disk store) is out of scope for
this phase; a restart loses run history. Anything that needs to survive a
restart should read this module's docstring before assuming otherwise.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict

#: How many completed runs to keep in memory before evicting the oldest.
MAX_RUNS = 50

_lock = threading.Lock()
_runs: "OrderedDict[str, dict]" = OrderedDict()


def new_exec_id() -> str:
    return uuid.uuid4().hex[:12]


def save_run(doc: dict) -> dict:
    """Store *doc* under its ``exec_id`` (assigned here if absent), evicting
    the oldest run past ``MAX_RUNS``. Returns *doc*."""
    with _lock:
        exec_id = doc.get("exec_id") or new_exec_id()
        doc["exec_id"] = exec_id
        doc.setdefault("saved_at", time.time())
        _runs[exec_id] = doc
        _runs.move_to_end(exec_id)
        while len(_runs) > MAX_RUNS:
            _runs.popitem(last=False)
        return doc


def get_run(exec_id: str) -> dict | None:
    with _lock:
        return _runs.get(exec_id)


def delete_run(exec_id: str) -> bool:
    with _lock:
        return _runs.pop(exec_id, None) is not None


def list_runs() -> list[dict]:
    """Newest first; each entry is a light summary, not the full document
    (turns/monitors/pcaps are large and only the run detail/CSV endpoints
    need them)."""
    with _lock:
        docs = list(_runs.values())
    docs.sort(key=lambda d: d.get("saved_at", 0), reverse=True)
    return [
        {
            "exec_id": d.get("exec_id"),
            "backend": d.get("backend"),
            "arm": d.get("arm"),
            "label": d.get("label"),
            "mock": d.get("mock", False),
            "timestamp": d.get("timestamp"),
            "turn_count": len(d.get("turns") or []),
            "error": d.get("error", ""),
        }
        for d in docs
    ]


def clear() -> None:
    """Test helper."""
    with _lock:
        _runs.clear()
