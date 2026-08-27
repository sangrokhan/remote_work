"""Recent-run store (DESIGN.md 5 point 5).

Persists to disk now: every ``save_run()`` writes the full doc to
``<run_store_dir()>/<exec_id>.json`` (default ``data/runs/``, configurable
via the ``RUN_STORE_DIR`` env var -- same pattern as
``aipt/web/routes_run.py``'s ``PUBLIC_AI_RECORDS_DIR``), and the in-memory
``OrderedDict`` (still capped at ``MAX_RUNS``, still what every read goes
through first) is lazily rehydrated from that directory the first time any
of this module's functions runs in a fresh process -- so a restart no
longer loses run history. The eviction policy (oldest run past
``MAX_RUNS``) now applies to disk too: an evicted run's JSON file is
deleted, matching token_traffic's ``core/store.py`` retention-pruned
on-disk store DESIGN.md §6 decision 5 referenced (mirrored rather than
inherited: this module is a from-scratch rewrite for AIPT's run schema).

Persistence failures (disk full, permission denied, read-only mount) are
logged and swallowed, never fatal to a run -- the same honesty-over-crash
posture ``aipt.gateway.netem_control``/``aipt.core.offload`` use elsewhere
in this codebase. A run that can't be persisted still succeeds and is
still returned to the caller; it just won't survive a restart.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

#: How many completed runs to keep -- in memory AND on disk. Evicting the
#: oldest run past this cap deletes its on-disk file too.
MAX_RUNS = 50

#: Where run JSON docs are persisted. Overridable so tests (and anything
#: else that wants isolation from a real box's accumulated run history) can
#: point this at a scratch directory -- read dynamically on every call
#: rather than cached at import time, matching
#: ``aipt/web/routes_run.py``'s ``public_ai_records_dir()``.
RUN_STORE_DIR_ENV = "RUN_STORE_DIR"
DEFAULT_RUN_STORE_DIR = "data/runs"

_lock = threading.Lock()
_runs: "OrderedDict[str, dict]" = OrderedDict()
#: Whether this process has already scanned disk to rehydrate ``_runs``.
#: Reset by ``clear()`` (test isolation: a monkeypatched RUN_STORE_DIR
#: between tests needs its own fresh rehydration pass).
_loaded_from_disk = False


def run_store_dir() -> Path:
    return Path(os.environ.get(RUN_STORE_DIR_ENV, DEFAULT_RUN_STORE_DIR))


def new_exec_id() -> str:
    return uuid.uuid4().hex[:12]


def _run_path(exec_id: str) -> Path:
    return run_store_dir() / f"{exec_id}.json"


def _write_to_disk(doc: dict) -> None:
    try:
        d = run_store_dir()
        d.mkdir(parents=True, exist_ok=True)
        _run_path(doc["exec_id"]).write_text(json.dumps(doc))
    except OSError as exc:  # pragma: no cover - defensive, disk/perm issues
        print(f"[aipt.web.store] failed to persist run {doc.get('exec_id')!r}: {exc}")


def _delete_from_disk(exec_id: str) -> None:
    try:
        _run_path(exec_id).unlink(missing_ok=True)
    except OSError:  # pragma: no cover - defensive
        pass


def _load_from_disk_locked() -> None:
    """Populate ``_runs`` from whatever's already on disk (process-restart
    recovery). Only ever called under ``_lock``; keeps at most the newest
    ``MAX_RUNS`` docs, same cap ``save_run()`` enforces going forward."""
    d = run_store_dir()
    if not d.is_dir():
        return
    docs = []
    for path in d.glob("*.json"):
        try:
            docs.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue  # corrupt/partial file -- skip, don't crash startup
    docs.sort(key=lambda doc: doc.get("saved_at", 0))
    for doc in docs[-MAX_RUNS:]:
        exec_id = doc.get("exec_id")
        if exec_id:
            _runs[exec_id] = doc


def _ensure_loaded_locked() -> None:
    global _loaded_from_disk
    if not _loaded_from_disk:
        _load_from_disk_locked()
        _loaded_from_disk = True


def save_run(doc: dict) -> dict:
    """Store *doc* under its ``exec_id`` (assigned here if absent) in
    memory AND on disk, evicting (memory + disk) the oldest run past
    ``MAX_RUNS``. Returns *doc*."""
    with _lock:
        _ensure_loaded_locked()
        exec_id = doc.get("exec_id") or new_exec_id()
        doc["exec_id"] = exec_id
        doc.setdefault("saved_at", time.time())
        _runs[exec_id] = doc
        _runs.move_to_end(exec_id)
        evicted_ids = []
        while len(_runs) > MAX_RUNS:
            evicted_id, _ = _runs.popitem(last=False)
            evicted_ids.append(evicted_id)
    # Disk I/O outside the lock -- keeps the lock held only for the
    # in-memory bookkeeping other requests are also contending for.
    _write_to_disk(doc)
    for evicted_id in evicted_ids:
        _delete_from_disk(evicted_id)
    return doc


def get_run(exec_id: str) -> dict | None:
    with _lock:
        _ensure_loaded_locked()
        doc = _runs.get(exec_id)
    if doc is not None:
        return doc
    # Not in the in-memory cap window (e.g. more than MAX_RUNS runs have
    # happened since a restart's rehydration pass) -- fall back to reading
    # its file straight off disk before giving up.
    path = _run_path(exec_id)
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None
    return None


def delete_run(exec_id: str) -> bool:
    with _lock:
        _ensure_loaded_locked()
        existed_in_memory = _runs.pop(exec_id, None) is not None
    existed_on_disk = _run_path(exec_id).is_file()
    _delete_from_disk(exec_id)
    return existed_in_memory or existed_on_disk


def list_runs() -> list[dict]:
    """Newest first; each entry is a light summary, not the full document
    (turns/monitors/pcaps are large and only the run detail/CSV endpoints
    need them)."""
    with _lock:
        _ensure_loaded_locked()
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
    """Test helper: wipes memory AND whatever's on disk under the current
    ``run_store_dir()``, and resets the rehydration flag so the next call
    re-scans disk fresh -- needed because tests monkeypatch RUN_STORE_DIR
    per-test and expect each test to start from a clean slate."""
    global _loaded_from_disk
    with _lock:
        _runs.clear()
        _loaded_from_disk = False
    d = run_store_dir()
    if d.is_dir():
        for path in d.glob("*.json"):
            try:
                path.unlink()
            except OSError:  # pragma: no cover - defensive
                pass

