"""aipt.web.store -- disk persistence for the run store (restart survives).

Covers what test_app.py's end-to-end round trip doesn't isolate on its own:
eviction deleting on-disk files, rehydration into a *fresh* process after
"restart" (simulated by resetting the module's in-memory state), the
get_run() disk fallback for a run outside the in-memory MAX_RUNS window,
and persistence failures being swallowed rather than raised.
"""

from __future__ import annotations

import json

import pytest

from aipt.web import store as run_store


@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv(run_store.RUN_STORE_DIR_ENV, str(tmp_path / "runs"))
    run_store.clear()
    yield tmp_path / "runs"
    run_store.clear()


def _simulate_restart():
    """Force the next store call to rehydrate from disk, as if this were a
    fresh process -- without touching the (monkeypatched) RUN_STORE_DIR
    env var or the files already written there."""
    with run_store._lock:
        run_store._runs.clear()
        run_store._loaded_from_disk = False


def test_save_run_writes_json_to_disk(isolated_store):
    run_dir = isolated_store
    doc = run_store.save_run({"backend": "mock", "arm": "dummy", "turns": []})
    exec_id = doc["exec_id"]

    on_disk = json.loads((run_dir / f"{exec_id}.json").read_text())
    assert on_disk["exec_id"] == exec_id
    assert on_disk["backend"] == "mock"


def test_restart_rehydrates_run_history(isolated_store):
    doc = run_store.save_run({"backend": "mock", "arm": "dummy", "turns": []})
    exec_id = doc["exec_id"]

    _simulate_restart()

    # Nothing in memory yet -- list_runs()/get_run() must trigger a fresh
    # disk scan and find it anyway.
    assert any(r["exec_id"] == exec_id for r in run_store.list_runs())
    assert run_store.get_run(exec_id)["backend"] == "mock"


def test_delete_run_removes_disk_file(isolated_store):
    run_dir = isolated_store
    doc = run_store.save_run({"backend": "mock", "arm": "dummy", "turns": []})
    exec_id = doc["exec_id"]
    assert (run_dir / f"{exec_id}.json").exists()

    assert run_store.delete_run(exec_id) is True
    assert not (run_dir / f"{exec_id}.json").exists()
    assert run_store.get_run(exec_id) is None
    # Second delete: nothing left in memory or on disk either way.
    assert run_store.delete_run(exec_id) is False


def test_eviction_past_max_runs_deletes_oldest_disk_file(isolated_store, monkeypatch):
    run_dir = isolated_store
    monkeypatch.setattr(run_store, "MAX_RUNS", 2)

    first = run_store.save_run({"backend": "mock", "arm": "dummy", "turns": []})
    run_store.save_run({"backend": "mock", "arm": "dummy", "turns": []})
    run_store.save_run({"backend": "mock", "arm": "dummy", "turns": []})

    # Oldest (first) was evicted from both memory and disk.
    assert run_store.get_run(first["exec_id"]) is None
    assert not (run_dir / f"{first['exec_id']}.json").exists()
    assert len(run_store.list_runs()) == 2


def test_get_run_falls_back_to_disk_outside_memory_window(isolated_store, monkeypatch):
    """A run evicted from the in-memory OrderedDict but whose file wasn't
    deleted (can't happen via save_run()'s own eviction, but simulates a
    process that rehydrated only its newest MAX_RUNS after a very long
    history) is still found via the disk fallback in get_run()."""
    run_dir = isolated_store
    doc = run_store.save_run({"backend": "mock", "arm": "dummy", "turns": []})
    exec_id = doc["exec_id"]

    # Evict from memory only, without touching the file on disk.
    with run_store._lock:
        run_store._runs.pop(exec_id, None)

    assert (run_dir / f"{exec_id}.json").exists()
    fetched = run_store.get_run(exec_id)
    assert fetched is not None
    assert fetched["exec_id"] == exec_id


def test_corrupt_run_file_is_skipped_not_fatal(isolated_store):
    run_dir = isolated_store
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "corrupt.json").write_text("{not valid json")

    _simulate_restart()

    # Rehydration must not raise despite the corrupt file.
    assert run_store.list_runs() == []


def test_save_run_survives_unwritable_store_dir(tmp_path, monkeypatch):
    """A RUN_STORE_DIR that can't actually be created/written to (e.g. a
    read-only mount) must not turn a successful run into a 500 -- the run
    still completes and is still returned; it just won't persist."""
    unwritable = tmp_path / "not_a_dir"
    unwritable.write_text("this is a file, not a directory")
    monkeypatch.setenv(run_store.RUN_STORE_DIR_ENV, str(unwritable / "runs"))
    run_store.clear()
    try:
        doc = run_store.save_run({"backend": "mock", "arm": "dummy", "turns": []})
        assert doc["exec_id"]
        # It's still retrievable from the in-memory cache in this same
        # process even though the disk write failed.
        assert run_store.get_run(doc["exec_id"]) is not None
    finally:
        run_store.clear()
