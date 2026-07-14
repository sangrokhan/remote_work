"""The store's two jobs: round-trip a run, and never let the disk fill with
synthetic runs that a chart cannot tell apart from real ones."""

import json

import pytest

from core import store
from core.record import SCHEMA_VERSION


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFIC_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("TRAFFIC_RETENTION_KEEP", raising=False)
    return tmp_path


def a_run(exec_id, *, mock=False, n=0):
    # The timestamp is what retention orders by, so give each run a distinct one.
    return {
        "exec_id": exec_id,
        "timestamp": f"2026-07-14T00:00:{n:02d}+00:00",
        "mock": mock,
        "params": {"measure": "both", "providers": ["gemini"], "turns": 3},
        "records": [{"provider": "gemini", "arm": "stateless", "phase": "steady",
                     "turn": 1, "wire_sent": 1000, "error": ""}],
        "summary": {"totals": {}, "failures": []},
    }


def test_a_run_round_trips(data_dir):
    store.save_run(a_run("exec_a"))
    got = store.get_run("exec_a")
    assert got["exec_id"] == "exec_a"
    assert got["params"]["measure"] == "both"
    assert got["records"][0]["wire_sent"] == 1000


def test_save_writes_one_json_file_per_run(data_dir):
    store.save_run(a_run("exec_a"))
    store.save_run(a_run("exec_b", n=1))
    assert sorted(p.name for p in data_dir.glob("*.json")) == ["exec_a.json",
                                                               "exec_b.json"]


def test_every_saved_run_carries_the_schema_version(data_dir):
    # A run written by an older layout must be identifiable, not silently charted
    # beside a current one whose columns mean something else.
    res = store.save_run(a_run("exec_a"))
    assert res["schema_version"] == SCHEMA_VERSION
    on_disk = json.loads((data_dir / "exec_a.json").read_text())
    assert on_disk["schema_version"] == SCHEMA_VERSION
    assert store.list_runs()["runs"][0]["schema_version"] == SCHEMA_VERSION


def test_an_exec_id_is_minted_when_the_run_has_none(data_dir):
    res = store.save_run({"params": {}, "records": []})
    assert res["ok"] and res["exec_id"].startswith("exec_")
    assert store.get_run(res["exec_id"]) is not None


def test_a_missing_run_is_none_not_an_explosion(data_dir):
    assert store.get_run("exec_nope") is None


def test_a_traversing_exec_id_is_refused(data_dir):
    assert store.get_run("../../etc/passwd") is None
    assert store.delete_run("../../etc/passwd")["ok"] is False


def test_delete_removes_the_file(data_dir):
    store.save_run(a_run("exec_a"))
    assert store.delete_run("exec_a")["ok"] is True
    assert store.get_run("exec_a") is None
    assert store.delete_run("exec_a")["ok"] is False


def test_a_corrupt_file_does_not_take_the_listing_down(data_dir):
    store.save_run(a_run("exec_a"))
    (data_dir / "exec_bad.json").write_text("{not json")
    assert [r["exec_id"] for r in store.list_runs()["runs"]] == ["exec_a"]


def test_the_listing_is_newest_first(data_dir):
    for i in range(3):
        store.save_run(a_run(f"exec_{i}", n=i))
    assert [r["exec_id"] for r in store.list_runs()["runs"]] == ["exec_2", "exec_1",
                                                                 "exec_0"]


# --- retention -----------------------------------------------------------------

def test_save_prunes_to_the_newest_keep_runs(monkeypatch, data_dir):
    # The previous layout had no policy and grew to 122 files of mostly synthetic
    # runs. Pruning on save means retention holds without anyone remembering it.
    monkeypatch.setenv("TRAFFIC_RETENTION_KEEP", "3")
    for i in range(6):
        store.save_run(a_run(f"exec_{i}", n=i))
    kept = [r["exec_id"] for r in store.list_runs()["runs"]]
    assert kept == ["exec_5", "exec_4", "exec_3"]
    assert len(list(data_dir.glob("*.json"))) == 3


def test_the_keep_limit_defaults_to_twenty(data_dir):
    assert store.retention_keep() == 20
    assert store.list_runs()["keep"] == 20


def test_prune_can_be_called_with_an_explicit_keep(data_dir):
    for i in range(5):
        store.save_run(a_run(f"exec_{i}", n=i))
    out = store.prune(keep=2)
    assert out["deleted_live"] == 3
    assert len(store.list_runs()["runs"]) == 2


def test_mock_runs_live_in_their_own_bucket(data_dir):
    store.save_run(a_run("exec_m", mock=True))
    listing = store.list_runs()
    assert listing["runs"] == []
    assert [r["exec_id"] for r in listing["mock_runs"]] == ["exec_m"]
    assert listing["mock_runs"][0]["mock"] is True
    # Still fetchable by id -- kept apart, not hidden.
    assert store.get_run("exec_m")["mock"] is True


def test_a_run_marked_mock_in_params_lands_in_the_mock_bucket(data_dir):
    store.save_run({"exec_id": "exec_m", "params": {"mock": True}, "records": []})
    assert [r["exec_id"] for r in store.list_runs()["mock_runs"]] == ["exec_m"]


def test_mock_runs_never_evict_a_live_one(monkeypatch, data_dir):
    # This is the whole reason for the split: a week of offline development must not
    # push out the one run somebody actually paid for.
    monkeypatch.setenv("TRAFFIC_RETENTION_KEEP", "2")
    store.save_run(a_run("exec_live", n=0))
    for i in range(10):
        store.save_run(a_run(f"exec_mock_{i}", mock=True, n=i))
    listing = store.list_runs()
    assert [r["exec_id"] for r in listing["runs"]] == ["exec_live"]
    assert len(listing["mock_runs"]) == 2      # mock has its own budget


def test_live_runs_do_not_evict_the_mock_bucket_either(monkeypatch, data_dir):
    monkeypatch.setenv("TRAFFIC_RETENTION_KEEP", "2")
    store.save_run(a_run("exec_m", mock=True, n=0))
    for i in range(5):
        store.save_run(a_run(f"exec_live_{i}", n=i))
    listing = store.list_runs()
    assert len(listing["runs"]) == 2
    assert [r["exec_id"] for r in listing["mock_runs"]] == ["exec_m"]
