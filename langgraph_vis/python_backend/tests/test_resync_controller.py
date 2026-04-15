from python_backend.app.resync_controller import resolve_replay_from
from python_backend.app.run_state_store import RunStateStore


def test_from_seq_takes_precedence_and_passes_through():
    store = RunStateStore()
    store.create_run(run_id="run_resync", thread_id="thread_resync")
    assert resolve_replay_from(store, "run_resync", from_seq="2") == 2
    assert resolve_replay_from(store, "run_resync", from_seq="0") == 0


def test_from_seq_and_last_event_id_prefers_from_seq():
    store = RunStateStore()
    store.create_run(run_id="run_resync_pre", thread_id="thread_resync_pre")
    node_started = store.append_event("run_resync_pre", "node_started", {"nodeId": "n1"})

    assert (
        resolve_replay_from(
            store,
            "run_resync_pre",
            from_seq="10",
            last_event_id=node_started["eventId"],
        )
        == 10
    )


def test_last_event_id_returns_event_seq():
    store = RunStateStore()
    store.create_run(run_id="run_resync_last", thread_id="thread_resync_last")
    started = store.append_event("run_resync_last", "node_started", {"nodeId": "n1"})
    assert resolve_replay_from(store, "run_resync_last", last_event_id=started["eventId"]) == 2


def test_unknown_last_event_id_returns_zero():
    store = RunStateStore()
    store.create_run(run_id="run_resync_unknown", thread_id="thread_resync_unknown")
    assert resolve_replay_from(store, "run_resync_unknown", last_event_id="missing") == 0


def test_unknown_run_raises():
    store = RunStateStore()
    try:
        resolve_replay_from(store, "missing", last_event_id="any")
    except Exception as error:
        assert str(error) == "requested run was not found"
    else:
        raise AssertionError("expected error")


def test_invalid_last_event_id_raises_typeerror():
    store = RunStateStore()
    store.create_run(run_id="run_resync_bad", thread_id="thread_resync_bad")
    try:
        resolve_replay_from(store, "run_resync_bad", last_event_id="")
    except Exception as error:
        assert type(error).__name__ == "TypeError"
    else:
        raise AssertionError("expected TypeError")


def test_invalid_from_seq_raises_typeerror():
    store = RunStateStore()
    store.create_run(run_id="run_resync_bad2", thread_id="thread_resync_bad2")
    try:
        resolve_replay_from(store, "run_resync_bad2", from_seq="abc")
    except Exception as error:
        assert type(error).__name__ == "TypeError"
    else:
        raise AssertionError("expected TypeError")
