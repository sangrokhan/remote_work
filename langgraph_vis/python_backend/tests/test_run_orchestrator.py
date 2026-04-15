from python_backend.app.run_orchestrator import RunOrchestrator
from python_backend.app.run_state_store import RunStateStore


def test_orchestrator_create_run_returns_run_started_event():
    orchestrator = RunOrchestrator()
    created = orchestrator.create_run(
        run_id="run_orch",
        thread_id="thread_orch",
        workflow_id="support_ticket_classifier_v1",
    )
    state = orchestrator.state("run_orch")

    assert created["runId"] == "run_orch"
    assert created["threadId"] == "thread_orch"
    assert created["event"]["eventType"] == "run_started"
    assert created["event"]["eventSeq"] == 1
    assert state["state"] == "running"
    assert state["cursor"]["state"] == "running"


def test_success_path_transitions():
    orchestrator = RunOrchestrator(store=RunStateStore())
    orchestrator.create_run(run_id="run_orch_path", thread_id="thread_orch_path")
    node_started = orchestrator.node_started("run_orch_path", "intent_parser")
    node_completed = orchestrator.node_completed("run_orch_path", "intent_parser")
    completed = orchestrator.complete_run("run_orch_path", {"result": "ok"})
    state = orchestrator.state("run_orch_path")

    assert node_started["eventSeq"] == 2
    assert node_completed["eventSeq"] == 3
    assert completed["eventSeq"] == 4
    assert state["state"] == "completed"
    assert state["eventSeq"] == 4


def test_awaiting_input_sets_state():
    orchestrator = RunOrchestrator()
    orchestrator.create_run(run_id="run_wait", thread_id="thread_wait")
    waiting = orchestrator.await_input("run_wait", {"payload": {"reason": "user_input_required"}})

    assert waiting["eventType"] == "awaiting_input"
    assert waiting["eventSeq"] == 2
    assert orchestrator.state("run_wait")["state"] == "awaiting_input"


def test_duplicate_event_id_is_idempotent():
    orchestrator = RunOrchestrator()
    orchestrator.create_run(run_id="run_dup", thread_id="thread_dup")
    first = orchestrator.node_started("run_dup", "intent_parser", {"eventId": "same-id"})
    second = orchestrator.node_started("run_dup", "intent_parser", {"eventId": "same-id"})

    assert first["eventSeq"] == second["eventSeq"]
    assert first["eventId"] == second["eventId"]


def test_terminal_state_ignores_follow_up():
    orchestrator = RunOrchestrator()
    orchestrator.create_run(run_id="run_fail", thread_id="thread_fail")
    failed = orchestrator.fail_run("run_fail", {"reason": "boom"})
    ignore = orchestrator.complete_run("run_fail", {"result": "late"})

    assert failed["eventType"] == "run_failed"
    assert orchestrator.state("run_fail")["state"] == "failed"
    assert ignore is None
    assert orchestrator.state("run_fail")["eventSeq"] == 2
