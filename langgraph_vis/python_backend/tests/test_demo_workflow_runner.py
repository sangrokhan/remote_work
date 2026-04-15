import asyncio

from python_backend.app.demo_workflow_runner import DemoWorkflowRunner
from python_backend.app.run_orchestrator import RunOrchestrator
from python_backend.app.run_state_store import RunStateStore
from python_backend.app.schema_registry import WorkflowSchemaRegistry


class _SequenceRandom:
    def __init__(self, values):
        self.values = list(values)
        self._index = 0

    def __call__(self):
        value = self.values[self._index]
        self._index += 1
        return value


async def _sleep_noop(_seconds: float) -> None:
    return None


def test_runner_completes_after_expected_three_node_path():
    store = RunStateStore()
    orchestrator = RunOrchestrator(store=store)
    orchestrator.create_run(
        run_id="run_demo",
        thread_id="thread_demo",
        workflow_id="langgraph_demo_workflow_v1",
    )

    runner = DemoWorkflowRunner(
        orchestrator=orchestrator,
        schema_registry=WorkflowSchemaRegistry(manifest_path="docs/week-01/schema-contract.fixture.json"),
        random_fn=_SequenceRandom([0.2, 0.2, 0.2, 0.9]).__call__,
        delay_fn=_sleep_noop,
        loop_probability=0.35,
        max_steps=12,
    )

    asyncio.run(runner.run_workflow("run_demo"))

    events = store.list_events("run_demo")
    run_completed = [event for event in events if event["eventType"] == "run_completed"]
    node_starts = [event for event in events if event["eventType"] == "node_started"]
    node_completions = [event for event in events if event["eventType"] == "node_completed"]

    assert len(node_starts) == 3
    assert len(node_completions) == 3
    assert len(run_completed) == 1

    node_order = [event["nodeId"] for event in node_starts]
    assert node_order == ["dummy_node_1", "dummy_node_2", "dummy_node_3"]

    final_state = store.get_run_state("run_demo")
    assert final_state["state"] == "completed"


def test_runner_supports_loop_back_to_first_node():
    store = RunStateStore()
    orchestrator = RunOrchestrator(store=store)
    orchestrator.create_run(
        run_id="run_loop_demo",
        thread_id="thread_loop_demo",
        workflow_id="langgraph_demo_workflow_v1",
    )

    runner = DemoWorkflowRunner(
        orchestrator=orchestrator,
        schema_registry=WorkflowSchemaRegistry(manifest_path="docs/week-01/schema-contract.fixture.json"),
        random_fn=_SequenceRandom([0.1] * 20).__call__,
        delay_fn=_sleep_noop,
        loop_probability=1.0,
        max_steps=6,
    )

    asyncio.run(runner.run_workflow("run_loop_demo"))

    events = store.list_events("run_loop_demo")
    node_starts = [event for event in events if event["eventType"] == "node_started"]
    node_completions = [event for event in events if event["eventType"] == "node_completed"]

    assert len(node_starts) == 6
    assert len(node_completions) == 6
    assert node_starts[0]["nodeId"] == "dummy_node_1"
    assert node_starts[-1]["nodeId"] == "dummy_node_3"
    assert store.get_run_state("run_loop_demo")["state"] == "completed"
