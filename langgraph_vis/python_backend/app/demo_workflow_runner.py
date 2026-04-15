"""Simple demo workflow runner for LangGraph-shaped executions."""

from __future__ import annotations

import asyncio
import uuid
from typing import Awaitable, Callable

from .run_error_contract import RunApiError, RUN_ERROR_CODES, RUN_ERROR_MESSAGES
from .run_orchestrator import RunOrchestrator
from .schema_registry import WorkflowSchemaRegistry

SleepFn = Callable[[float], Awaitable[None]]
RandomFn = Callable[[], float]


class DemoWorkflowRunner:
    """Generate `node_started`/`node_completed` events in sequence for a workflow."""

    def __init__(
        self,
        *,
        orchestrator: RunOrchestrator,
        schema_registry: WorkflowSchemaRegistry | None = None,
        *,
        random_fn: RandomFn | None = None,
        delay_fn: SleepFn | None = None,
        loop_probability: float = 0.35,
        max_steps: int = 24,
        min_node_delay_seconds: float = 3.0,
        max_node_delay_seconds: float = 5.0,
    ):
        self.orchestrator = orchestrator
        self.schema_registry = schema_registry or WorkflowSchemaRegistry()
        self.random_fn = random_fn or __import__("random").random
        self.delay_fn = delay_fn or asyncio.sleep
        self.loop_probability = loop_probability
        self.max_steps = max_steps
        self.min_node_delay_seconds = min_node_delay_seconds
        self.max_node_delay_seconds = max_node_delay_seconds

    async def run_workflow(self, run_id: str, *, workflow_id: str | None = None) -> None:
        """Run a workflow by emitting events to the run store."""
        run_state = self.orchestrator.state(run_id)
        effective_workflow_id = workflow_id or run_state.get("workflowId")

        if not effective_workflow_id:
            raise RunApiError(
                400,
                RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"],
                f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RUN_PAYLOAD']]}: workflowId is required",
            )

        request_id = str(uuid.uuid4())
        workflow = await self.schema_registry.get_by_id(effective_workflow_id, {"requestId": request_id})
        nodes = self._sorted_nodes(workflow.get("nodes", []))
        if not nodes:
            raise RunApiError(
                400,
                RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"],
                f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RUN_PAYLOAD']]}: workflow has no nodes",
            )

        next_by_from = self._build_next_by_from(workflow.get("edges", []))
        start_node_id = nodes[0]["id"]
        end_node_id = nodes[-1]["id"]
        current_node_id = start_node_id
        next_node_fallback = None

        step = 0
        for step in range(1, self.max_steps + 1):
            run_state = self.orchestrator.state(run_id)
            if run_state["state"] != "running":
                return

            self.orchestrator.node_started(
                run_id,
                current_node_id,
                {
                    "iteration": step,
                    "workflowId": effective_workflow_id,
                },
            )
            await self.delay_fn(self._next_delay_seconds())

            run_state = self.orchestrator.state(run_id)
            if run_state["state"] != "running":
                return

            self.orchestrator.node_completed(
                run_id,
                current_node_id,
                {
                    "iteration": step,
                    "workflowId": effective_workflow_id,
                },
            )

            if current_node_id == end_node_id:
                if self._should_loop_back() and step < self.max_steps:
                    current_node_id = start_node_id
                    continue
                break

            next_node_fallback = next_by_from.get(current_node_id)
            if next_node_fallback is None:
                break
            current_node_id = next_node_fallback

        self.orchestrator.complete_run(
            run_id,
            {
                "workflowId": effective_workflow_id,
                "steps": step,
                "status": "completed",
            },
        )

    def _next_delay_seconds(self) -> float:
        delay_ratio = self.random_fn()
        low = min(self.min_node_delay_seconds, self.max_node_delay_seconds)
        high = max(self.min_node_delay_seconds, self.max_node_delay_seconds)
        normalized = max(0.0, min(1.0, delay_ratio))
        return low + (high - low) * normalized

    def _should_loop_back(self) -> bool:
        return self.random_fn() < self.loop_probability

    def _sorted_nodes(self, nodes):
        return sorted(
            nodes,
            key=lambda node: int(node.get("order", 0)),
        )

    def _build_next_by_from(self, edges):
        next_by_from = {}
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source = edge.get("from")
            target = edge.get("to")
            if isinstance(source, str) and isinstance(target, str) and source and target:
                next_by_from.setdefault(source, target)
        return next_by_from
