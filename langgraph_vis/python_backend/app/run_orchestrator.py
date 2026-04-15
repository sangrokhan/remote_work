"""Run orchestration facade for Python backend."""

from __future__ import annotations

import time
import uuid
from .run_state_store import RunStateStore


DEFAULT_RUN_PREFIX = "run"
DEFAULT_THREAD_PREFIX = "thread"


def _next_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"


def _normalize_optional_string(value, fallback: str):
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


class RunOrchestrator:
    def __init__(self, *, store: RunStateStore | None = None):
        self.store = store or RunStateStore()

    def create_run(self, *, run_id: str | None = None, thread_id: str | None = None, workflow_id: str | None = None, workflow_version: str | None = None):
        resolved_run_id = _normalize_optional_string(run_id, _next_id(DEFAULT_RUN_PREFIX))
        resolved_thread_id = _normalize_optional_string(thread_id, _next_id(DEFAULT_THREAD_PREFIX))

        event = self.store.create_run(
            run_id=resolved_run_id,
            thread_id=resolved_thread_id,
            workflow_id=workflow_id,
            workflow_version=workflow_version or "1.0.0",
        )
        return {"runId": resolved_run_id, "threadId": resolved_thread_id, "event": event}

    def emit(self, run_id: str, event_type: str, payload=None, options=None):
        return self.store.append_event(run_id, event_type, payload or {}, options or {})

    def node_started(self, run_id: str, node_id: str, options=None):
        merged_payload = {"nodeId": node_id}
        if options and isinstance(options, dict) and options.get("payload"):
            merged_payload.update(options["payload"])
        return self.emit(run_id, "node_started", merged_payload, options or {})

    def node_completed(self, run_id: str, node_id: str, options=None):
        merged_payload = {"nodeId": node_id}
        if options and isinstance(options, dict) and options.get("payload"):
            merged_payload.update(options["payload"])
        return self.emit(run_id, "node_completed", merged_payload, options or {})

    def await_input(self, run_id: str, options=None):
        payload = (options or {}).get("payload", {})
        return self.emit(run_id, "awaiting_input", payload, options or {})

    def complete_run(self, run_id: str, payload=None, options=None):
        return self.emit(run_id, "run_completed", payload or {}, options or {})

    def fail_run(self, run_id: str, error, options=None):
        merged_payload = {"error": error}
        if options and isinstance(options, dict) and options.get("payload"):
            merged_payload.update(options["payload"])
        return self.emit(run_id, "run_failed", merged_payload, options or {})

    def cancel_run(self, run_id: str, reason, options=None):
        merged_payload = {"reason": reason}
        if options and isinstance(options, dict) and options.get("payload"):
            merged_payload.update(options["payload"])
        return self.emit(run_id, "run_cancelled", merged_payload, options or {})

    def mark_recovered(self, run_id: str, options=None):
        payload = {"replayed": True}
        if options and isinstance(options, dict) and options.get("payload"):
            payload.update(options["payload"])
        return self.emit(run_id, "event_recovered", payload, options or {})

    def state(self, run_id: str):
        return self.store.get_run_state(run_id)

    def events(self, run_id: str, query=None):
        query = query or {}
        return self.store.list_events(run_id, from_seq=query.get("fromSeq"), last_event_id=query.get("lastEventId"))
