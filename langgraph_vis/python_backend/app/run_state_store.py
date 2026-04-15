"""In-memory run state store for Python backend PoC."""

from __future__ import annotations

import uuid
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict

from .run_state_machine import (
    assert_transition,
    is_terminal_state,
    is_valid_run_state,
)
from .run_error_contract import RunNotFoundError
from .sse_envelope import build_sse_envelope

IDENTIFIERS = {
    "runId": r"^[a-z][a-z0-9_-]{1,63}$",
    "threadId": r"^[a-z][a-z0-9_-]{1,63}$",
}

EVENT_STATE_MAP = {
    "run_started": "running",
    "run_cancelled": "cancelled",
    "awaiting_input": "awaiting_input",
    "run_completed": "completed",
    "run_failed": "failed",
}


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise TypeError(message)


def _is_valid_id(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 0


def _snapshot_event(event: Dict[str, Any]) -> Dict[str, Any]:
    return dict(event)


def _make_checkpoint(state: str, event_seq: int) -> Dict[str, Any]:
    return {"state": state, "eventSeq": event_seq}


def _next_event_id_from_input(event_id: Any) -> str:
    if isinstance(event_id, str) and event_id:
        return event_id
    return str(uuid.uuid4())


def _normalize_from_seq(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        if value.strip() == "":
            return None
    if isinstance(value, bool):
        raise TypeError("fromSeq must be a positive integer")
    if isinstance(value, float) and not value.is_integer():
        raise TypeError("fromSeq must be a positive integer")
    try:
        num = int(value)
    except Exception:
        raise TypeError("fromSeq must be a positive integer")
    if num < 0:
        raise TypeError("fromSeq must be a positive integer")
    return num


def _validate_checkpoint(checkpoint: Any) -> None:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be an object")


def _build_run_dict(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "runId": run["runId"],
        "threadId": run["threadId"],
        "workflowId": run["workflowId"],
        "workflowVersion": run["workflowVersion"],
        "state": run["state"],
        "eventSeq": run["eventSeq"],
        "cursor": run["cursor"],
        "lastEventId": run["lastEventId"],
        "createdAt": run["createdAt"],
        "updatedAt": run["updatedAt"],
    }


class RunStateStore:
    def __init__(self, *, event_factory=build_sse_envelope):
        self.event_factory = event_factory
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._regexp_cache = {
            "runId": re.compile(IDENTIFIERS["runId"]),
            "threadId": re.compile(IDENTIFIERS["threadId"]),
        }

    def _build_event(self, run: Dict[str, Any], event_type: str, payload: Dict[str, Any], options: Dict[str, Any] = None) -> Dict[str, Any]:
        options = options or {}
        return self.event_factory(
            {
                "eventId": _next_event_id_from_input(options.get("eventId")),
                "runId": run["runId"],
                "threadId": run["threadId"],
                "eventSeq": run["eventSeq"] + 1,
                "eventType": event_type,
                "payload": payload,
                "checkpoint": options.get("checkpoint", _make_checkpoint(run["state"], run["eventSeq"])),
            "issuedAt": options.get("issuedAt") or datetime.now(timezone.utc).isoformat(),
                **(options.get("extraMeta") or {}),
            }
        )

    def create_run(self, *, run_id: str, thread_id: str, workflow_id: str | None = None, workflow_version: str = "1.0.0", event_id: str | None = None) -> Dict[str, Any]:
        _ensure(_is_valid_id(run_id), "runId is required")
        _ensure(_is_valid_id(thread_id), "threadId is required")
        _ensure(bool(self._regexp_cache["runId"].match(run_id)), "runId format invalid")
        _ensure(bool(self._regexp_cache["threadId"].match(thread_id)), "threadId format invalid")

        if run_id in self._runs:
            raise ValueError(f"run already exists: {run_id}")

        now = datetime.now(timezone.utc).isoformat()
        run = {
            "runId": run_id,
            "threadId": thread_id,
            "workflowId": workflow_id,
            "workflowVersion": workflow_version,
            "state": "queued",
            "eventSeq": 0,
            "events": [],
            "eventIdToSeq": {},
            "lastEventId": None,
            "createdAt": now,
            "updatedAt": now,
            "cursor": None,
        }

        self._runs[run_id] = run
        return self.append_event(
            run_id,
            "run_started",
            {"workflowId": workflow_id, "workflowVersion": workflow_version},
            {"eventId": event_id},
        )

    def append_event(self, run_id: str, event_type: str, payload: Dict[str, Any] | None = None, options: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
        run = self._runs.get(run_id)
        if not run:
            raise RunNotFoundError(run_id)

        _ensure(is_valid_run_state(run["state"]), f"invalid stored run state: {run['state']}")
        _ensure(isinstance(event_type, str) and event_type, "eventType is required")
        options = options or {}
        payload = payload or {}

        provided_event_id = options.get("eventId")
        if provided_event_id is not None:
            _ensure(isinstance(provided_event_id, str) and provided_event_id, "eventId must be a non-empty string")
            existing_seq = run["eventIdToSeq"].get(provided_event_id)
            if isinstance(existing_seq, int):
                return _snapshot_event(run["events"][existing_seq - 1])

        next_state = EVENT_STATE_MAP.get(event_type, run["state"])
        if is_terminal_state(run["state"]) and event_type != "event_recovered":
            return None
        if not is_valid_run_state(next_state):
            raise ValueError(f"invalid target state: {next_state}")

        if options.get("checkpoint") is not None:
            _validate_checkpoint(options["checkpoint"])

        if next_state != run["state"]:
            assert_transition(run["state"], next_state)

        event = self._build_event(
            run,
            event_type,
            payload,
            {
                **options,
                "checkpoint": options.get("checkpoint", _make_checkpoint(next_state, run["eventSeq"] + 1)),
            },
        )
        run["events"].append(event)
        run["eventIdToSeq"][event["eventId"]] = event["eventSeq"]
        run["lastEventId"] = event["eventId"]
        run["eventSeq"] = event["eventSeq"]
        run["state"] = next_state
        run["updatedAt"] = event["issuedAt"]
        run["cursor"] = event["checkpoint"]

        return _snapshot_event(event)

    def get_run_state(self, run_id: str) -> Dict[str, Any]:
        run = self._runs.get(run_id)
        if not run:
            raise RunNotFoundError(run_id)

        state = _build_run_dict(run)
        cursor = _make_checkpoint(state["state"], state["eventSeq"])
        if run["lastEventId"]:
            cursor["lastEventId"] = run["lastEventId"]
        state["cursor"] = cursor
        return deepcopy(state)

    def list_events(self, run_id: str, *, from_seq: int | None = None, last_event_id: str | None = None):
        run = self._runs.get(run_id)
        if not run:
            raise RunNotFoundError(run_id)

        effective_from_seq = _normalize_from_seq(from_seq)
        if effective_from_seq is None and last_event_id:
            found_seq = run["eventIdToSeq"].get(last_event_id)
            if isinstance(found_seq, int):
                effective_from_seq = found_seq
            else:
                effective_from_seq = 0

        return [
            _snapshot_event(event)
            for event in run["events"]
            if event["eventSeq"] > (effective_from_seq or 0)
        ]

    def get_event(self, run_id: str, event_seq: int):
        run = self._runs.get(run_id)
        if not run:
            raise RunNotFoundError(run_id)
        idx = event_seq - 1
        if idx >= 0 and idx < len(run["events"]):
            return _snapshot_event(run["events"][idx])
        return None
