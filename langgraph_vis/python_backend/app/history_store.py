"""Week 3 in-memory canonical history store."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from .canonical_events import CANONICAL_SCHEMA_VERSION, to_canonical_events
from .run_error_contract import RunNotFoundError


DEFAULT_HISTORY_LIMIT = 100
MAX_HISTORY_LIMIT = 500
FAILURE_CONTEXT_SCHEMA_VERSION = "1.0.0"


DEFAULT_RESOLUTION_HINTS_BY_CATEGORY = {
    "llm": [
        "Check model/provider status",
        "Retry the run with reduced context size",
        "Capture the request payload and trace identifiers",
    ],
    "io": [
        "Confirm network endpoints are reachable",
        "Retry after transient infra issues",
        "Validate API credentials and quotas",
    ],
    "state": [
        "Inspect canonical event ordering",
        "Replay from last checkpoint",
        "Validate node/transition invariants",
    ],
    "unknown": [
        "Collect request/response trace and retry once",
        "Escalate to on-call with runId and eventId",
    ],
}


def _ensure(value: bool, message: str) -> None:
    if not value:
        raise TypeError(message)


def _ensure_list_of_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _normalize_hint_list(value: Any) -> list[str]:
    hints = _ensure_list_of_strings(value)
    if hints:
        return hints
    return []


def _normalize_reference_list(value: Any) -> list[str]:
    return _ensure_list_of_strings(value)


def _fallback_resolution_hints(failure_category: str) -> list[str]:
    if not isinstance(failure_category, str) or not failure_category.strip():
        return DEFAULT_RESOLUTION_HINTS_BY_CATEGORY["unknown"]
    return DEFAULT_RESOLUTION_HINTS_BY_CATEGORY.get(failure_category, DEFAULT_RESOLUTION_HINTS_BY_CATEGORY["unknown"])


def _build_diagnostic_context(event: Dict[str, Any]) -> Dict[str, Any]:
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}

    failure_code = payload.get("failureCode") or payload.get("errorCode") or payload.get("code") or "UNKNOWN"
    failure_category = payload.get("failureCategory") or payload.get("category") or "unknown"
    if not isinstance(failure_category, str) or not failure_category.strip():
        failure_category = "unknown"
    else:
        failure_category = failure_category.strip()
    root_cause = payload.get("rootCause")
    if root_cause is None:
        root_cause = payload.get("reason") or payload.get("error") or payload.get("message")
    if root_cause is None:
        root_cause = "failure was not annotated"

    explicit_hints = _normalize_hint_list(payload.get("resolutionHints"))
    if explicit_hints:
        resolution_hints = explicit_hints
    else:
        resolution_hints = _fallback_resolution_hints(failure_category)

    return {
        "schemaVersion": FAILURE_CONTEXT_SCHEMA_VERSION,
        "eventType": event["eventType"],
        "eventId": event["eventId"],
        "eventSeq": event["eventSeq"],
        "errorAt": event["issuedAt"],
        "nodeId": event.get("nodeId"),
        "failureCode": failure_code,
        "failureCategory": failure_category,
        "retryable": payload.get("retryable"),
        "retryInfo": payload.get("retryInfo"),
        "rootCause": root_cause,
        "resolutionHints": resolution_hints,
        "evidenceRefs": _normalize_reference_list(payload.get("evidenceRefs")),
        "reason": payload.get("reason"),
        "error": payload.get("error"),
    }





def _normalize_positive_int(value: Any, *, field_name: str) -> int | None:
    if value is None or value == "":
        return None

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be integer")
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
        if not value.isdigit():
            raise TypeError(f"{field_name} must be integer")
    if isinstance(value, float):
        if not value.is_integer():
            raise TypeError(f"{field_name} must be integer")
        value = int(value)
    if not isinstance(value, int):
        try:
            value = int(value)
        except Exception:
            raise TypeError(f"{field_name} must be integer")
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be integer")

    if value < 0:
        raise TypeError(f"{field_name} must be non-negative integer")
    return value


def _normalize_limit(value: Any, *, field_name: str) -> int:
    value = _normalize_positive_int(value, field_name=field_name) or DEFAULT_HISTORY_LIMIT
    if value == 0:
        return DEFAULT_HISTORY_LIMIT
    if value > MAX_HISTORY_LIMIT:
        raise TypeError(f"{field_name} must be <= {MAX_HISTORY_LIMIT}")
    return value


def _build_node_summary(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    nodes = {}
    for event in events:
        node_id = event.get("nodeId")
        if not node_id:
            continue

        summary = nodes.setdefault(
            node_id,
            {
                "nodeId": node_id,
                "state": "unknown",
                "startSeq": None,
                "endSeq": None,
                "lastSeq": None,
                "tokenCount": 0,
            },
        )
        event_seq = event["eventSeq"]
        summary["lastSeq"] = event_seq
        if event["eventType"] == "node_started":
            summary["state"] = "running"
            summary["startSeq"] = event_seq if summary["startSeq"] is None else min(summary["startSeq"], event_seq)
        elif event["eventType"] == "node_token":
            summary["tokenCount"] += 1
            summary["state"] = "running"
        elif event["eventType"] == "node_completed":
            summary["state"] = "completed"
            summary["endSeq"] = event_seq
        elif event["eventType"] == "node_failed":
            summary["state"] = "failed"
            summary["endSeq"] = event_seq
    return nodes


def _build_failure_context(events: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    for event in reversed(events):
        if event["eventType"] != "run_failed":
            continue
        return _build_diagnostic_context(event)
    return None


class RunHistoryStore:
    def __init__(self, *, run_store):
        self.run_store = run_store

    def _snapshot_history(self, run_id: str) -> Dict[str, Any]:
        run_state = self.run_store.get_run_state(run_id)
        raw_events = self.run_store.list_events(run_id)
        canonical_events = to_canonical_events(raw_events)
        nodes = _build_node_summary(canonical_events)
        failure_context = _build_failure_context(canonical_events)
        cursor = deepcopy(run_state["cursor"])
        cursor.setdefault("schemaVersion", CANONICAL_SCHEMA_VERSION)

        return {
            "runId": run_state["runId"],
            "threadId": run_state["threadId"],
            "events": canonical_events,
            "nodes": nodes,
            "finalState": {
                "state": run_state["state"],
                "eventSeq": run_state["eventSeq"],
                "lastEventId": run_state["lastEventId"],
            },
            "failureContext": failure_context,
            "cursor": cursor,
            "totalEvents": len(canonical_events),
        }

    def get_history(
        self,
        run_id: str,
        *,
        from_seq: int | None = None,
        last_event_id: str | None = None,
        limit: int | None = None,
        node_id: str | None = None,
    ) -> Dict[str, Any]:
        history = self._snapshot_history(run_id)

        from_seq = _normalize_positive_int(from_seq, field_name="fromSeq")
        limit = _normalize_limit(limit, field_name="limit")
        if from_seq is not None and last_event_id is not None:
            raise TypeError("fromSeq and lastEventId are mutually exclusive")
        if last_event_id is not None and (not isinstance(last_event_id, str) or not last_event_id.strip()):
            raise TypeError("lastEventId must be non-empty string")

        events = history["events"]
        effective_from_seq = from_seq or 0

        if last_event_id is not None and not from_seq:
            for event in events:
                if event["eventId"] == last_event_id:
                    effective_from_seq = event["eventSeq"]
                    break

        filtered = [event for event in events if event["eventSeq"] > effective_from_seq]
        if node_id is not None and node_id.strip():
            filtered = [event for event in filtered if event.get("nodeId") == node_id]
            filtered_nodes = {
                node: summary
                for node, summary in history["nodes"].items()
                if node == node_id
            }
        else:
            filtered_nodes = history["nodes"]

        if len(filtered) > limit:
            page = filtered[:limit]
            has_more = True
            last_event = page[-1]
            state = last_event.get("checkpoint", {}).get("state") or history["finalState"]["state"]
            next_cursor = {
                "eventSeq": last_event["eventSeq"],
                "eventId": last_event["eventId"],
                "state": state,
                "runId": run_id,
                "lastEventId": last_event["eventId"],
                "schemaVersion": CANONICAL_SCHEMA_VERSION,
            }
        else:
            page = filtered
            has_more = False
            next_cursor = None

        return {
            "runId": history["runId"],
            "threadId": history["threadId"],
            "events": page,
            "nodes": filtered_nodes,
            "finalState": history["finalState"],
            "failureContext": history["failureContext"],
            "cursor": history["cursor"],
            "pagination": {
                "hasMore": has_more,
                "nextCursor": next_cursor,
            },
            "count": len(page),
            "totalEvents": history["totalEvents"],
        }
