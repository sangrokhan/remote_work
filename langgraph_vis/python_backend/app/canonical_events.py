"""Week 3 canonical event definitions and transformation helpers."""

from __future__ import annotations

from datetime import datetime
from copy import deepcopy
from typing import Any, Dict, Iterable, List


CANONICAL_SCHEMA_VERSION = "1.0.0"
CANONICAL_SOURCE_STREAM = "stream"

ALLOWED_EVENT_TYPES = {
    "run_started",
    "run_awaiting_input",
    "run_completed",
    "run_failed",
    "run_cancelled",
    "node_started",
    "node_token",
    "node_completed",
    "node_failed",
    "event_recovered",
}

TERMINAL_EVENT_TYPES = {"run_completed", "run_failed", "run_cancelled"}

EVENT_TYPE_ALIASES = {
    "awaiting_input": "run_awaiting_input",
}

REQUIRED_CANONICAL_FIELDS = [
    "eventId",
    "runId",
    "threadId",
    "eventSeq",
    "eventType",
    "issuedAt",
    "checkpoint",
    "canonicalMeta",
]

ALLOWED_CANONICAL_SOURCE = {
    CANONICAL_SOURCE_STREAM,
    "orchestrator",
    "replayer",
}


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise TypeError(message)


def _require_type(value: Any, name: str, *, is_event_field: bool = False) -> None:
    if is_event_field:
        _ensure(isinstance(value, str) and value.strip(), f"{name} must be non-empty string")
        return

    _ensure(isinstance(value, bool), f"{name} must be boolean")


def _normalize_event_seq(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TypeError("eventSeq must be a positive integer")
    return value


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError("payload must be object")
    return deepcopy(payload)


def _normalize_checkpoint(checkpoint: Any) -> Dict[str, Any]:
    if checkpoint is None:
        return {}
    _ensure(isinstance(checkpoint, dict), "checkpoint must be object")
    return deepcopy(checkpoint)


def _is_iso_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except Exception:
        return False


def _canonical_from_raw_type(raw_type: str) -> str:
    canonical_type = EVENT_TYPE_ALIASES.get(raw_type, raw_type)
    if canonical_type not in ALLOWED_EVENT_TYPES:
        raise TypeError(f"unsupported eventType for canonical conversion: {raw_type}")
    return canonical_type


def to_canonical_event(raw_event: Dict[str, Any], *, source: str = CANONICAL_SOURCE_STREAM) -> Dict[str, Any]:
    if not isinstance(raw_event, dict):
        raise TypeError("raw event must be object")

    raw_event_type = raw_event.get("eventType")
    _ensure_type = isinstance(raw_event_type, str) and raw_event_type.strip()
    if not _ensure_type:
        raise TypeError("eventType must be non-empty string")

    raw_payload = raw_event.get("payload", {})
    canonical_type = _canonical_from_raw_type(raw_event_type)
    payload = _normalize_payload(raw_payload)
    checkpoint = _normalize_checkpoint(raw_event.get("checkpoint"))
    _ensure(source in ALLOWED_CANONICAL_SOURCE, "canonicalMeta.source must be one of stream|orchestrator|replayer")

    event_id = raw_event.get("eventId")
    run_id = raw_event.get("runId")
    thread_id = raw_event.get("threadId")
    issued_at = raw_event.get("issuedAt")
    event_seq = raw_event.get("eventSeq")
    _require_type(event_id, "eventId", is_event_field=True)
    _require_type(run_id, "runId", is_event_field=True)
    _require_type(thread_id, "threadId", is_event_field=True)
    _require_type(issued_at, "issuedAt", is_event_field=True)
    event_seq = _normalize_event_seq(event_seq)

    canonical_meta = {
        "isTerminal": canonical_type in TERMINAL_EVENT_TYPES,
        "source": source,
        "replayable": canonical_type != "event_recovered",
        "schemaVersion": CANONICAL_SCHEMA_VERSION,
    }

    return {
        "eventId": event_id,
        "runId": run_id,
        "threadId": thread_id,
        "eventSeq": event_seq,
        "eventType": canonical_type,
        "issuedAt": issued_at,
        "payload": payload,
        "checkpoint": checkpoint or {},
        "canonicalMeta": canonical_meta,
        "nodeId": payload.get("nodeId"),
    }


def validate_canonical_event(event: Dict[str, Any]) -> None:
    if not isinstance(event, dict):
        raise TypeError("event must be object")

    for field in REQUIRED_CANONICAL_FIELDS:
        if field not in event:
            raise TypeError(f"missing required field: {field}")

    _require_type(event.get("eventId"), "eventId", is_event_field=True)
    _require_type(event.get("runId"), "runId", is_event_field=True)
    _require_type(event.get("threadId"), "threadId", is_event_field=True)
    _require_type(event.get("issuedAt"), "issuedAt", is_event_field=True)
    _ensure(event.get("eventType") in ALLOWED_EVENT_TYPES, "invalid eventType")
    _normalize_event_seq(event.get("eventSeq"))
    checkpoint = event.get("checkpoint")
    _ensure(isinstance(checkpoint, dict), "checkpoint must be object")
    payload = event.get("payload")
    _ensure(payload is None or isinstance(payload, dict), "payload must be object")
    if "state" in checkpoint:
        _ensure(isinstance(checkpoint["state"], str), "checkpoint.state must be string")
    if "eventSeq" in checkpoint:
        _ensure(isinstance(checkpoint["eventSeq"], int) and checkpoint["eventSeq"] >= 1, "checkpoint.eventSeq must be positive integer")

    meta = event.get("canonicalMeta")
    if not isinstance(meta, dict):
        raise TypeError("canonicalMeta must be object")
    _require_type(meta.get("isTerminal"), "canonicalMeta.isTerminal")
    _require_type(meta.get("replayable"), "canonicalMeta.replayable")
    _require_type(meta.get("source"), "canonicalMeta.source", is_event_field=True)
    _require_type(meta.get("schemaVersion"), "canonicalMeta.schemaVersion", is_event_field=True)

    if meta["source"] not in ALLOWED_CANONICAL_SOURCE:
        raise TypeError("canonicalMeta.source must be one of stream|orchestrator|replayer")
    if not _is_iso_datetime(event["issuedAt"]):
        raise TypeError("issuedAt must be iso datetime string")


def to_canonical_events(raw_events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(raw_events, (list, tuple)):
        raise TypeError("raw events must be array")

    seen = set()
    canonical = []
    for raw_event in raw_events:
        converted = to_canonical_event(raw_event)
        validate_canonical_event(converted)
        event_id = converted["eventId"]
        if event_id in seen:
            continue
        seen.add(event_id)
        canonical.append(converted)

    canonical.sort(key=lambda e: (e["eventSeq"], e["eventId"]))
    return canonical
