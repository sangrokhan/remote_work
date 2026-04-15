"""SSE envelope helpers for Python backend."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

REQUIRED_ENVELOPE_FIELDS = [
    "eventId",
    "runId",
    "threadId",
    "eventSeq",
    "eventType",
    "payload",
    "checkpoint",
    "issuedAt",
]


def _ensure_valid_string(value, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field_name} must be a non-empty string")


def _ensure_positive_integer(value, field_name: str) -> None:
    if not isinstance(value, int) or value < 1:
        raise TypeError(f"{field_name} must be a positive integer")


def build_sse_envelope(payload: dict | None = None) -> dict:
    source = payload or {}
    event_id = source.get("eventId", str(uuid.uuid4()))
    run_id = source.get("runId")
    thread_id = source.get("threadId")
    event_seq = source.get("eventSeq")
    event_type = source.get("eventType")
    event_payload = source.get("payload", {})
    checkpoint = source.get("checkpoint", {})
    issued_at = source.get("issuedAt", datetime.now(timezone.utc).isoformat())
    extra = {k: v for k, v in source.items() if k not in {
        "eventId",
        "runId",
        "threadId",
        "eventSeq",
        "eventType",
        "payload",
        "checkpoint",
        "issuedAt",
    }}

    _ensure_valid_string(event_id, "eventId")
    _ensure_valid_string(run_id, "runId")
    _ensure_valid_string(thread_id, "threadId")
    _ensure_positive_integer(event_seq, "eventSeq")
    _ensure_valid_string(event_type, "eventType")
    if not isinstance(issued_at, str) or not issued_at:
        raise TypeError("issuedAt must be a non-empty string")
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be an object")
    if not isinstance(event_payload, dict):
        raise TypeError("payload must be an object")

    return {
        "eventId": event_id,
        "runId": run_id,
        "threadId": thread_id,
        "eventSeq": event_seq,
        "eventType": event_type,
        "payload": event_payload,
        "checkpoint": checkpoint,
        "issuedAt": issued_at,
        **extra,
    }


def validate_envelope(envelope: dict) -> bool:
    if not isinstance(envelope, dict):
        raise TypeError("envelope must be object")

    for field in REQUIRED_ENVELOPE_FIELDS:
        if field not in envelope:
            raise TypeError(f"missing required field: {field}")

    _ensure_valid_string(envelope.get("eventId"), "eventId")
    _ensure_valid_string(envelope.get("runId"), "runId")
    _ensure_valid_string(envelope.get("threadId"), "threadId")
    _ensure_positive_integer(envelope.get("eventSeq"), "eventSeq")
    _ensure_valid_string(envelope.get("eventType"), "eventType")
    payload = envelope.get("payload")
    if not isinstance(payload, dict) and payload is not None:
        raise TypeError("payload must be object")
    if not isinstance(envelope.get("checkpoint"), dict):
        raise TypeError("checkpoint must be an object")
    if not (isinstance(envelope.get("issuedAt"), str) and envelope.get("issuedAt")):
        raise TypeError("issuedAt must be a non-empty string")

    return True
