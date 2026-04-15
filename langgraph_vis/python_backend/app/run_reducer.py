"""Utilities to reduce run events into canonical state/cursor output."""

from __future__ import annotations

from typing import Dict, List, Any


STATE_BY_EVENT = {
    "run_started": "running",
    "awaiting_input": "awaiting_input",
    "run_completed": "completed",
    "run_failed": "failed",
    "run_cancelled": "cancelled",
}

META_ONLY_EVENTS = {"event_recovered"}


def _clone_event(event: Dict[str, Any]) -> Dict[str, Any]:
    return dict(event)


def reduce_run_events(events: List[Dict[str, Any]] = None, *, state: str = "queued", cursor: Dict[str, Any] | None = None):
    if events is None or not isinstance(events, list):
        raise TypeError("events must be array")

    known_events = set()
    sorted_events = sorted(events, key=lambda e: (e.get("eventSeq", 0), str(e.get("eventId", "")))
    )

    base_seq = cursor.get("eventSeq", 0) if cursor else 0
    current_state = state
    last_seq = base_seq
    out = []

    for raw_event in sorted_events:
        if not isinstance(raw_event, dict):
            continue

        event = _clone_event(raw_event)
        event_seq = event.get("eventSeq")
        if not isinstance(event_seq, int):
            continue
        if event_seq <= last_seq:
            continue

        event_id = event.get("eventId")
        if event_id in known_events:
            continue
        known_events.add(event_id)

        if event.get("eventType") in META_ONLY_EVENTS:
            out.append(event)
            last_seq = event_seq
            continue

        mapped_state = STATE_BY_EVENT.get(event.get("eventType"))
        if mapped_state:
            current_state = mapped_state

        out.append(event)
        last_seq = event_seq

    return {
        "state": current_state,
        "eventSeq": last_seq,
        "cursor": {
            "state": current_state,
            "eventSeq": last_seq,
        },
        "events": out,
    }
