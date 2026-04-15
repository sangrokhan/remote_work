"""Replay cursor helpers for run event streaming endpoints."""

from __future__ import annotations


def resolve_replay_from(store, run_id, *, from_seq=None, last_event_id=None):
    if from_seq is None and last_event_id is None:
        return 0

    if from_seq is not None:
        try:
            parsed = int(from_seq)
        except Exception:
            raise TypeError("fromSeq must be a positive integer")
        if parsed < 0:
            raise TypeError("fromSeq must be a positive integer")
        return parsed

    if not isinstance(last_event_id, str) or not last_event_id:
        raise TypeError("lastEventId must be a non-empty string")

    all_events = store.list_events(run_id)
    for event in all_events:
        if event.get("eventId") == last_event_id:
            return event.get("eventSeq", 0)
    return 0
