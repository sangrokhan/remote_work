"""FastAPI router for run state/events APIs."""

from __future__ import annotations

import asyncio
import re
import uuid
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .run_error_contract import (
    RUN_ERROR_CODES,
    RUN_ERROR_MESSAGES,
    RunApiError,
    normalize_run_error,
)
from .resync_controller import resolve_replay_from
from .run_state_machine import is_terminal_state

HEARTBEAT_MS = 20_000
IDLE_TIMEOUT_MS = 90_000
RUN_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")


def _parse_query(request: Request):
    params = request.query_params
    from_seq = params.get("fromSeq")
    last_event_id = params.get("lastEventId")
    from_seq = None if from_seq is None or str(from_seq).strip() == "" else str(from_seq)
    if last_event_id is None:
        last_event_id = None
    elif str(last_event_id).strip() == "":
        last_event_id = ""
    else:
        last_event_id = str(last_event_id)
    return {"fromSeq": from_seq, "lastEventId": last_event_id}


def _parse_replay_cursor(request: Request, header_last_event_id: Optional[str], *, request_id: str):
    query = _parse_query(request)

    if query["lastEventId"] is not None:
        if query["lastEventId"] == "":
            raise RunApiError(
                400,
                RUN_ERROR_CODES["INVALID_RECONNECT_QUERY"],
                f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RECONNECT_QUERY']]}: invalid last-event-id query",
                request_id,
            )
        if query["fromSeq"] is not None:
            raise RunApiError(
                400,
                RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"],
                f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RUN_PAYLOAD']]}: fromSeq cannot be used with lastEventId",
                request_id,
            )

    if header_last_event_id is not None:
        normalized_header_last_event_id = header_last_event_id.strip()
        if not normalized_header_last_event_id:
            raise RunApiError(
                400,
                RUN_ERROR_CODES["INVALID_RECONNECT_QUERY"],
                f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RECONNECT_QUERY']]}: invalid last-event-id header",
                request_id,
            )
        if query["fromSeq"] is not None:
            raise RunApiError(
                400,
                RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"],
                f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RUN_PAYLOAD']]}: fromSeq cannot be used with lastEventId",
                request_id,
            )
        query["lastEventId"] = normalized_header_last_event_id

    return {
        "fromSeq": query["fromSeq"],
        "lastEventId": query["lastEventId"],
    }


def _sse_event_payload(event: dict) -> str:
    return "".join(
        [
            f"id: {event['eventId']}\n",
            "event: run-event\n",
            f"data: {__import__('json').dumps(event)}\n\n",
        ]
    )


def _heartbeat(now: datetime | None = None) -> str:
    now_iso = (now or datetime.now(timezone.utc)).isoformat()
    return f": heartbeat {now_iso}\n\n"


def create_run_state_router(*, store):
    router = APIRouter()

    def _method_not_allowed_payload():
        return {
            "code": RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"],
            "message": RUN_ERROR_MESSAGES[RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"]],
        }

    async def _on_error(error: BaseException, request_id: str):
        normalized = normalize_run_error(error, request_id)
        return JSONResponse(status_code=normalized["status"], content=normalized["body"])

    async def _list_events(run_id: str, request_id: str, request: Request, header_last_event_id: Optional[str]):
        cursor = _parse_replay_cursor(request, header_last_event_id, request_id=request_id)
        run_id_match = RUN_ID.match(run_id)
        if not run_id_match:
            raise RunApiError(
                400,
                RUN_ERROR_CODES["INVALID_RUN_ID"],
                f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RUN_ID']]}: {run_id}",
                request_id,
            )

        try:
            replay_from_seq = resolve_replay_from(
                store,
                run_id,
                from_seq=cursor["fromSeq"],
                last_event_id=cursor["lastEventId"],
            )
        except TypeError as error:
            raise RunApiError(
                    400,
                    RUN_ERROR_CODES["INVALID_RECONNECT_QUERY"],
                f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RECONNECT_QUERY']]}: {error}",
                    request_id,
                    error,
                ) from error
        return store.list_events(run_id, from_seq=replay_from_seq)

    @router.get("/api/runs/{run_id}/state")
    async def get_state(run_id: str, request: Request):
        request_id = str(uuid.uuid4())
        try:
            if not RUN_ID.match(run_id):
                raise RunApiError(
                    400,
                    RUN_ERROR_CODES["INVALID_RUN_ID"],
                    f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RUN_ID']]}: {run_id}",
                    request_id,
                )
            state = store.get_run_state(run_id)
            return state
        except Exception as error:
            return await _on_error(error, request_id)

    @router.api_route("/api/runs/{run_id}/state", methods=["POST", "PUT", "PATCH", "DELETE"])
    async def get_state_not_allowed(run_id: str, request: Request):
        request_id = str(uuid.uuid4())
        payload = {**_method_not_allowed_payload(), "requestId": request_id}
        return JSONResponse(status_code=405, content=payload, headers={"Allow": "GET"})

    @router.get("/api/runs/{run_id}/events")
    async def get_events(
        run_id: str,
        request: Request,
        last_event_id: Optional[str] = Header(default=None, alias="last-event-id"),
    ):
        request_id = str(uuid.uuid4())
        try:
            events = await _list_events(run_id, request_id, request, last_event_id)
            run_state = store.get_run_state(run_id)
            return {"runId": run_id, "events": events, "cursor": run_state["cursor"]}
        except Exception as error:
            return await _on_error(error, request_id)

    @router.api_route("/api/runs/{run_id}/events", methods=["POST", "PUT", "PATCH", "DELETE"])
    async def get_events_not_allowed(run_id: str):
        request_id = str(uuid.uuid4())
        payload = {**_method_not_allowed_payload(), "requestId": request_id}
        return JSONResponse(status_code=405, content=payload, headers={"Allow": "GET"})

    def _stream_events(
        run_id: str,
        request: Request,
        header_last_event_id: Optional[str],
    ):
        request_id = str(uuid.uuid4())
        cursor = _parse_replay_cursor(request, header_last_event_id, request_id=request_id)
        run_id_match = RUN_ID.match(run_id)
        if not run_id_match:
            raise RunApiError(
                    400,
                    RUN_ERROR_CODES["INVALID_RUN_ID"],
                f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RUN_ID']]}: {run_id}",
                    request_id,
                )

        try:
            replay_from_seq = resolve_replay_from(
                store,
                run_id,
                from_seq=cursor["fromSeq"],
                last_event_id=cursor["lastEventId"],
            )
        except TypeError as error:
            raise RunApiError(
                400,
                RUN_ERROR_CODES["INVALID_RECONNECT_QUERY"],
                f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RECONNECT_QUERY']]}: {error}",
                request_id,
                error,
            ) from error

        next_seq = replay_from_seq

        async def _generator():
            nonlocal next_seq
            last_activity_ms = time.time() * 1000
            last_heartbeat_ms = 0.0

            # Emit snapshot events first.
            while True:
                current_events = store.list_events(run_id, from_seq=next_seq)
                if current_events:
                    for event in current_events:
                        if event["eventSeq"] > next_seq:
                            yield _sse_event_payload(event)
                            next_seq = event["eventSeq"]
                    last_activity_ms = time.time() * 1000
                    last_heartbeat_ms = last_activity_ms
                else:
                    run_state = store.get_run_state(run_id)
                    if is_terminal_state(run_state["state"]):
                        break

                    now_ms = time.time() * 1000
                    if now_ms - last_activity_ms >= IDLE_TIMEOUT_MS:
                        break

                    if now_ms - last_heartbeat_ms >= HEARTBEAT_MS:
                        yield _heartbeat(datetime.now(timezone.utc))
                        last_heartbeat_ms = now_ms

                    await asyncio.sleep(min((HEARTBEAT_MS / 1000.0), 0.5))

                if not current_events:
                    pass

        return StreamingResponse(
            _generator(),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/api/runs/{run_id}/events/stream")
    async def event_stream(
        run_id: str,
        request: Request,
        last_event_id: Optional[str] = Header(default=None, alias="last-event-id"),
    ):
        request_id = str(uuid.uuid4())
        try:
            return _stream_events(run_id, request, last_event_id)
        except Exception as error:
            normalized = normalize_run_error(error, request_id)
            return JSONResponse(status_code=normalized["status"], content=normalized["body"])

    @router.api_route("/api/runs/{run_id}/events/stream", methods=["POST", "PUT", "PATCH", "DELETE"])
    async def event_stream_not_allowed(run_id: str):
        request_id = str(uuid.uuid4())
        payload = {**_method_not_allowed_payload(), "requestId": request_id}
        return JSONResponse(status_code=405, content=payload, headers={"Allow": "GET"})

    return router
