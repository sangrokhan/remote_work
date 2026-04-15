"""Week 3 history API for canonical events and node summaries."""

from __future__ import annotations

import re
import uuid
from typing import Optional

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse

from .history_store import RunHistoryStore
from .run_error_contract import (
    RUN_ERROR_CODES,
    RUN_ERROR_MESSAGES,
    RunApiError,
    normalize_run_error,
)
from .resync_controller import resolve_replay_from

RUN_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")


def _parse_query(request: Request):
    params = request.query_params
    from_seq = params.get("fromSeq")
    last_event_id = params.get("lastEventId")
    limit = params.get("limit")
    node_id = params.get("nodeId")

    from_seq = None if from_seq is None or str(from_seq).strip() == "" else str(from_seq)
    last_event_id = None if last_event_id is None else str(last_event_id)
    limit = None if limit is None or str(limit).strip() == "" else str(limit)
    node_id = None if node_id is None or str(node_id).strip() == "" else str(node_id)
    return {
        "fromSeq": from_seq,
        "lastEventId": last_event_id,
        "limit": limit,
        "nodeId": node_id,
    }


def _method_not_allowed_payload():
    return {
        "code": RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"],
        "message": RUN_ERROR_MESSAGES[RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"]],
    }


def _build_error_payload(exc: BaseException, *, request_id: str):
    normalized = normalize_run_error(exc, request_id)
    return JSONResponse(status_code=normalized["status"], content=normalized["body"])


def _as_run_error_for_typeerror(error: BaseException, query: dict, request_id: str) -> RunApiError:
    has_cursor_keys = query["fromSeq"] is not None or query["lastEventId"] is not None
    if has_cursor_keys:
        return RunApiError(
            400,
            RUN_ERROR_CODES["INVALID_RECONNECT_QUERY"],
            f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RECONNECT_QUERY']]}: {error}",
            request_id,
            error,
        )
    return RunApiError(
        400,
        RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"],
        f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RUN_PAYLOAD']]}: {error}",
        request_id,
        error,
    )


def _parse_limit(raw_limit: str | None, request_id: str) -> int | None:
    if raw_limit is None:
        return None

    try:
        limit = int(raw_limit)
    except Exception as error:
        raise RunApiError(
            400,
            RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"],
            f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RUN_PAYLOAD']]}: invalid limit",
            request_id,
            error,
        )

    if limit <= 0:
        raise RunApiError(
            400,
            RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"],
            f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RUN_PAYLOAD']]}: limit must be positive integer",
            request_id,
        )
    return limit


def create_run_history_router(*, store, history_store: RunHistoryStore | None = None):
    router = APIRouter()
    history_store = history_store or RunHistoryStore(run_store=store)

    async def _history_payload(run_id: str, request: Request, header_last_event_id: Optional[str], *, request_id: str):
        query = _parse_query(request)
        header_last_event_id = header_last_event_id.strip() if header_last_event_id and header_last_event_id.strip() else None
        if header_last_event_id is None and request.headers.get("last-event-id") is not None:
            maybe_header = request.headers.get("last-event-id")
            if isinstance(maybe_header, str) and maybe_header.strip() == "":
                raise RunApiError(
                    400,
                    RUN_ERROR_CODES["INVALID_RECONNECT_QUERY"],
                    f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RECONNECT_QUERY']]}: invalid last-event-id header",
                    request_id,
                )
        if query["lastEventId"] is not None and str(query["lastEventId"]).strip() == "":
            raise RunApiError(
                400,
                RUN_ERROR_CODES["INVALID_RECONNECT_QUERY"],
                f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RECONNECT_QUERY']]}: invalid last-event-id query",
                request_id,
            )

        if query["fromSeq"] is not None and (query["lastEventId"] is not None or header_last_event_id is not None):
            raise RunApiError(
                400,
                RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"],
                f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RUN_PAYLOAD']]}: fromSeq cannot be used with lastEventId",
                request_id,
            )
        if query["lastEventId"] is None and header_last_event_id is not None:
            query["lastEventId"] = header_last_event_id

        replay_from_seq = resolve_replay_from(
            store,
            run_id,
            from_seq=query["fromSeq"],
            last_event_id=query["lastEventId"],
        )

        limit = _parse_limit(query["limit"], request_id)
        return history_store.get_history(
            run_id,
            from_seq=replay_from_seq,
            node_id=query["nodeId"],
            limit=limit,
        )

    @router.get("/api/runs/{run_id}/history")
    async def get_history(
        run_id: str,
        request: Request,
        last_event_id: Optional[str] = Header(default=None, alias="last-event-id"),
    ):
        request_id = str(uuid.uuid4())
        if not RUN_ID.match(run_id):
            return _build_error_payload(
                RunApiError(
                    400,
                    RUN_ERROR_CODES["INVALID_RUN_ID"],
                    f"{RUN_ERROR_MESSAGES[RUN_ERROR_CODES['INVALID_RUN_ID']]}: {run_id}",
                    request_id,
                ),
                request_id=request_id,
            )
        query = _parse_query(request)
        try:
            return await _history_payload(run_id, request, last_event_id, request_id=request_id)
        except RunApiError as error:
            return _build_error_payload(error, request_id=request_id)
        except TypeError as error:
            return _build_error_payload(_as_run_error_for_typeerror(error, query, request_id), request_id=request_id)
        except Exception as error:
            return _build_error_payload(error, request_id=request_id)

    @router.api_route("/api/runs/{run_id}/history", methods=["POST", "PUT", "PATCH", "DELETE"])
    async def get_history_not_allowed(run_id: str):
        request_id = str(uuid.uuid4())
        payload = {**_method_not_allowed_payload(), "requestId": request_id}
        return JSONResponse(status_code=405, content=payload, headers={"Allow": "GET"})

    return router
