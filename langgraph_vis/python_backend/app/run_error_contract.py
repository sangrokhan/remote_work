"""Run API error contract and helpers for the Python backend."""

from __future__ import annotations

import uuid
from typing import Any, Dict


RUN_ERROR_CODES = {
    "RUN_NOT_FOUND": "RUN_NOT_FOUND",
    "INVALID_RUN_ID": "INVALID_RUN_ID",
    "INVALID_RUN_PAYLOAD": "INVALID_RUN_PAYLOAD",
    "INVALID_RECONNECT_QUERY": "INVALID_RECONNECT_QUERY",
    "INVALID_RUN_TRANSITION": "INVALID_RUN_TRANSITION",
    "INTERNAL_ERROR": "INTERNAL_ERROR",
}

RUN_ERROR_MESSAGES = {
    RUN_ERROR_CODES["RUN_NOT_FOUND"]: "requested run was not found",
    RUN_ERROR_CODES["INVALID_RUN_ID"]: "invalid run id",
    RUN_ERROR_CODES["INVALID_RUN_PAYLOAD"]: "invalid run api payload",
    RUN_ERROR_CODES["INVALID_RECONNECT_QUERY"]: "invalid reconnect query",
    RUN_ERROR_CODES["INVALID_RUN_TRANSITION"]: "invalid run state transition",
    RUN_ERROR_CODES["INTERNAL_ERROR"]: "internal server error",
}


class RunApiError(Exception):
    def __init__(self, status: int, code: str, message: str, request_id: str | None = None, cause: BaseException | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.request_id = request_id or str(uuid.uuid4())
        self.cause = cause


class RunNotFoundError(RunApiError):
    def __init__(self, run_id: str, request_id: str | None = None, cause: BaseException | None = None):
        super().__init__(
            404,
            RUN_ERROR_CODES["RUN_NOT_FOUND"],
            RUN_ERROR_MESSAGES[RUN_ERROR_CODES["RUN_NOT_FOUND"]],
            request_id,
            cause,
        )
        self.run_id = run_id


def _normalize_by_code(err: BaseException) -> Dict[str, Any] | None:
    if getattr(err, "code", None) == RUN_ERROR_CODES["RUN_NOT_FOUND"]:
        return {
            "status": 404,
            "body": {
                "code": RUN_ERROR_CODES["RUN_NOT_FOUND"],
                "message": RUN_ERROR_MESSAGES[RUN_ERROR_CODES["RUN_NOT_FOUND"]],
            },
        }

    if getattr(err, "code", None) == RUN_ERROR_CODES["INVALID_RUN_TRANSITION"]:
        return {
            "status": 409,
            "body": {
                "code": RUN_ERROR_CODES["INVALID_RUN_TRANSITION"],
                "message": RUN_ERROR_MESSAGES[RUN_ERROR_CODES["INVALID_RUN_TRANSITION"]],
            },
        }

    if getattr(err, "name", None) == "InvalidRunTransitionError":
        return {
            "status": 409,
            "body": {
                "code": RUN_ERROR_CODES["INVALID_RUN_TRANSITION"],
                "message": RUN_ERROR_MESSAGES[RUN_ERROR_CODES["INVALID_RUN_TRANSITION"]],
            },
        }

    if getattr(err, "__class__", None).__name__ == "RunNotFoundError":
        return {
            "status": 404,
            "body": {
                "code": RUN_ERROR_CODES["RUN_NOT_FOUND"],
                "message": RUN_ERROR_MESSAGES[RUN_ERROR_CODES["RUN_NOT_FOUND"]],
            },
        }

    if isinstance(getattr(err, "message", None), str) and err.message.startswith("requested run was not found"):
        return {
            "status": 404,
            "body": {
                "code": RUN_ERROR_CODES["RUN_NOT_FOUND"],
                "message": RUN_ERROR_MESSAGES[RUN_ERROR_CODES["RUN_NOT_FOUND"]],
            },
        }

    return None


def normalize_run_error(err: BaseException, request_id: str | None = None) -> Dict[str, Any]:
    if isinstance(err, RunApiError):
        return {
            "status": err.status,
            "body": {
                "code": err.code,
                "message": str(err),
                "requestId": err.request_id,
            },
        }

    mapped = _normalize_by_code(err)
    if mapped is not None:
        normalized = mapped["body"]
        return {
            "status": mapped["status"],
            "body": {
                "code": normalized["code"],
                "message": normalized["message"],
                "requestId": request_id or str(uuid.uuid4()),
            },
        }

    return {
        "status": 500,
        "body": {
            "code": RUN_ERROR_CODES["INTERNAL_ERROR"],
            "message": RUN_ERROR_MESSAGES[RUN_ERROR_CODES["INTERNAL_ERROR"]],
            "requestId": request_id or str(uuid.uuid4()),
        },
    }
