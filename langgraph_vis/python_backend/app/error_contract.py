"""Schema API error contract and helpers."""

from __future__ import annotations

import uuid
from typing import Any, Dict

ERROR_CODES = {
    "WORKFLOW_NOT_FOUND": "WORKFLOW_NOT_FOUND",
    "INVALID_WORKFLOW_ID": "INVALID_WORKFLOW_ID",
    "INVALID_WORKFLOW_PAYLOAD": "INVALID_WORKFLOW_PAYLOAD",
    "WORKFLOW_REGISTRY_ERROR": "WORKFLOW_REGISTRY_ERROR",
    "INTERNAL_ERROR": "INTERNAL_ERROR",
}

ERROR_MESSAGES = {
    ERROR_CODES["WORKFLOW_NOT_FOUND"]: "requested workflow was not found",
    ERROR_CODES["INVALID_WORKFLOW_ID"]: "invalid workflow id",
    ERROR_CODES["INVALID_WORKFLOW_PAYLOAD"]: "invalid workflow schema payload",
    ERROR_CODES["WORKFLOW_REGISTRY_ERROR"]: "workflow registry processing error",
    ERROR_CODES["INTERNAL_ERROR"]: "internal server error",
}


class WorkflowApiError(Exception):
    def __init__(self, status: int, code: str, message: str, request_id: str | None = None, cause: BaseException | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.request_id = request_id or str(uuid.uuid4())
        self.cause = cause


def normalize_error(err: BaseException, request_id: str | None = None) -> Dict[str, Any]:
    if isinstance(err, WorkflowApiError):
        return {
            "status": err.status,
            "body": {
                "code": err.code,
                "message": str(err),
                "requestId": err.request_id,
            },
        }

    return {
        "status": 500,
        "body": {
            "code": ERROR_CODES["INTERNAL_ERROR"],
            "message": ERROR_MESSAGES[ERROR_CODES["INTERNAL_ERROR"]],
            "requestId": request_id or str(uuid.uuid4()),
        },
    }


def to_error_response_body(code: str, message: str, request_id: str | None = None) -> Dict[str, str]:
    return {
        "code": code,
        "message": message,
        "requestId": request_id or str(uuid.uuid4()),
    }
