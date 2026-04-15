"""FastAPI router for workflow schema API."""

from __future__ import annotations

import uuid
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .error_contract import ERROR_CODES, ERROR_MESSAGES, normalize_error
from .schema_registry import WorkflowSchemaRegistry

def _normalize_request_id(request: Request) -> str:
    return str(uuid.uuid4())


def create_workflow_schema_router(*, registry: WorkflowSchemaRegistry | None = None) -> APIRouter:
    registry = registry or WorkflowSchemaRegistry()
    router = APIRouter()

    @router.get("/api/workflows/{workflow_id}/schema")
    async def get_schema(workflow_id: str, request: Request):
        request_id = _normalize_request_id(request)
        try:
            workflow = await registry.get_by_id(workflow_id, {"requestId": request_id})
            return workflow
        except Exception as error:
            normalized = normalize_error(error, request_id)
            return JSONResponse(status_code=normalized["status"], content=normalized["body"])

    @router.api_route("/api/workflows/{workflow_id}/schema", methods=["POST", "PUT", "PATCH", "DELETE"])
    async def reject_non_get_schema():
        request_id = str(uuid.uuid4())
        return JSONResponse(
            status_code=405,
            content={
                "code": ERROR_CODES["INVALID_WORKFLOW_PAYLOAD"],
                "message": ERROR_MESSAGES[ERROR_CODES["INVALID_WORKFLOW_PAYLOAD"]],
                "requestId": request_id,
            },
            headers={"Allow": "GET"},
        )

    return router
