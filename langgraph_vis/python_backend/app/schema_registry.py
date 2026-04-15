"""Workflow schema registry for Python backend."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

from .error_contract import ERROR_CODES, ERROR_MESSAGES, WorkflowApiError

IDENTIFIERS = {
    "workflowId": re.compile(r"^[a-z][a-z0-9_-]{1,63}$"),
    "nodeId": re.compile(r"^[a-z][a-z0-9_-]{1,63}$"),
    "stateKey": re.compile(r"^[a-z][a-zA-Z0-9_-]{0,127}$"),
    "version": re.compile(r"^\d+\.\d+\.\d+$"),
}


def _default_manifest_path() -> str:
    return str(
        Path(os.getcwd()) / "docs" / "week-01" / "schema-contract.fixture.json"
    )


class WorkflowSchemaRegistry:
    def __init__(self, *, manifest_path: str | None = None):
        self.manifest_path = manifest_path or _default_manifest_path()
        self.cache = {}
        self.loaded = False
        self._loading = None
        self._loading_lock = asyncio.Lock()

    async def load(self) -> None:
        if self.loaded:
            return

        async with self._loading_lock:
            if self.loaded:
                return
            if self._loading is None:
                self._loading = asyncio.create_task(self._load_impl())
        await self._loading

    async def _load_impl(self) -> None:
        try:
            raw = Path(self.manifest_path).read_text(encoding="utf-8")
            parsed = json.loads(raw)
            workflows = parsed if isinstance(parsed, list) else [parsed]

            for workflow in workflows:
                if not isinstance(workflow, dict) or not isinstance(workflow.get("workflowId"), str) or not IDENTIFIERS["workflowId"].match(workflow["workflowId"]):
                    raise ValueError(f"invalid workflowId: {workflow.get('workflowId') if isinstance(workflow, dict) else workflow}")
                workflow_id = workflow["workflowId"]
                if workflow_id in self.cache:
                    raise ValueError(f"duplicate workflowId: {workflow_id}")
                self.validate_workflow_payload(workflow)
                self.cache[workflow_id] = workflow

            self.loaded = True
        except Exception as error:
            self.loaded = False
            self.cache = {}
            raise WorkflowApiError(
                500,
                ERROR_CODES["WORKFLOW_REGISTRY_ERROR"],
                ERROR_MESSAGES[ERROR_CODES["WORKFLOW_REGISTRY_ERROR"]],
                None,
                error,
            )
        finally:
            self._loading = None

    def validate_workflow_payload(self, schema: dict) -> None:
        if not isinstance(schema, dict):
            raise ValueError("schema payload is not an object")
        if not isinstance(schema.get("schemaVersion"), str) or not schema.get("schemaVersion"):
            raise ValueError("schemaVersion required")
        if not IDENTIFIERS["workflowId"].match(schema.get("workflowId") or ""):
            raise ValueError(f"invalid workflowId: {schema.get('workflowId')}")
        if not IDENTIFIERS["version"].match(schema.get("version") or ""):
            raise ValueError(f"invalid version: {schema.get('version')}")
        if not isinstance(schema.get("nodes"), list) or len(schema.get("nodes")) < 1:
            raise ValueError("nodes must be non-empty array")
        if not isinstance(schema.get("edges"), list):
            raise ValueError("edges must be array")

        node_ids = set()
        state_keys = set()
        edge_ids = set()
        node_errors = []

        for node in schema.get("nodes", []):
            if not isinstance(node, dict):
                raise ValueError("node must be object")
            node_id = node.get("id")
            state_key = node.get("stateKey")

            if not isinstance(node_id, str) or not IDENTIFIERS["nodeId"].match(node_id):
                node_errors.append(f"invalid node id: {node_id}")
            elif node_id in node_ids:
                node_errors.append(f"duplicate node id: {node_id}")

            if not isinstance(state_key, str) or not IDENTIFIERS["stateKey"].match(state_key):
                node_errors.append(f"invalid stateKey: {state_key}")
            elif state_key in state_keys:
                node_errors.append(f"duplicate stateKey: {state_key}")

            if node_id and node_id not in node_ids:
                node_ids.add(node_id)
            if state_key and state_key not in state_keys:
                state_keys.add(state_key)

            if not isinstance(node.get("label"), str):
                raise ValueError(f"label required for node {node_id}")
            if not isinstance(node.get("description"), str):
                raise ValueError(f"description required for node {node_id}")
            order = node.get("order")
            if not isinstance(order, int) or order < 0:
                raise ValueError(f"order required for node {node_id}")

        if node_errors:
            raise ValueError(", ".join(node_errors))

        for edge in schema.get("edges", []):
            if not isinstance(edge, dict):
                raise ValueError("edge must be object")
            edge_id = edge.get("id")
            from_node = edge.get("from")
            to_node = edge.get("to")
            if isinstance(edge_id, str):
                if edge_id in edge_ids:
                    raise ValueError(f"duplicate edge id: {edge_id}")
                edge_ids.add(edge_id)
            if from_node not in node_ids:
                raise ValueError(f"edge.from references missing node: {from_node}")
            if to_node not in node_ids:
                raise ValueError(f"edge.to references missing node: {to_node}")
            if edge_id and (not isinstance(edge_id, str) or not IDENTIFIERS["nodeId"].match(edge_id)):
                raise ValueError(f"edge.id must match pattern: {edge_id}")
            if edge.get("label") is not None and not isinstance(edge.get("label"), str):
                raise ValueError(f"edge.label must be string: {from_node} -> {to_node}")
            if edge.get("metadata") is not None and not isinstance(edge.get("metadata"), dict):
                raise ValueError(f"edge.metadata must be object: {from_node} -> {to_node}")

    async def get_by_id(self, workflow_id: str, request_context: dict | None = None):
        await self.load()

        request_id = request_context.get("requestId") if request_context else None
        if not IDENTIFIERS["workflowId"].match(workflow_id):
            raise WorkflowApiError(
                400,
                ERROR_CODES["INVALID_WORKFLOW_ID"],
                f"{ERROR_MESSAGES[ERROR_CODES['INVALID_WORKFLOW_ID']]}: {workflow_id}",
                request_id,
            )

        schema = self.cache.get(workflow_id)
        if schema is None:
            raise WorkflowApiError(
                404,
                ERROR_CODES["WORKFLOW_NOT_FOUND"],
                f"{ERROR_MESSAGES[ERROR_CODES['WORKFLOW_NOT_FOUND']]}: {workflow_id}",
                request_id,
            )
        return schema
