import json
import os
import tempfile
import asyncio

from python_backend.app.error_contract import ERROR_CODES, ERROR_MESSAGES
from python_backend.app.schema_registry import WorkflowSchemaRegistry


def _temp_manifest(payload):
    fd, path = tempfile.mkstemp(prefix="week1-registry-", suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp)
    return path


def test_duplicate_workflow_id_fails_load():
    manifest = [
        {
            "schemaVersion": "1.0.0",
            "workflowId": "support_ticket_classifier_v1",
            "version": "1.0.0",
            "nodes": [
                {"id": "intent_parser", "label": "의도 분류", "description": "테스트", "stateKey": "intent", "order": 1},
            ],
            "edges": [],
        },
        {
            "schemaVersion": "1.0.0",
            "workflowId": "support_ticket_classifier_v1",
            "version": "1.0.0",
            "nodes": [
                {"id": "knowledge_search", "label": "검색", "description": "테스트", "stateKey": "search", "order": 2},
            ],
            "edges": [],
        },
    ]
    path = _temp_manifest(manifest)
    try:
        registry = WorkflowSchemaRegistry(manifest_path=path)
        asyncio.run(registry.get_by_id("support_ticket_classifier_v1"))
        assert False, "expected WorkflowApiError"
    except Exception as error:
        assert getattr(error, "code", None) == ERROR_CODES["WORKFLOW_REGISTRY_ERROR"]
    finally:
        os.remove(path)


def test_duplicate_node_and_state_key_fails():
    manifest = {
        "schemaVersion": "1.0.0",
        "workflowId": "duplicate_test_workflow",
        "version": "1.0.0",
        "nodes": [
            {"id": "intent_parser", "label": "의도 분류", "description": "테스트", "stateKey": "intent", "order": 1},
            {"id": "intent_parser", "label": "의도 분류 2", "description": "테스트", "stateKey": "intent", "order": 2},
        ],
        "edges": [],
    }
    path = _temp_manifest(manifest)
    try:
        registry = WorkflowSchemaRegistry(manifest_path=path)
        try:
            asyncio.run(registry.get_by_id("duplicate_test_workflow"))
            assert False, "expected WorkflowApiError"
        except Exception as error:
            assert getattr(error, "code", None) == ERROR_CODES["WORKFLOW_REGISTRY_ERROR"]
    finally:
        os.remove(path)


def test_duplicate_edge_id_fails():
    manifest = {
        "schemaVersion": "1.0.0",
        "workflowId": "edge_dup_workflow",
        "version": "1.0.0",
        "nodes": [
            {"id": "intent_parser", "label": "의도 분류", "description": "테스트", "stateKey": "intent", "order": 1},
            {"id": "knowledge_search", "label": "검색", "description": "테스트", "stateKey": "search", "order": 2},
            {"id": "response_draft", "label": "응답", "description": "테스트", "stateKey": "draft", "order": 3},
        ],
        "edges": [
            {"id": "same-edge", "from": "intent_parser", "to": "knowledge_search", "label": "intent"},
            {"id": "same-edge", "from": "knowledge_search", "to": "response_draft", "label": "search"},
        ],
    }
    path = _temp_manifest(manifest)
    try:
        registry = WorkflowSchemaRegistry(manifest_path=path)
        try:
            asyncio.run(registry.get_by_id("edge_dup_workflow"))
            assert False, "expected WorkflowApiError"
        except Exception as error:
            assert getattr(error, "code", None) == ERROR_CODES["WORKFLOW_REGISTRY_ERROR"]
    finally:
        os.remove(path)


def test_edge_from_type_validation_fails():
    manifest = {
        "schemaVersion": "1.0.0",
        "workflowId": "edge_type_workflow",
        "version": "1.0.0",
        "nodes": [
            {"id": "intent_parser", "label": "의도 분류", "description": "테스트", "stateKey": "intent", "order": 1},
        ],
        "edges": [
            {"from": 1, "to": "intent_parser"},
        ],
    }
    path = _temp_manifest(manifest)
    try:
        registry = WorkflowSchemaRegistry(manifest_path=path)
        try:
            asyncio.run(registry.get_by_id("edge_type_workflow"))
            assert False, "expected WorkflowApiError"
        except Exception as error:
            assert getattr(error, "code", None) == ERROR_CODES["WORKFLOW_REGISTRY_ERROR"]
    finally:
        os.remove(path)


def test_negative_order_validation_fails():
    manifest = {
        "schemaVersion": "1.0.0",
        "workflowId": "negative_order_workflow",
        "version": "1.0.0",
        "nodes": [
            {"id": "intent_parser", "label": "의도 분류", "description": "테스트", "stateKey": "intent", "order": -1},
        ],
        "edges": [],
    }
    path = _temp_manifest(manifest)
    try:
        registry = WorkflowSchemaRegistry(manifest_path=path)
        try:
            asyncio.run(registry.get_by_id("negative_order_workflow"))
            assert False, "expected WorkflowApiError"
        except Exception as error:
            assert getattr(error, "code", None) == ERROR_CODES["WORKFLOW_REGISTRY_ERROR"]
    finally:
        os.remove(path)


def test_get_by_id_invalid_workflow_id():
    registry = WorkflowSchemaRegistry(
        manifest_path="docs/week-01/schema-contract.fixture.json"
    )
    result = asyncio.run(registry.get_by_id("support_ticket_classifier_v1"))
    assert result["workflowId"] == "support_ticket_classifier_v1"
