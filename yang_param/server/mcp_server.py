"""MCP stdio server exposing all 16 YANG schema tools."""
from __future__ import annotations
import asyncio
import json
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

import tools
from tools.explore import list_modules, search_nodes, find_leaf
from tools.tree import get_node, get_children, get_ancestors
from tools.keys import get_path_to_leaf, get_required_keys, resolve_instance_path
from tools.types import get_type_info, validate_value, resolve_identityref
from tools.builder import build_edit_config, build_get_config, build_get, build_delete_config, validate_edit_config

TOOLS = [
    {"name": "list_modules", "description": "List all loaded YANG modules", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "search_nodes", "description": "Search schema nodes by keyword", "inputSchema": {"type": "object", "properties": {"keyword": {"type": "string"}, "kind": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["keyword"]}},
    {"name": "find_leaf", "description": "Find a leaf/leaf-list node by name", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}, "parent_hint": {"type": "string"}}, "required": ["name"]}},
    {"name": "get_node", "description": "Get a node by node_id or schema_path", "inputSchema": {"type": "object", "properties": {"node_id_or_path": {"type": "string"}}, "required": ["node_id_or_path"]}},
    {"name": "get_children", "description": "Get children of a node", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
    {"name": "get_ancestors", "description": "Get ancestors from root to a node", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
    {"name": "get_path_to_leaf", "description": "Get full path from root to target node", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
    {"name": "get_required_keys", "description": "Get all list keys required on path to a node", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
    {"name": "resolve_instance_path", "description": "Build instance path with key predicates", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}, "key_values": {"type": "object"}}, "required": ["node_id", "key_values"]}},
    {"name": "get_type_info", "description": "Get type info for a leaf node", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
    {"name": "validate_value", "description": "Validate a string value against a leaf type", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}, "value": {"type": "string"}}, "required": ["node_id", "value"]}},
    {"name": "resolve_identityref", "description": "Resolve identityref candidates by type name", "inputSchema": {"type": "object", "properties": {"type_name": {"type": "string"}, "value": {"type": "string"}}, "required": ["type_name", "value"]}},
    {"name": "build_edit_config", "description": "Build a NETCONF edit-config RPC XML", "inputSchema": {"type": "object", "properties": {"target_node_id": {"type": "string"}, "key_values": {"type": "object"}, "value": {"type": "string"}, "operation": {"type": "string"}, "datastore": {"type": "string"}}, "required": ["target_node_id", "key_values"]}},
    {"name": "build_get_config", "description": "Build a NETCONF get-config RPC XML", "inputSchema": {"type": "object", "properties": {"target_node_id": {"type": "string"}, "key_values": {"type": "object"}, "datastore": {"type": "string"}}, "required": ["target_node_id"]}},
    {"name": "build_get", "description": "Build a NETCONF get RPC XML (config + operational state)", "inputSchema": {"type": "object", "properties": {"target_node_id": {"type": "string"}, "key_values": {"type": "object"}}, "required": ["target_node_id"]}},
    {"name": "build_delete_config", "description": "Build a NETCONF delete-config RPC XML", "inputSchema": {"type": "object", "properties": {"datastore": {"type": "string"}}, "required": ["datastore"]}},
    {"name": "validate_edit_config", "description": "Validate generated NETCONF XML structure", "inputSchema": {"type": "object", "properties": {"xml": {"type": "string"}}, "required": ["xml"]}},
]

_DISPATCH = {
    "list_modules": lambda a: list_modules(),
    "search_nodes": lambda a: search_nodes(**a),
    "find_leaf": lambda a: find_leaf(**a),
    "get_node": lambda a: get_node(a["node_id_or_path"]),
    "get_children": lambda a: get_children(a["node_id"]),
    "get_ancestors": lambda a: get_ancestors(a["node_id"]),
    "get_path_to_leaf": lambda a: get_path_to_leaf(a["node_id"]),
    "get_required_keys": lambda a: get_required_keys(a["node_id"]),
    "resolve_instance_path": lambda a: resolve_instance_path(a["node_id"], a["key_values"]),
    "get_type_info": lambda a: get_type_info(a["node_id"]),
    "validate_value": lambda a: validate_value(a["node_id"], a["value"]),
    "resolve_identityref": lambda a: resolve_identityref(a["type_name"], a["value"]),
    "build_edit_config": lambda a: build_edit_config(**a),
    "build_get_config": lambda a: build_get_config(**a),
    "build_get": lambda a: build_get(**a),
    "build_delete_config": lambda a: build_delete_config(a["datastore"]),
    "validate_edit_config": lambda a: validate_edit_config(a["xml"]),
}

app = Server("yang-schema-tool")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
        for t in TOOLS
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    handler = _DISPATCH.get(name)
    if not handler:
        return [types.TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
    result = handler(arguments)
    return [types.TextContent(type="text", text=json.dumps(result))]


async def _main():
    yang_dir = os.environ.get("YANG_DIR", "data/yang")
    db_path = os.environ.get("YANG_DB", "schema.db")
    tools.init_store(yang_dir, db_path)
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_main())
