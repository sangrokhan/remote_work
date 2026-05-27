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
    {"name": "list_modules", "description": "List all loaded YANG modules. Use to orient when the user names a module or you need to know what schemas are available.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "search_nodes", "description": "Search schema nodes by keyword across all kinds (container/list/leaf). Use to DISAMBIGUATE when a name is vague or find_leaf returns several candidates; returns node_id + schema_path for each hit so you can show the user full paths and let them pick. STEP 1 (disambiguation).", "inputSchema": {"type": "object", "properties": {"keyword": {"type": "string"}, "kind": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["keyword"]}},
    {"name": "find_leaf", "description": "Find a leaf/leaf-list node by exact-ish name; returns matching node(s) with node_id and schema_path. STEP 1: locate the TARGET node. If more than one candidate or the name is ambiguous, do NOT guess - confirm with the user (use search_nodes/get_ancestors to show paths). The returned node_id feeds get_required_keys, get_type_info, and the build_* tools.", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}, "parent_hint": {"type": "string"}}, "required": ["name"]}},
    {"name": "get_node", "description": "Get one node's metadata by node_id or schema_path. Use to confirm a node's identity/kind once you have an id or full path.", "inputSchema": {"type": "object", "properties": {"node_id_or_path": {"type": "string"}}, "required": ["node_id_or_path"]}},
    {"name": "get_children", "description": "Get direct children of a node. Use to drill down a container/list when narrowing toward the target leaf.", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
    {"name": "get_ancestors", "description": "Get ancestors root->node. Use to show the user a candidate's full schema path when disambiguating which node they meant.", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
    {"name": "get_path_to_leaf", "description": "Get the full schema path from root to the target node (no key values yet). Use to display the structural path before resolving keys.", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
    {"name": "get_required_keys", "description": "List every list key that must be supplied on the path to the target node, with each key's name/type/constraints. STEP 2: call right after the TARGET node_id is confirmed. If any key value is unknown, ASK THE USER for it before continuing. Output tells you what key_values resolve_instance_path and build_* need.", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
    {"name": "resolve_instance_path", "description": "Build the full keyed instance path, e.g. /pfx:root[k1='v1']/child[k2='v2']/leaf. STEP 3: call once you have all key_values from get_required_keys. Show the result to the user to confirm the keyed path before asking for a value or building XML.", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}, "key_values": {"type": "object"}}, "required": ["node_id", "key_values"]}},
    {"name": "get_type_info", "description": "Get the leaf's type, range/length/pattern constraints, and enum/default info. STEP 4 (writes only): call before asking the user for a value so you know the expected form. If the type is identityref, follow with resolve_identityref.", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
    {"name": "validate_value", "description": "Check a candidate value against the leaf's type. STEP 4: call after the user gives a value and before build_edit_config; if invalid, report why and ask again.", "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}, "value": {"type": "string"}}, "required": ["node_id", "value"]}},
    {"name": "resolve_identityref", "description": "List valid identityref values for a type. Use when get_type_info reports an identityref so you can offer/validate the allowed identities.", "inputSchema": {"type": "object", "properties": {"type_name": {"type": "string"}, "value": {"type": "string"}}, "required": ["type_name", "value"]}},
    {"name": "build_edit_config", "description": "Build an edit-config RPC XML to WRITE a value. STEP 5 (write): call only after TARGET confirmed, all key_values gathered (get_required_keys), and value type-checked (validate_value). operation merge/create/replace/delete; datastore running/candidate. Follow with validate_edit_config.", "inputSchema": {"type": "object", "properties": {"target_node_id": {"type": "string"}, "key_values": {"type": "object"}, "value": {"type": "string"}, "operation": {"type": "string"}, "datastore": {"type": "string"}}, "required": ["target_node_id", "key_values"]}},
    {"name": "build_get_config", "description": "Build a get-config RPC XML to READ config data at the target (with key predicates). STEP 5 (read): needs TARGET + key_values. No value/type steps required.", "inputSchema": {"type": "object", "properties": {"target_node_id": {"type": "string"}, "key_values": {"type": "object"}, "datastore": {"type": "string"}}, "required": ["target_node_id"]}},
    {"name": "build_get", "description": "Build a get RPC XML to READ config + operational state at the target. STEP 5 (read): needs TARGET + key_values. Use instead of build_get_config when operational state is wanted.", "inputSchema": {"type": "object", "properties": {"target_node_id": {"type": "string"}, "key_values": {"type": "object"}}, "required": ["target_node_id"]}},
    {"name": "build_delete_config", "description": "Build a delete-config RPC XML that wipes an entire datastore. Destructive - confirm with the user before emitting.", "inputSchema": {"type": "object", "properties": {"datastore": {"type": "string"}}, "required": ["datastore"]}},
    {"name": "validate_edit_config", "description": "Validate the structure of generated NETCONF XML. FINAL STEP: run on any build_* output before presenting it to the user.", "inputSchema": {"type": "object", "properties": {"xml": {"type": "string"}}, "required": ["xml"]}},
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
