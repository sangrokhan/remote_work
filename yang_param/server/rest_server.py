from __future__ import annotations
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from pydantic import BaseModel

import tools
from tools.explore import list_modules, search_nodes, find_leaf
from tools.tree import get_node, get_children, get_ancestors, get_root_nodes
from tools.keys import get_path_to_leaf, get_required_keys, resolve_instance_path
from tools.types import get_type_info, validate_value, resolve_identityref
from tools.builder import build_edit_config, build_get_config, build_get, build_delete_config, validate_edit_config

_MCP_SERVER = Path(__file__).parent / "mcp_server.py"

_SYSTEM_PROMPT = (
    "You are a YANG/NETCONF expert assistant. You have tools to explore YANG "
    "schemas and build NETCONF XML. The user talks to you over multiple turns, "
    "so resolve one piece of information at a time and confirm before moving on.\n"
    "\n"
    "A correct NETCONF operation needs three things, gathered IN THIS ORDER:\n"
    "  1. TARGET NODE  - the exact schema node (its node_id).\n"
    "  2. KEYS         - a key value for every list on the path to that node.\n"
    "  3. VALUE        - the value to write (edit-config only), type-validated.\n"
    "\n"
    "Standard procedure for a request like 'set leaf X to Y':\n"
    "  1. Find the TARGET node. If the user gave a full schema path, call "
    "get_node(path) to resolve its node_id. Otherwise call find_leaf(name=X). "
    "search_nodes takes a single keyword, never a full path. If the name is "
    "ambiguous or matches several nodes, use search_nodes / get_ancestors to show "
    "each candidate's full schema path and ASK THE USER which one they mean. Do "
    "not guess the node.\n"
    "  2. Once the node_id is confirmed, call get_required_keys(node_id) to learn "
    "which list keys must be supplied. If any key value is unknown, ASK THE USER "
    "for it (one prompt listing all missing keys is fine).\n"
    "  3. Call resolve_instance_path(node_id, key_values) to build the keyed "
    "instance path. Use its returned path; do not write a path yourself.\n"
    "  4. For a write: call get_type_info(node_id) (and resolve_identityref when "
    "the type is an identityref) so you know the expected value form, then ASK "
    "THE USER for the value and check it with validate_value before building.\n"
    "  5. Build the RPC: build_edit_config to write, build_get_config / build_get "
    "to read, build_delete_config to delete. Finish with validate_edit_config.\n"
    "\n"
    "Rules:\n"
    "  - NEVER write a node_id, schema path, instance path, key, or value from "
    "memory or by guessing. Every one of these MUST come verbatim from a tool "
    "result. If you do not have it from a tool, call the tool or ask the user.\n"
    "  - Resolve TARGET before KEYS, and KEYS before VALUE. Do not ask for a "
    "value until the keyed path is settled.\n"
    "  - Emit the final NETCONF XML exactly as returned by the build_* tool, in a "
    "code block. Do not edit or hand-write any part of it."
)

_VIEWER_HTML = Path(__file__).parent / "templates" / "viewer.html"

app = FastAPI(title="YANG Schema Tool", version="0.1.0")


@app.get("/")
def viewer():
    return FileResponse(_VIEWER_HTML, media_type="text/html")


@app.get("/tools/get_root_nodes")
def api_get_root_nodes(module: str) -> dict:
    return get_root_nodes(module)


# --- Explore ---

@app.get("/tools/list_modules")
def api_list_modules() -> dict:
    return list_modules()

@app.get("/tools/search_nodes")
def api_search_nodes(keyword: str, kind: str | None = None, top_k: int = 10) -> dict:
    return search_nodes(keyword, kind=kind, top_k=top_k)

@app.get("/tools/find_leaf")
def api_find_leaf(name: str, parent_hint: str | None = None) -> dict:
    return find_leaf(name, parent_hint=parent_hint)


# --- Tree ---

@app.get("/tools/get_node")
def api_get_node(node_id_or_path: str) -> dict:
    return get_node(node_id_or_path)

@app.get("/tools/get_children")
def api_get_children(node_id: str) -> dict:
    return get_children(node_id)

@app.get("/tools/get_ancestors")
def api_get_ancestors(node_id: str) -> dict:
    return get_ancestors(node_id)


# --- Keys ---

@app.get("/tools/get_path_to_leaf")
def api_get_path_to_leaf(node_id: str) -> dict:
    return get_path_to_leaf(node_id)

@app.get("/tools/get_required_keys")
def api_get_required_keys(node_id: str) -> dict:
    return get_required_keys(node_id)

class ResolveInstancePathRequest(BaseModel):
    node_id: str
    key_values: dict[str, str]

@app.post("/tools/resolve_instance_path")
def api_resolve_instance_path(req: ResolveInstancePathRequest) -> dict:
    return resolve_instance_path(req.node_id, req.key_values)


# --- Types ---

@app.get("/tools/get_type_info")
def api_get_type_info(node_id: str) -> dict:
    return get_type_info(node_id)

@app.get("/tools/validate_value")
def api_validate_value(node_id: str, value: str) -> dict:
    return validate_value(node_id, value)

@app.get("/tools/resolve_identityref")
def api_resolve_identityref(type_name: str, value: str) -> dict:
    return resolve_identityref(type_name, value)


# --- Builders ---

class EditConfigRequest(BaseModel):
    target_node_id: str
    key_values: dict[str, str]
    value: str | None = None
    operation: str = "merge"
    datastore: str = "running"

@app.post("/tools/build_edit_config")
def api_build_edit_config(req: EditConfigRequest) -> dict:
    return build_edit_config(
        req.target_node_id, req.key_values, req.value,
        operation=req.operation, datastore=req.datastore,
    )

class GetConfigRequest(BaseModel):
    target_node_id: str
    key_values: dict[str, str] | None = None
    datastore: str = "running"

@app.post("/tools/build_get_config")
def api_build_get_config(req: GetConfigRequest) -> dict:
    return build_get_config(req.target_node_id, key_values=req.key_values, datastore=req.datastore)

class GetRequest(BaseModel):
    target_node_id: str
    key_values: dict[str, str] | None = None

@app.post("/tools/build_get")
def api_build_get(req: GetRequest) -> dict:
    return build_get(req.target_node_id, key_values=req.key_values)

class DeleteConfigRequest(BaseModel):
    datastore: str

@app.post("/tools/build_delete_config")
def api_build_delete_config(req: DeleteConfigRequest) -> dict:
    return build_delete_config(req.datastore)

class ValidateRequest(BaseModel):
    xml: str

@app.post("/tools/validate_edit_config")
def api_validate_edit_config(req: ValidateRequest) -> dict:
    return validate_edit_config(req.xml)


class ChatRequest(BaseModel):
    message: str
    llm_url: str
    model: str = "gemma4-e4b-it"
    history: list[dict[str, Any]] = []

@app.post("/chat")
async def chat(req: ChatRequest):
    yang_dir = os.environ.get("YANG_DIR", "data/yang")
    db_path = os.environ.get("YANG_DB", "schema.db")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(_MCP_SERVER)],
        env={**os.environ, "YANG_DIR": yang_dir, "YANG_DB": db_path},
    )

    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_resp = await session.list_tools()

                openai_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.inputSchema,
                        },
                    }
                    for t in tools_resp.tools
                ]

                messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    *req.history,
                    {"role": "user", "content": req.message},
                ]

                llm = AsyncOpenAI(
                    base_url=req.llm_url,
                    api_key="none",
                    http_client=httpx.AsyncClient(verify=False, timeout=120),
                )
                for _ in range(10):
                    response = await llm.chat.completions.create(
                        model=req.model,
                        messages=messages,
                        tools=openai_tools,
                        temperature=0.1,
                    )
                    msg = response.choices[0].message
                    messages.append(msg)

                    if not msg.tool_calls:
                        return {"response": msg.content or "", "done": True}

                    for tc in msg.tool_calls:
                        tool_name = tc.function.name
                        tool_args = json.loads(tc.function.arguments)
                        result = await session.call_tool(tool_name, tool_args)
                        tool_text = result.content[0].text if result.content else "{}"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_text,
                        })

                return {"response": "Max tool iterations reached.", "done": True}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Chat error:\n%s", tb)
        if "wrong version number" in tb or "WRONG_VERSION_NUMBER" in tb:
            raise HTTPException(
                status_code=400,
                detail="SSL error: check URL scheme (http:// vs https://) and port.",
            )
        raise HTTPException(status_code=500, detail=tb)


# Startup: load store from env vars if not already loaded
@app.on_event("startup")
def startup():
    if tools._store is None:
        yang_dir = os.environ.get("YANG_DIR", "data/yang")
        db_path = os.environ.get("YANG_DB", "schema.db")
        tools.init_store(yang_dir, db_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.rest_server:app", host="0.0.0.0", port=8000, reload=False)
