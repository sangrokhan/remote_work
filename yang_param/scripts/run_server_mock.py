"""MCP server launcher using mock store — runs without libyang for integration tests."""
from __future__ import annotations
import sys
import os
from unittest.mock import MagicMock

for _m in ["libyang", "indexer", "indexer.store", "indexer.parser", "indexer.normalizer"]:
    sys.modules.setdefault(_m, MagicMock())

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import tools
from mcp.server.stdio import stdio_server

IF_NS = "urn:ietf:params:xml:ns:yang:ietf-interfaces"


class _Node:
    def __init__(self, node_id, name, ns, kind, path, parent_id=None, keys=None):
        self.node_id = node_id
        self.name = name
        self.namespace = ns
        self.node_kind = kind
        self.schema_path = path
        self.module = "ietf-interfaces"
        self.prefix = "if"
        self.parent_id = parent_id
        self.keys = keys or []
        self.children_ids = []
        self.config = True
        self.description = ""
        self.type_info = {"base": "string"}
        self.mandatory = False
        self.default = None
        self.when_expr = None
        self.must_exprs = []


_INTERFACES = _Node("n1", "interfaces", IF_NS, "container", "/ietf-interfaces:interfaces")
_INTERFACE  = _Node("n2", "interface",  IF_NS, "list",      "/ietf-interfaces:interfaces/interface", parent_id="n1", keys=["name"])
_NAME       = _Node("n3", "name",       IF_NS, "leaf",      "/ietf-interfaces:interfaces/interface/name", parent_id="n2")
_MTU        = _Node("n4", "mtu",        IF_NS, "leaf",      "/ietf-interfaces:interfaces/interface/mtu",  parent_id="n2")

_BY_ID   = {n.node_id: n for n in [_INTERFACES, _INTERFACE, _NAME, _MTU]}
_BY_PATH = {n.schema_path: n for n in [_INTERFACES, _INTERFACE, _NAME, _MTU]}


class _MockStore:
    def get_by_id(self, node_id):
        return _BY_ID.get(node_id)

    def get_by_path(self, path):
        return _BY_PATH.get(path)


tools.init_store_from_instance(_MockStore())

from server.mcp_server import app  # noqa: E402


async def main():
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
