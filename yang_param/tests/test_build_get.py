"""Unit tests for build_get — no libyang required."""
from __future__ import annotations
import sys
from unittest.mock import MagicMock

for _m in ["libyang", "indexer", "indexer.store", "indexer.parser", "indexer.normalizer"]:
    sys.modules.setdefault(_m, MagicMock())

import tools
from lxml import etree

NETCONF_NS = "urn:ietf:params:xml:ns:netconf:base:1.0"
NC = f"{{{NETCONF_NS}}}"
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

from tools.builder import build_get, validate_edit_config  # noqa: E402


def test_build_get_returns_xml():
    result = build_get("n4", {"name": "eth0"})
    assert result["xml"] is not None
    assert "error" not in result


def test_build_get_rpc_structure():
    result = build_get("n4", {"name": "eth0"})
    root = etree.fromstring(result["xml"].encode())
    assert root.tag == f"{NC}rpc"
    get_el = root.find(f"{NC}get")
    assert get_el is not None, "<get> element missing"
    filt = get_el.find(f"{NC}filter")
    assert filt is not None, "<filter> element missing"
    assert filt.get("type") == "subtree"


def test_build_get_no_source_element():
    result = build_get("n4", {"name": "eth0"})
    assert "<source>" not in result["xml"]


def test_build_get_key_value_in_filter():
    result = build_get("n4", {"name": "eth0"})
    root = etree.fromstring(result["xml"].encode())
    name_els = root.findall(f".//{{{IF_NS}}}name")
    assert name_els and name_els[0].text == "eth0"


def test_build_get_no_key_values():
    result = build_get("n4")
    assert result["xml"] is not None
    root = etree.fromstring(result["xml"].encode())
    assert root.find(f"{NC}get") is not None


def test_build_get_unknown_node():
    result = build_get("bad-id", {})
    assert result["xml"] is None
    assert "error" in result


def test_validate_accepts_get_xml():
    result = build_get("n4", {"name": "eth0"})
    v = validate_edit_config(result["xml"])
    assert v["valid"] is True
