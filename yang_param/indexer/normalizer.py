from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NodeRecord:
    node_id: str
    schema_path: str
    name: str
    module: str
    namespace: str
    prefix: str
    node_kind: str
    config: bool
    parent_path: Optional[str]
    parent_id: Optional[str]
    child_paths: list
    keys: list
    type_info: dict
    type_base: str
    default: Optional[str]
    mandatory: bool
    description: str
    when_expr: Optional[str]
    must_exprs: list


def _make_node_id(raw: dict) -> str:
    """Stable ID: <module>:<schema_path hash>"""
    module = raw.get("module", "")
    path = raw.get("schema_path", "")
    h = hashlib.sha1(path.encode()).hexdigest()[:8]
    return f"{module}:{h}"


def normalize(raw: dict, path_to_id: dict) -> NodeRecord:
    """Convert a raw parser dict to a NodeRecord, resolving parent_id via path_to_id."""
    node_id = _make_node_id(raw)
    parent_path = raw.get("parent_path")
    parent_id = path_to_id.get(parent_path) if parent_path else None

    return NodeRecord(
        node_id=node_id,
        schema_path=raw["schema_path"],
        name=raw["name"],
        module=raw["module"],
        namespace=raw.get("namespace", ""),
        prefix=raw.get("prefix", ""),
        node_kind=raw["node_kind"],
        config=raw["config"],
        parent_path=parent_path,
        parent_id=parent_id,
        child_paths=raw.get("child_paths", []),
        keys=raw.get("keys", []),
        type_info=raw.get("type_info", {}),
        type_base=raw.get("type_base", ""),
        default=raw.get("default"),
        mandatory=raw.get("mandatory", False),
        description=raw.get("description", ""),
        when_expr=raw.get("when_expr"),
        must_exprs=raw.get("must_exprs", []),
    )


def normalize_all(raw_nodes: list[dict]) -> list[NodeRecord]:
    """Normalize all raw nodes, building path→id map for parent linking."""
    # First pass: assign IDs
    path_to_id: dict[str, str] = {}
    for raw in raw_nodes:
        node_id = _make_node_id(raw)
        path_to_id[raw["schema_path"]] = node_id

    # Second pass: normalize with resolved parent_ids
    return [normalize(raw, path_to_id) for raw in raw_nodes]
