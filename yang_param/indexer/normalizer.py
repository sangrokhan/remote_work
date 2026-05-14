from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NodeRecord:
    node_id: str          # f"{module}:{sha256(schema_path)[:12]}"
    schema_path: str
    name: str
    module: str
    namespace: str
    prefix: str
    node_kind: str
    config: bool
    parent_id: str | None
    children_ids: list[str]
    keys: list[str]
    type_info: dict
    default: str | None
    mandatory: bool
    description: str
    when_expr: str | None
    must_exprs: list[str] = field(default_factory=list)


def _make_node_id(module: str, schema_path: str) -> str:
    digest = hashlib.sha256(schema_path.encode()).hexdigest()[:12]
    return f"{module}:{digest}"


def normalize(raw: dict, path_to_id: dict[str, str]) -> NodeRecord:
    node_id = _make_node_id(raw["module"], raw["schema_path"])
    path_to_id[raw["schema_path"]] = node_id
    parent_id = path_to_id.get(raw["parent_path"]) if raw.get("parent_path") else None
    children_ids = [
        path_to_id.get(p, _make_node_id(raw["module"], p))
        for p in raw.get("child_paths", [])
    ]
    return NodeRecord(
        node_id=node_id,
        schema_path=raw["schema_path"],
        name=raw["name"],
        module=raw["module"],
        namespace=raw.get("namespace", ""),
        prefix=raw.get("prefix", ""),
        node_kind=raw["node_kind"],
        config=raw["config"],
        parent_id=parent_id,
        children_ids=children_ids,
        keys=raw.get("keys", []),
        type_info=raw.get("type_info", {}),
        default=raw.get("default"),
        mandatory=raw.get("mandatory", False),
        description=raw.get("description", ""),
        when_expr=raw.get("when_expr"),
        must_exprs=raw.get("must_exprs", []),
    )


def normalize_all(raw_nodes: list[dict]) -> list[NodeRecord]:
    # Pass 1: build path→id map
    path_to_id: dict[str, str] = {}
    for raw in raw_nodes:
        path_to_id[raw["schema_path"]] = _make_node_id(raw["module"], raw["schema_path"])
    # Pass 2: normalize with known IDs
    return [normalize(raw, path_to_id) for raw in raw_nodes]
