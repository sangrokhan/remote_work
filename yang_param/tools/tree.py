from __future__ import annotations
from tools import get_store
from indexer.normalizer import NodeRecord


def _record_to_dict(r: NodeRecord) -> dict:
    return {
        "node_id": r.node_id,
        "name": r.name,
        "schema_path": r.schema_path,
        "node_kind": r.node_kind,
        "module": r.module,
        "namespace": r.namespace,
        "prefix": r.prefix,
        "config": r.config,
        "keys": r.keys,
        "description": r.description,
    }


def get_node(node_id_or_path: str) -> dict:
    store = get_store()
    r = store.get_by_id(node_id_or_path) or store.get_by_path(node_id_or_path)
    if not r:
        return {"node": None, "error": f"Node not found: {node_id_or_path}"}
    return {"node": _record_to_dict(r)}


def get_children(node_id: str) -> dict:
    store = get_store()
    parent = store.get_by_id(node_id)
    if not parent:
        return {"children": [], "error": f"Node not found: {node_id}"}

    children = []
    for cid in parent.children_ids:
        child = store.get_by_id(cid)
        if child:
            children.append(_record_to_dict(child))

    # Fallback: search by parent_id
    if not children:
        children = [
            _record_to_dict(r) for r in store.all_records()
            if r.parent_id == node_id
        ]

    return {"children": children}


def get_ancestors(node_id: str) -> dict:
    store = get_store()
    ancestors = []
    current = store.get_by_id(node_id)
    if not current:
        return {"ancestors": [], "error": f"Node not found: {node_id}"}

    while current and current.parent_id:
        parent = store.get_by_id(current.parent_id)
        if not parent:
            break
        ancestors.append(_record_to_dict(parent))
        current = parent

    ancestors.reverse()  # root-first
    return {"ancestors": ancestors}
