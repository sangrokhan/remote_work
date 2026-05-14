from __future__ import annotations
from tools import get_store


def list_modules() -> dict:
    return {"modules": get_store().list_modules()}


def search_nodes(keyword: str, kind: str | None = None, top_k: int = 10) -> dict:
    records = get_store().search_by_keyword(keyword, kind=kind, top_k=top_k)
    return {
        "nodes": [
            {
                "node_id": r.node_id,
                "name": r.name,
                "schema_path": r.schema_path,
                "node_kind": r.node_kind,
                "module": r.module,
                "description": r.description,
            }
            for r in records
        ]
    }


def find_leaf(name: str, parent_hint: str | None = None) -> dict:
    store = get_store()
    candidates = store.search_by_name(name)
    candidates = [r for r in candidates if r.node_kind in ("leaf", "leaf-list")]

    if parent_hint:
        filtered = [r for r in candidates if parent_hint.lower() in r.schema_path.lower()]
        if filtered:
            candidates = filtered

    if not candidates:
        return {"node": None, "error": f"No leaf named '{name}' found"}

    r = candidates[0]
    return {
        "node": {
            "node_id": r.node_id,
            "name": r.name,
            "schema_path": r.schema_path,
            "node_kind": r.node_kind,
            "module": r.module,
            "type_info": r.type_info,
            "description": r.description,
            "candidates": len(candidates),
        }
    }
