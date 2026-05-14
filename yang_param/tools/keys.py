from __future__ import annotations
from tools import get_store
from tools.tree import get_ancestors, _record_to_dict


def get_path_to_leaf(node_id: str) -> dict:
    store = get_store()
    target = store.get_by_id(node_id)
    if not target:
        return {"path": [], "error": f"Node not found: {node_id}"}

    ancestors_result = get_ancestors(node_id)
    path = ancestors_result["ancestors"] + [_record_to_dict(target)]
    return {"path": path}


def get_required_keys(node_id: str) -> dict:
    store = get_store()
    target = store.get_by_id(node_id)
    if not target:
        return {"required_keys": [], "error": f"Node not found: {node_id}"}

    path_result = get_path_to_leaf(node_id)
    required_keys = []

    for node_dict in path_result["path"]:
        if node_dict["node_kind"] == "list":
            n = store.get_by_id(node_dict["node_id"])
            if n:
                for key_name in n.keys:
                    key_leaf = store.get_by_path(f"{n.schema_path}/{key_name}")
                    key_type = key_leaf.type_info if key_leaf else {}
                    required_keys.append({
                        "list_path": n.schema_path,
                        "key_name": key_name,
                        "type": key_type.get("base", "string"),
                        "constraints": {k: v for k, v in key_type.items() if k != "base"},
                    })

    return {"target_path": target.schema_path, "required_keys": required_keys}


def resolve_instance_path(node_id: str, key_values: dict[str, str]) -> dict:
    store = get_store()
    target = store.get_by_id(node_id)
    if not target:
        return {"instance_path": None, "missing_keys": [], "error": f"Node not found: {node_id}"}

    keys_result = get_required_keys(node_id)
    missing = [k["key_name"] for k in keys_result["required_keys"] if k["key_name"] not in key_values]
    if missing:
        return {"instance_path": None, "missing_keys": missing}

    path_result = get_path_to_leaf(node_id)
    segments = []
    for node_dict in path_result["path"]:
        n = store.get_by_id(node_dict["node_id"])
        if not n:
            continue
        if n.node_kind == "list":
            predicates = "".join(f"[{k}='{key_values[k]}']" for k in n.keys)
            segments.append(f"{n.name}{predicates}")
        else:
            segments.append(n.name)

    # Prefix first segment with module prefix
    root = path_result["path"][0]
    root_rec = store.get_by_id(root["node_id"])
    prefix = root_rec.prefix if root_rec else "if"
    segments[0] = f"{prefix}:{segments[0]}"
    instance_path = "/" + "/".join(segments)

    return {"instance_path": instance_path, "missing_keys": []}
