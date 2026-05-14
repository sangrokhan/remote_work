from __future__ import annotations
import re
from tools import get_store

_INT_BASES = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
_INT_NATIVE_RANGE = {
    "int8": (-128, 127), "int16": (-32768, 32767),
    "int32": (-2**31, 2**31-1), "int64": (-2**63, 2**63-1),
    "uint8": (0, 255), "uint16": (0, 65535),
    "uint32": (0, 4294967295), "uint64": (0, 2**64-1),
}


def get_type_info(node_id: str) -> dict:
    store = get_store()
    r = store.get_by_id(node_id)
    if not r:
        return {"type": None, "error": f"Node not found: {node_id}"}
    return {"type": r.type_info, "node_kind": r.node_kind}


def validate_value(node_id: str, value: str) -> dict:
    store = get_store()
    r = store.get_by_id(node_id)
    if not r:
        return {"valid": False, "error": f"Node not found: {node_id}"}

    t = r.type_info
    base = t.get("base", "string")

    if base in _INT_BASES:
        try:
            v = int(value)
        except ValueError:
            return {"valid": False, "error": f"Value '{value}' is not an integer"}
        ranges = t.get("range", [])
        if ranges:
            in_range = any(int(rng["min"]) <= v <= int(rng["max"]) for rng in ranges)
            if not in_range:
                desc = " or ".join(f"{rng['min']}..{rng['max']}" for rng in ranges)
                return {"valid": False, "error": f"Value out of range: must be {desc}"}
        else:
            lo, hi = _INT_NATIVE_RANGE.get(base, (-2**63, 2**63-1))
            if not (lo <= v <= hi):
                return {"valid": False, "error": f"Value out of native range for {base}"}

    elif base == "boolean":
        if value not in ("true", "false"):
            return {"valid": False, "error": "Boolean value must be 'true' or 'false'"}

    elif base == "string":
        patterns = t.get("pattern", [])
        for pat in patterns:
            if not re.fullmatch(pat, value):
                return {"valid": False, "error": f"Does not match pattern: {pat}"}

    elif base == "enumeration":
        enums = t.get("enum", [])
        if value not in enums:
            return {"valid": False, "error": f"Value must be one of: {enums}"}

    return {"valid": True}


def resolve_identityref(type_name: str, value: str) -> dict:
    store = get_store()
    candidates = [
        r for r in store.all_records()
        if r.type_info.get("base") == "identityref"
        and type_name.lower() in r.schema_path.lower()
    ]
    return {
        "type": type_name,
        "value": value,
        "candidates": [{"node_id": r.node_id, "schema_path": r.schema_path} for r in candidates],
    }
