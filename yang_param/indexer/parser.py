from __future__ import annotations
import os
import libyang
from typing import Iterator


def _extract_type_info(node) -> dict:
    try:
        t = node.type()
        base_int = t.base()
        base_name = libyang.Type.BASENAMES.get(base_int, str(base_int))
        info = {"base": base_name}
        try:
            ranges = list(t.all_ranges())
            if ranges:
                parsed = []
                for r in ranges:
                    if isinstance(r, str) and ".." in r:
                        lo, hi = r.split("..", 1)
                        parsed.append({"min": lo.strip(), "max": hi.strip()})
                    else:
                        parsed.append({"min": str(r[0]), "max": str(r[1])})
                info["range"] = parsed
        except Exception:
            pass
        try:
            patterns = list(t.all_patterns())
            if patterns:
                info["pattern"] = list(patterns)
        except Exception:
            pass
        try:
            enums = list(t.all_enums())
            if enums:
                info["enum"] = [e[0] for e in enums]
        except Exception:
            pass
        return info
    except Exception:
        return {"base": "unknown"}


def _iter_nodes(node, parent_path: str | None) -> Iterator[dict]:
    try:
        keyword = node.keyword()
    except Exception:
        return

    if keyword not in ("container", "list", "leaf", "leaf-list", "choice", "case"):
        return

    mod = node.module()
    path = node.schema_path()
    name = node.name()

    try:
        config = not node.config_false()
    except Exception:
        config = True

    keys = []
    if keyword == "list":
        try:
            keys = [k.name() if hasattr(k, "name") else str(k) for k in node.keys()]
        except Exception:
            pass

    type_info: dict = {}
    type_base = ""
    if keyword in ("leaf", "leaf-list"):
        type_info = _extract_type_info(node)
        type_base = type_info.get("base", "")

    try:
        description = node.description() or ""
    except Exception:
        description = ""

    try:
        mandatory = node.mandatory()
    except Exception:
        mandatory = False

    try:
        default_val = node.default()
        default = str(default_val) if default_val is not None else None
    except Exception:
        default = None

    child_paths = []
    try:
        for child in node.children():
            child_paths.append(child.schema_path())
    except Exception:
        pass

    try:
        mod_name = mod.name()
    except Exception:
        mod_name = ""

    try:
        prefix = mod.prefix()
    except Exception:
        prefix = ""

    # namespace: libyang 4.x Module has no .ns(), use module name as fallback
    namespace = mod_name

    yield {
        "schema_path": path,
        "name": name,
        "module": mod_name,
        "namespace": namespace,
        "prefix": prefix,
        "node_kind": keyword,
        "config": config,
        "parent_path": parent_path,
        "child_paths": child_paths,
        "keys": keys,
        "type_info": type_info,
        "type_base": type_base,
        "default": default,
        "mandatory": mandatory,
        "description": description,
        "when_expr": None,
        "must_exprs": [],
    }

    try:
        for child in node.children():
            yield from _iter_nodes(child, path)
    except Exception:
        pass


def parse_yang_dir(yang_dir: str, modules: list[str] | None = None) -> Iterator[dict]:
    ctx = libyang.Context(yang_dir)
    if modules:
        to_load = modules
    else:
        to_load = [
            f[:-5] for f in os.listdir(yang_dir)
            if f.endswith(".yang") and "@" not in f
        ]
    loaded = []
    for mod_name in to_load:
        try:
            loaded.append(ctx.load_module(mod_name))
        except Exception:
            pass
    for mod in loaded:
        for top_node in mod.children():
            yield from _iter_nodes(top_node, None)
