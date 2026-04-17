from __future__ import annotations

from typing import Any, Dict, List, Mapping


def _normalize_node_id(node: Any) -> str:
    if node is None:
        return ""
    if isinstance(node, str):
        return node

    for attr in ("value", "name", "id"):
        if hasattr(node, attr):
            candidate = getattr(node, attr)
            if isinstance(candidate, str):
                return candidate

    # Fallback for enums or simple wrappers
    if hasattr(node, "__name__"):
        return str(getattr(node, "__name__"))

    return str(node)


def _normalize_condition(condition: Any) -> str | None:
    if condition is None:
        return None
    if isinstance(condition, str):
        return condition
    if callable(condition):
        return getattr(condition, "__name__", str(condition))
    if hasattr(condition, "__name__"):
        try:
            return str(condition.__name__)
        except Exception:
            pass
    return str(condition)


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _extract_nodes_from_value(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, Mapping):
        candidates = list(value.keys())
    else:
        candidates = _ensure_list(value)

    return [_normalize_node_id(item) for item in candidates if _normalize_node_id(item)]


def _extract_edges_from_mapping(edge_map: Mapping[str, Any]) -> List[Dict[str, Any]]:
    parsed_edges: List[Dict[str, Any]] = []

    for source, target_value in edge_map.items():
        if isinstance(target_value, Mapping):
            for target, cond in target_value.items():
                parsed_edges.append(
                {
                        "from": _normalize_node_id(source),
                        "to": _normalize_node_id(target),
                        "condition": _normalize_condition(cond),
                    }
                )
            continue

        targets = _ensure_list(target_value)
        for target in targets:
            parsed_edges.append(
                {
                    "from": _normalize_node_id(source),
                    "to": _normalize_node_id(target),
                    "condition": None,
                }
            )
    return parsed_edges


def _extract_edge_object(edge: Any) -> List[Dict[str, Any]]:
    if isinstance(edge, Mapping):
        source = _normalize_node_id(edge.get("source"))
        target = _normalize_node_id(edge.get("target"))
        if not source and "from" in edge:
            source = _normalize_node_id(edge["from"])
        if not target and "to" in edge:
            target = _normalize_node_id(edge["to"])
        condition = (
            edge.get("condition")
            or edge.get("expression")
            or edge.get("guard")
            or edge.get("path")
            or edge.get("router")
            or edge.get("routing")
            or None
        )
        if source and target:
            return [
                {
                    "from": source,
                    "to": target,
                    "condition": _normalize_condition(condition),
                }
            ]
        return []

    if isinstance(edge, (tuple, list)) and len(edge) >= 2:
        source = _normalize_node_id(edge[0])
        target = _normalize_node_id(edge[1])
        condition = str(edge[2]) if len(edge) > 2 and edge[2] is not None else None
        if source and target:
            return [{"from": source, "to": target, "condition": condition}]
        return []

    source = _normalize_node_id(getattr(edge, "source", None))
    target = _normalize_node_id(getattr(edge, "target", None))
    if not source:
        source = _normalize_node_id(getattr(edge, "from_node", None))
    if not target:
        target = _normalize_node_id(getattr(edge, "to_node", None))
    if not source and hasattr(edge, "name"):
        source = _normalize_node_id(getattr(edge, "name"))
    condition = getattr(edge, "condition", None) or getattr(edge, "expression", None)
    if source and target:
        return [
            {
                "from": source,
                "to": target,
                "condition": _normalize_condition(condition),
            }
        ]
    return []


def _extract_edge_values(graph: Any) -> List[Any]:
    for path in (
        "edges",
        "graph_edges",
        "conditional_edges",
        "conditional_edges_dict",
        "branch_edges",
        "branches",
        "routes",
        "__edges__",
        "_conditional_edges",
        "_branch_edges",
    ):
        if hasattr(graph, path):
            return _ensure_list(getattr(graph, path))

    if hasattr(graph, "__dict__"):
        for key in (
            "edges",
            "graph_edges",
            "conditional_edges",
            "conditional_edges_dict",
            "branch_edges",
            "branches",
            "routes",
            "__edges__",
            "_conditional_edges",
            "_branch_edges",
        ):
            if key in graph.__dict__:
                return _ensure_list(graph.__dict__[key])
    return []


def _extract_node_values(graph: Any) -> List[Any]:
    for path in (
        "nodes",
        "state_nodes",
        "node_map",
        "__nodes__",
        "node_names",
    ):
        if hasattr(graph, path):
            value = getattr(graph, path)
            if isinstance(value, Mapping):
                return list(value.keys())
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, list):
                return value
            return _ensure_list(value)

    if hasattr(graph, "__dict__"):
        for key in (
            "nodes",
            "state_nodes",
            "node_map",
            "__nodes__",
            "node_names",
        ):
            if key in graph.__dict__:
                value = graph.__dict__[key]
                if isinstance(value, Mapping):
                    return list(value.keys())
                if isinstance(value, tuple):
                    return list(value)
                if isinstance(value, list):
                    return value
                return _ensure_list(value)
    return []


def serialize_stategraph_to_json(graph: Any) -> Dict[str, Any]:
    """Serialize a LangGraph state graph object into a generic node/edge JSON."""

    graph_obj = graph
    if hasattr(graph_obj, "get_graph") and callable(graph_obj.get_graph):
        try:
            graph_obj = graph_obj.get_graph()
        except Exception:
            graph_obj = graph

    node_values = _extract_node_values(graph_obj)
    edge_values = _extract_edge_values(graph_obj)

    node_ids = list(dict.fromkeys(_extract_nodes_from_value(node_values)))
    edges: List[Dict[str, Any]] = []

    if isinstance(edge_values, Mapping):
        edges.extend(_extract_edges_from_mapping(edge_values))
    else:
        for edge in _ensure_list(edge_values):
            edges.extend(_extract_edge_object(edge))

    if not edges and hasattr(graph_obj, "get_graph") and hasattr(graph_obj, "get_reachable_edges"):
        try:
            for source, targets in graph_obj.get_reachable_edges().items():
                edges.extend(
                    {
                        "from": _normalize_node_id(source),
                        "to": _normalize_node_id(target),
                        "condition": None,
                    }
                    for target in _ensure_list(targets)
                    if _normalize_node_id(source) and _normalize_node_id(target)
                )
        except Exception:
            pass

    if not node_ids:
        for edge in edges:
            node_ids.append(_normalize_node_id(edge.get("from")))
            node_ids.append(_normalize_node_id(edge.get("to")))
        node_ids = [node for node in dict.fromkeys(node_ids) if node]

    minimal_edges = []
    for edge in edges:
        source = edge.get("from")
        target = edge.get("to")
        if not source or not target:
            continue
        edge_payload = {"from": source, "to": target}
        if "condition" in edge and edge.get("condition") is not None:
            edge_payload["condition"] = edge.get("condition")
        minimal_edges.append(edge_payload)

    for edge in minimal_edges:
        source = edge.get("from")
        target = edge.get("to")
        if source and source not in node_ids:
            node_ids.append(source)
        if target and target not in node_ids:
            node_ids.append(target)

    return {
        "nodes": [{"id": node_id} for node_id in node_ids],
        "edges": minimal_edges,
    }
