from __future__ import annotations

from spar.router.schemas import Route, RouteResult

_ROUTE_DOC_TYPES: dict[Route, list[str]] = {
    Route.STRUCTURED_LOOKUP: ["parameter_ref", "counter_ref", "alarm_ref"],
    Route.DEFINITION_EXPLAIN: ["feature_desc", "spec"],
    Route.PROCEDURAL: ["mop", "install_guide"],
    Route.DIAGNOSTIC: ["alarm_ref", "feature_desc"],
    Route.COMPARATIVE: ["release_notes", "feature_desc"],
    Route.DEFAULT_RAG: ["feature_desc", "spec", "mop", "install_guide", "release_notes"],
}

# Priority order: alarm_code > param_name > mo_name
_ENTITY_PRIORITY: list[tuple[str, str]] = [
    ("alarm_code", "alarm_ref"),
    ("param_name", "parameter_ref"),
    ("mo_name", "parameter_ref"),
]


def doc_types_for_route(result: RouteResult) -> list[str]:
    if result.route == Route.STRUCTURED_LOOKUP and result.entities:
        for key, doc_type in _ENTITY_PRIORITY:
            if key in result.entities:
                return [doc_type]
    return _ROUTE_DOC_TYPES.get(result.route, ["feature_desc"])


def build_expr(result: RouteResult) -> str | None:
    clauses: list[str] = []

    if result.product and result.product != "both":
        clauses.append(f'product == "{result.product}"')
    if result.release:
        clauses.append(f'release == "{result.release}"')

    return " && ".join(clauses) if clauses else None
