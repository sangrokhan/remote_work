"""Maps RouteResult to target doc_types and builds Milvus filter expressions (product/release/keyword clauses)."""
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


def build_expr(
    result: RouteResult,
    matched_terms: list[str] | None = None,
) -> str | None:
    clauses: list[str] = []

    if result.product and result.product != "both":
        clauses.append(f'product == "{result.product}"')
    if result.release:
        clauses.append(f'release == "{result.release}"')

    if matched_terms:
        term_clauses = [f'array_contains(keywords, "{t}")' for t in matched_terms]
        clauses.append("(" + " || ".join(term_clauses) + ")")

    return " && ".join(clauses) if clauses else None


def resolve_alarm_entity(entities: dict) -> dict | None:
    """Resolve an extracted ``alarm_code`` entity against AlarmIndex.

    Returns a dict with keys ``alarm_id``, ``alarm_name``, ``severity``,
    ``category``, ``module``, ``pdf_ref``, ``keywords`` if a match is
    found; otherwise ``None``.

    Structured-lookup shortcut used before vector search when the regex
    router has identified an exact alarm code.
    """
    code = entities.get("alarm_code") if entities else None
    if not code:
        return None

    from spar.retrieval.alarm_index import get_alarm_index

    rec = get_alarm_index().lookup(code)
    if rec is None:
        return None

    payload = rec.to_dict()
    payload["keywords"] = rec.to_keywords()
    return payload
