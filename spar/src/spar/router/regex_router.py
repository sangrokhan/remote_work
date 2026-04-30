from __future__ import annotations

import re

from spar.router.schemas import Route, RouteResult

_ALARM_CODE_RE = re.compile(r"\bALM-(\d+)\b", re.IGNORECASE)
_ALARM_WORD_RE = re.compile(r"\balarm\s+(\d+)\b", re.IGNORECASE)
_MO_NAME_RE = re.compile(r"\b([A-Z][A-Za-z]{4,}(?:DU|FDD|TDD|Cell|Ran|NR|LTE|SA|NSA))\b")
_PARAM_NAME_RE = re.compile(
    r"\b([a-z][a-zA-Z]{3,}(?:Power|Timer|Threshold|Max|Min|Offset|Hysteresis|Period|Level|Limit))\b"
)


class RegexRouter:
    """Layer 1: regex fast-path. Returns None on no match."""

    def route(self, query: str) -> RouteResult | None:
        m = _ALARM_CODE_RE.search(query)
        if m:
            return RouteResult(
                route=Route.STRUCTURED_LOOKUP,
                confidence=1.0,
                layer="regex",
                entities={"alarm_code": m.group(0).upper()},
            )

        m = _ALARM_WORD_RE.search(query)
        if m:
            return RouteResult(
                route=Route.STRUCTURED_LOOKUP,
                confidence=1.0,
                layer="regex",
                entities={"alarm_code": f"ALM-{m.group(1)}"},
            )

        m = _MO_NAME_RE.search(query)
        if m:
            return RouteResult(
                route=Route.STRUCTURED_LOOKUP,
                confidence=1.0,
                layer="regex",
                entities={"mo_name": m.group(1)},
            )

        m = _PARAM_NAME_RE.search(query)
        if m:
            return RouteResult(
                route=Route.STRUCTURED_LOOKUP,
                confidence=0.9,
                layer="regex",
                entities={"param_name": m.group(1)},
            )

        return None
