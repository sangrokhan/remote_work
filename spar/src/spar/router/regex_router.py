from __future__ import annotations

import re

from spar.router.schemas import Route, RouteResult

_ALARM_CODE_RE = re.compile(r"\bALM-(\d+)\b", re.IGNORECASE)
_ALARM_WORD_RE = re.compile(r"\balarm\s+(\d+)\b", re.IGNORECASE)
_MO_NAME_RE = re.compile(r"\b([A-Z][A-Za-z]{4,}(?:DU|FDD|TDD|Cell|Ran|NR|LTE|SA|NSA))\b")
_PARAM_NAME_RE = re.compile(
    r"\b([a-z][a-zA-Z]{3,}(?:Power|Timer|Threshold|Max|Min|Offset|Hysteresis|Period|Level|Limit))\b"
)
# Counter group ID: G-0042, C-123
_COUNTER_GROUP_ID_RE = re.compile(r"\b([A-Z]-\d{3,5})\b")
# Counter name: dot-separated uppercase hierarchy (e.g. CELL.UE.MaxConnectedNbr)
_COUNTER_NAME_RE = re.compile(r"\b([A-Z]{2,8}(?:\.[A-Z][A-Za-z0-9]{1,}){2,})\b")
_SPEC_NUM_RE = re.compile(
    r"\b(?:3GPP\s+)?TS\s*(\d{2})[\.\s]?(\d{3})\b", re.IGNORECASE
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

        m = _SPEC_NUM_RE.search(query)
        if m:
            return RouteResult(
                route=Route.DEFINITION_EXPLAIN,
                confidence=1.0,
                layer="regex",
                entities={"spec_number": f"{m.group(1)}.{m.group(2)}"},
            )

        m = _COUNTER_GROUP_ID_RE.search(query)
        if m:
            return RouteResult(
                route=Route.STRUCTURED_LOOKUP,
                confidence=1.0,
                layer="regex",
                entities={"mid_group_id": m.group(1)},
            )

        m = _COUNTER_NAME_RE.search(query)
        if m:
            return RouteResult(
                route=Route.STRUCTURED_LOOKUP,
                confidence=0.95,
                layer="regex",
                entities={"counter_name": m.group(1)},
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
