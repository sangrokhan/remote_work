"""Layer-1 regex router: extracts MO names, alarm codes, counter IDs, param names, spec numbers for fast routing."""
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

# Korean intent signals — must be checked before _SPEC_NUM_RE which fires on almost all queries
_COMPARATIVE_RE = re.compile(
    r"차이점은|차이는|무엇이 다른가|어떻게 다른가|비교|와\s+\S+의\s+차이|\bvs\b",
)
_PROCEDURAL_RE = re.compile(
    r"절차는|절차가|어떤\s+순서로|어떻게\s+수행|어떻게\s+진행|순서로\s+진행|단계로\s+수행|처리해야\s+하는가|수행되는가|진행되는가|어떻게\s+되는가",
)
_DIAGNOSTIC_RE = re.compile(
    r"원인은|원인이\s+무엇|원인과\s+확인|점검해야\s+할|점검할\s+|점검해야\s+하는가|안\s+될\s+때|안되는\s+이유|이유는\s+무엇|실패.*원인|확인할\s+조건|우선\s+확인|우선\s+점검|어떤\s+요소를|무엇을\s+점검|어떤\s+요인|미충족으로|어떤\s+보호\s+요구",
)
# Structured lookup via specific numeric/identifier value questions
_NUMERIC_LOOKUP_RE = re.compile(
    r"얼마인가|몇\s+개|기본값은|몇\s*가지인가|몇\s*초|몇\s*ms|식별자는\s+무엇|파라미터는\s+무엇|최대\s+크기는|최소\s+크기는|몇\s+개가\s+제시|어느\s+주파수|몇\s+가지",
)
# Definition/explanation intent — fires before spec_num to prevent spec_num false positives
_DEFINITION_RE = re.compile(
    r"[는은]\s+무엇인가|이란\s+무엇|란\s+무엇|무엇을\s+의미|[을를]\s+뜻하는가|[을를]\s+가리키는가|어떤\s+의미인가|어떤\s+개념인가|정의는\s+무엇",
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

        # Korean intent signals — before spec number check to avoid spec-number overriding intent
        if _COMPARATIVE_RE.search(query):
            return RouteResult(route=Route.COMPARATIVE, confidence=0.85, layer="regex", entities={})

        if _PROCEDURAL_RE.search(query):
            return RouteResult(route=Route.PROCEDURAL, confidence=0.85, layer="regex", entities={})

        if _DIAGNOSTIC_RE.search(query):
            return RouteResult(route=Route.DIAGNOSTIC, confidence=0.85, layer="regex", entities={})

        if _NUMERIC_LOOKUP_RE.search(query):
            return RouteResult(route=Route.STRUCTURED_LOOKUP, confidence=0.85, layer="regex", entities={})

        if _DEFINITION_RE.search(query):
            return RouteResult(route=Route.DEFINITION_EXPLAIN, confidence=0.80, layer="regex", entities={})

        m = _SPEC_NUM_RE.search(query)
        if m:
            return RouteResult(
                route=Route.DEFINITION_EXPLAIN,
                confidence=0.7,
                layer="regex",
                entities={"spec_number": f"{m.group(1)}.{m.group(2)}"},
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
