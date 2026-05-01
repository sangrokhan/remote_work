"""Query context preparation: conversation history + relevant acronyms for LLM."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from spar.llm import LLMRole, get_client
from spar.prompts import load_prompt

_log = logging.getLogger(__name__)
_REWRITE_SYSTEM_PROMPT = load_prompt("query_rewrite_system.txt")

MAX_HISTORY_TURNS = 5


@dataclass
class QueryRewriteResult:
    original: str
    rewritten: str
    complexity: Literal["simple", "complex"]
    rationale: str


def format_history(history: list[dict[str, str]], max_turns: int = MAX_HISTORY_TURNS) -> str:
    if not history:
        return ""
    # each turn = user + assistant → 2 messages; keep last max_turns turns
    recent = history[-(max_turns * 2):]
    lines = []
    for msg in recent:
        prefix = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{prefix}: {msg.get('content', '')}")
    return "\n".join(lines)


def extract_relevant_acronyms(
    query: str,
    history: list[dict[str, str]],
    acronyms: dict[str, Any],
) -> dict[str, str]:
    all_text = query + " " + " ".join(m.get("content", "") for m in history)
    all_text_upper = all_text.upper()

    result: dict[str, str] = {}
    for acronym, data in acronyms.get("global", {}).items():
        expansion = data.get("expansion", "")
        if acronym.upper() in all_text_upper:
            result[acronym] = expansion
        for variant in data.get("variants", []):
            if variant and variant.upper() in all_text_upper:
                result[variant] = expansion
    return result


def build_context(
    query: str,
    history: list[dict[str, str]],
    acronyms: dict[str, Any],
    max_turns: int = MAX_HISTORY_TURNS,
) -> str:
    """Return formatted context string (history + acronyms) to inject into LLM prompt."""
    parts = []

    history_str = format_history(history, max_turns)
    if history_str:
        parts.append(f"Conversation history:\n{history_str}")

    relevant = extract_relevant_acronyms(query, history, acronyms)
    if relevant:
        acr_lines = "\n".join(f"- {k}: {v}" for k, v in relevant.items())
        parts.append(f"Relevant acronyms:\n{acr_lines}")

    return "\n\n".join(parts)


async def rewrite_query(
    query: str,
    history: list[dict[str, str]],
    acronyms: dict[str, Any],
    max_turns: int = MAX_HISTORY_TURNS,
) -> QueryRewriteResult:
    history_str = format_history(history, max_turns)
    user_content = f"Conversation history:\n{history_str}\n\nQuery: {query}" if history_str else f"Query: {query}"
    try:
        client = await get_client(LLMRole.ROUTER)
        raw = await client.chat(
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=256,
        )
        parsed = json.loads(raw)
        return QueryRewriteResult(
            original=query,
            rewritten=parsed.get("rewritten", query),
            complexity=parsed.get("complexity", "simple"),
            rationale=parsed.get("rationale", ""),
        )
    except Exception as exc:
        _log.warning("rewrite_query fallback — %s: %s", type(exc).__name__, exc)
        return QueryRewriteResult(original=query, rewritten=query, complexity="simple", rationale="fallback")
