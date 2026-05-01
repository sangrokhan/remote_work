"""Query context preparation: conversation history + relevant acronyms for LLM."""

from __future__ import annotations

from typing import Any

MAX_HISTORY_TURNS = 5


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
