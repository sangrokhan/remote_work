from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI


def load_acronyms(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Acronym dictionary not found: {path}") from None


def build_reverse_index(acronyms: dict) -> dict[str, str]:
    """전체어·variants → abbreviation 역방향 인덱스 빌드."""
    reverse: dict[str, str] = {}
    for abbrev, info in acronyms.get("global", {}).items():
        expansion: str = info["expansion"]
        for form in [expansion, expansion.lower(), expansion.replace("-", "").lower()]:
            reverse[form] = abbrev
        for variant in info.get("variants", []):
            for form in [variant, variant.lower(), variant.replace("-", "").lower()]:
                reverse[form] = abbrev
    return reverse


def _apply_global(text: str, global_dict: dict) -> str:
    for abbrev, info in global_dict.items():
        expansion: str = info["expansion"]
        text = re.sub(rf"\b{re.escape(abbrev)}\b(?!\()", f"{abbrev}({expansion})", text)
        for variant in info.get("variants", []):
            text = re.sub(rf"\b{re.escape(variant)}\b(?!\()", f"{variant}({expansion})", text)
    return text


def _apply_conflicts_no_llm(text: str, conflicts: dict) -> str:
    for abbrev, info in conflicts.items():
        candidates: list[str] = info["candidates"]
        expansion = "|".join(candidates)
        text = re.sub(rf"\b{re.escape(abbrev)}\b(?!\()", f"{abbrev}({expansion})", text)
        for variant in info.get("variants", []):
            text = re.sub(rf"\b{re.escape(variant)}\b(?!\()", f"{variant}({expansion})", text)
    return text


def map_abbreviations(
    text: str,
    acronyms: dict,
    llm_client: OpenAI | None = None,
    model: str = "google/gemma-4-E4B-it",
) -> str:
    """인제스트·쿼리 공용 약어 매핑 진입점."""
    text = _apply_global(text, acronyms.get("global", {}))
    conflicts = acronyms.get("conflicts", {})
    if not conflicts:
        return text
    if llm_client is None:
        return _apply_conflicts_no_llm(text, conflicts)
    return _resolve_and_apply_conflicts(text, conflicts, llm_client, model)


def _resolve_and_apply_conflicts(
    text: str,
    conflicts: dict,
    client: OpenAI,
    model: str,
) -> str:
    contexts = _collect_conflict_contexts(text, conflicts)
    if not contexts:
        return text
    resolutions = _llm_batch_classify(contexts, conflicts, client, model)
    return _apply_conflict_resolutions(text, conflicts, resolutions)


def _collect_conflict_contexts(text: str, conflicts: dict) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for abbrev in conflicts:
        for m in re.finditer(rf"\b{re.escape(abbrev)}\b", text):
            start = max(0, m.start() - 200)
            end = min(len(text), m.end() + 200)
            result.setdefault(abbrev, []).append(text[start:end])
    return result


def _llm_batch_classify(
    contexts: dict[str, list[str]],
    conflicts: dict,
    client: OpenAI,
    model: str,
) -> dict[str, str]:
    items = []
    for abbrev, ctxs in contexts.items():
        candidates = conflicts[abbrev]["candidates"]
        items.append(
            f'약어: "{abbrev}"\n후보: {candidates}\n문맥: "...{ctxs[0]}..."'
        )
    prompt = (
        "다음 약어들의 의미를 문맥에 맞는 후보 중 하나로만 분류해줘. "
        "확신이 없으면 \"uncertain\"으로 답해줘.\n"
        'JSON 형식으로만 답해줘: {"약어": "선택한 후보 또는 uncertain"}\n\n'
        + "\n\n".join(items)
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {abbrev: "uncertain" for abbrev in contexts}


def _apply_conflict_resolutions(
    text: str,
    conflicts: dict,
    resolutions: dict[str, str],
) -> str:
    for abbrev, chosen in resolutions.items():
        candidates: list[str] = conflicts[abbrev]["candidates"]
        expansion = chosen if chosen in candidates else "|".join(candidates)
        text = re.sub(rf"\b{re.escape(abbrev)}\b(?!\()", f"{abbrev}({expansion})", text)
        for variant in conflicts[abbrev].get("variants", []):
            text = re.sub(rf"\b{re.escape(variant)}\b(?!\()", f"{variant}({expansion})", text)
    return text


def expand_query(
    query: str,
    acronyms: dict,
    reverse_index: dict[str, str],
    llm_client: OpenAI | None = None,
    model: str = "google/gemma-4-E4B-it",
) -> str:
    """쿼리 약어 정방향 확장 + 역방향 전체어→약어 추가."""
    # 정방향: 약어 → 전체어 병기
    expanded = map_abbreviations(query, acronyms, llm_client=llm_client, model=model)

    # 역방향: 전체어 토큰 → 약어 주입
    tokens = query.split()
    extra: list[str] = []
    for token in tokens:
        clean = token.strip(".,;:?!()")
        abbrev = reverse_index.get(clean) or reverse_index.get(clean.lower())
        if abbrev and abbrev not in expanded:
            extra.append(abbrev)

    if extra:
        expanded = expanded + " " + " ".join(extra)
    return expanded
