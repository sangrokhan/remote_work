from __future__ import annotations

import re
from typing import Any

Chunk = dict[str, Any]


def tag_chunk(chunk: Chunk, keywords: set[str]) -> Chunk:
    """chunk['text']에서 keywords 매칭 → chunk['keywords'] 채움. 대소문자 무시, 단어 경계 기준. max 50."""
    text = chunk.get("text", "")
    matched: list[str] = []
    for kw in keywords:
        if len(matched) >= 50:
            break
        pattern = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
        if pattern.search(text):
            matched.append(kw)
    chunk["keywords"] = matched
    return chunk
