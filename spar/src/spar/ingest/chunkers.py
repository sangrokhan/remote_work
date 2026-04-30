"""Doc-type별 텍스트 청킹 전략.

PRD Task 1.3 — md-aware는 헤더 경계로 분할, 나머지는 fixed-size fallback.
TODO(Task 1.3): mop/install_guide는 절차 헤더 + 단계 묶음, parameter_ref는 항목 단위로 교체.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

Chunk = dict[str, Any]

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _make_chunk_id(source_doc: str, idx: int, section: str) -> str:
    return hashlib.sha1(f"{source_doc}::{idx}::{section}".encode()).hexdigest()[:24]


def _empty_meta() -> dict[str, Any]:
    return {
        "product": "",
        "release": "",
        "deployment_type": "",
        "mo_name": "",
        "page": 0,
    }


_FENCED_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)


def chunk_markdown(text: str, source_doc: str, *, max_words: int = 500) -> list[Chunk]:
    """헤더 경계로 분할. 섹션이 max_words 초과 시 단어 단위 추가 분할."""
    if not text.strip():
        return []

    # Find fenced regions in original text to skip false-positive headers inside them
    fenced_regions = [(m.start(), m.end()) for m in _FENCED_RE.finditer(text)]

    def _in_fenced(pos: int) -> bool:
        return any(s <= pos < e for s, e in fenced_regions)

    matches = [m for m in _HEADER_RE.finditer(text) if not _in_fenced(m.start())]
    if not matches:
        # 헤더 없으면 fixed로 위임 (doc_type=spec 가정)
        return chunk_fixed(text, source_doc=source_doc, doc_type="spec", words=max_words)

    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        raw_section = m.group(2).strip()
        # Strip ATX closing hashes (e.g. "Architecture ##" → "Architecture")
        section = re.sub(r'\s+#+\s*$', '', raw_section).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((section, body))

    chunks: list[Chunk] = []
    idx = 0
    for section, body in sections:
        words = body.split()
        if not words:
            continue
        for w_start in range(0, len(words), max_words):
            piece = " ".join(words[w_start : w_start + max_words])
            chunks.append(
                {
                    "chunk_id": _make_chunk_id(source_doc, idx, section),
                    "doc_type": "spec",
                    "source_doc": source_doc,
                    "section": section,
                    "text": piece,
                    **_empty_meta(),
                }
            )
            idx += 1
    return chunks


def chunk_fixed(
    text: str, source_doc: str, *, doc_type: str, words: int = 500
) -> list[Chunk]:
    """단어 기준 고정 크기 분할 (헤더 무시)."""
    tokens = text.split()
    if not tokens:
        return []
    chunks: list[Chunk] = []
    for idx, start in enumerate(range(0, len(tokens), words)):
        piece = " ".join(tokens[start : start + words])
        chunks.append(
            {
                "chunk_id": _make_chunk_id(source_doc, idx, ""),
                "doc_type": doc_type,
                "source_doc": source_doc,
                "section": "",
                "text": piece,
                **_empty_meta(),
            }
        )
    return chunks


def dispatch(text: str, source_doc: str, *, doc_type: str) -> list[Chunk]:
    """doc_type → 청크 전략 라우팅."""
    if doc_type == "spec":
        return chunk_markdown(text, source_doc=source_doc)
    return chunk_fixed(text, source_doc=source_doc, doc_type=doc_type)
