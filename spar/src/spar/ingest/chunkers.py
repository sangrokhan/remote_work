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
_SECTION_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.*)")
_FENCED_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)


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


def _parse_section_header(raw: str) -> tuple[str, str]:
    """'4.1.1 Overview' → ('4.1.1', 'Overview'). No number → ('', raw)."""
    raw = re.sub(r"\s+#+\s*$", "", raw).strip()
    m = _SECTION_NUM_RE.match(raw)
    if m:
        return m.group(1), m.group(2).strip()
    return "", raw


def _parent_section_nums(section_num: str) -> list[str]:
    """'4.1.1' → ['4', '4.1']"""
    if not section_num:
        return []
    parts = section_num.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


def _split_with_overlap(words: list[str], max_words: int, overlap: int) -> list[list[str]]:
    if len(words) <= max_words:
        return [words]
    result: list[list[str]] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        result.append(words[start:end])
        if end == len(words):
            break
        start = end - overlap
    return result


def chunk_3gpp_sections(
    text: str,
    source_doc: str,
    *,
    max_words: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """### 기준 섹션 분할. max_words 초과 시 overlap 슬라이딩 윈도우로 재분할.

    각 청크 메타데이터:
      section_num         e.g. "4.1.1"
      section_title       e.g. "Overview"
      section_depth       int (3 = ###, 4 = ####, ...)
      parent_sections     list[str], e.g. ["4", "4.1"]
      chunk_index         global sequential index
      chunk_index_in_section  index within the section (0 when no split)
    """
    if not text.strip():
        return []

    fenced = [(m.start(), m.end()) for m in _FENCED_RE.finditer(text)]

    def _in_fenced(pos: int) -> bool:
        return any(s <= pos < e for s, e in fenced)

    matches = [m for m in _HEADER_RE.finditer(text) if not _in_fenced(m.start())]
    if not matches:
        return chunk_fixed(text, source_doc=source_doc, doc_type="spec", words=max_words)

    sections: list[tuple[re.Match[str], str]] = []
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((m, body))

    chunks: list[Chunk] = []
    global_idx = 0
    for header_match, body in sections:
        depth = len(header_match.group(1))
        raw_header = header_match.group(2).strip()
        section_num, section_title = _parse_section_header(raw_header)
        parents = _parent_section_nums(section_num)

        words = body.split()
        if not words:
            continue

        sub_chunks = _split_with_overlap(words, max_words, overlap)
        for sub_idx, sub_words in enumerate(sub_chunks):
            piece = " ".join(sub_words)
            chunks.append(
                {
                    "chunk_id": _make_chunk_id(source_doc, global_idx, section_num or raw_header),
                    "doc_type": "spec",
                    "source_doc": source_doc,
                    "section": f"{section_num} {section_title}".strip() if section_num else section_title,
                    "section_num": section_num,
                    "section_title": section_title,
                    "section_depth": depth,
                    "parent_sections": parents,
                    "chunk_index": global_idx,
                    "chunk_index_in_section": sub_idx,
                    "text": piece,
                    **_empty_meta(),
                }
            )
            global_idx += 1
    return chunks


def chunk_markdown(text: str, source_doc: str, *, max_words: int = 500) -> list[Chunk]:
    """헤더 경계로 분할. 섹션이 max_words 초과 시 단어 단위 추가 분할."""
    return chunk_3gpp_sections(text, source_doc, max_words=max_words, overlap=0)


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
                "section_num": "",
                "section_title": "",
                "section_depth": 0,
                "parent_sections": [],
                "chunk_index": idx,
                "chunk_index_in_section": 0,
                "text": piece,
                **_empty_meta(),
            }
        )
    return chunks


def dispatch(text: str, source_doc: str, *, doc_type: str) -> list[Chunk]:
    """doc_type → 청크 전략 라우팅 (텍스트 기반 문서)."""
    if doc_type == "spec":
        return chunk_3gpp_sections(text, source_doc=source_doc)
    return chunk_fixed(text, source_doc=source_doc, doc_type=doc_type)


def dispatch_records(records: list, source_doc: str, *, doc_type: str) -> list[Chunk]:
    """doc_type → 청크 전략 라우팅 (Excel record 기반 문서: parameter_ref/counter_ref/alarm_ref)."""
    from spar.chunkers.reference_chunker import chunk_alarm_ref, chunk_counter_ref, chunk_parameter_ref

    if doc_type == "parameter_ref":
        return chunk_parameter_ref(records, source_doc=source_doc)
    if doc_type == "counter_ref":
        return chunk_counter_ref(records, source_doc=source_doc)
    if doc_type == "alarm_ref":
        return chunk_alarm_ref(records, source_doc=source_doc)
    raise ValueError(f"dispatch_records: unsupported doc_type '{doc_type}'")
