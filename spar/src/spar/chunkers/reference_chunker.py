"""Reference 문서(Parameter/Counter/Alarm) Excel record → Chunk 변환. 1 record = 1 chunk."""
from __future__ import annotations

from typing import TYPE_CHECKING

from spar.ingest.chunkers import Chunk, _empty_meta, _make_chunk_id

if TYPE_CHECKING:
    from spar.parsers.alarm_ref_parser import AlarmRecord
    from spar.parsers.counter_ref_parser import CounterRecord
    from spar.parsers.parameter_ref_parser import ParameterRecord


def chunk_parameter_ref(records: list[ParameterRecord], source_doc: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, r in enumerate(records):
        meta = _empty_meta()
        meta["mo_name"] = r.leaf_mo
        chunk: Chunk = {
            "chunk_id": _make_chunk_id(source_doc, i, r.param_name),
            "text": r.to_chunk_text(),
            "doc_type": "parameter_ref",
            "source_doc": source_doc,
            "section": r.param_name,
            "yang_path": r.yang_path,
            "parent_sections": [],
            "chunk_index": i,
            "keywords": [],
            **meta,
        }
        chunks.append(chunk)
    return chunks


def chunk_counter_ref(records: list[CounterRecord], source_doc: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, r in enumerate(records):
        meta = _empty_meta()
        meta["mo_name"] = r.mid_group
        chunk: Chunk = {
            "chunk_id": _make_chunk_id(source_doc, i, r.counter_name),
            "text": r.to_chunk_text(),
            "doc_type": "counter_ref",
            "source_doc": source_doc,
            "section": r.large_group,
            "parent_sections": [],
            "chunk_index": i,
            "keywords": [],
            **meta,
        }
        chunks.append(chunk)
    return chunks


def chunk_alarm_ref(records: list[AlarmRecord], source_doc: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, r in enumerate(records):
        meta = _empty_meta()
        chunk: Chunk = {
            "chunk_id": _make_chunk_id(source_doc, i, r.alarm_id),
            "text": r.to_chunk_text(),
            "doc_type": "alarm_ref",
            "source_doc": source_doc,
            "section": r.alarm_id,
            "parent_sections": [],
            "chunk_index": i,
            "keywords": [],
            **meta,
        }
        chunks.append(chunk)
    return chunks
