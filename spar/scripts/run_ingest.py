#!/usr/bin/env python3
"""SPAR md → Milvus ingest 파이프라인.

입력: .md (또는 .txt)
단계: read → abbrev_map → chunk → embed → milvus insert

PDF 입력은 별도 단계: scripts/convert_pdf_to_md.py 사용 후 본 명령으로 이어감.

Usage:
    python scripts/run_ingest.py --input-file data/skt-md/parameter_ref/foo.md \\
        --doc-type parameter_ref
    python scripts/run_ingest.py --input-dir data/tspec-llm/3GPP-clean/Rel-18/ \\
        --doc-type spec
    python scripts/run_ingest.py --input-file ... --doc-type ... --dry-run
    python scripts/run_ingest.py --input-file ... --doc-type ... --force
"""

from __future__ import annotations

import argparse
import json as _json
import re as _re
import sys
from pathlib import Path
from typing import Any, ContextManager

# src/ 레이아웃 — 설치 없이 직접 실행 시 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# scripts/ 디렉토리 — extract_acronyms 직접 import
sys.path.insert(0, str(Path(__file__).parent))

# .env 자동 로드 (python-dotenv 설치 시)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from extract_acronyms import find_abbreviations_section, merge_into, parse_entries, to_dict_schema
from spar.ingest.chunkers import dispatch as chunk_dispatch
from spar.preprocessing.abbrev_mapper import (
    extract_terms,
    get_all_keywords,
    load_acronyms,
    load_entity_glossary,
    map_abbreviations,
)
from spar.retrieval.milvus_client import DOC_TYPES, EMBED_DIM, SparMilvusClient

ALLOWED_SUFFIXES = {".md", ".txt"}

_SPEC_FNAME_RE = _re.compile(r"^(\d{2})(\d{3})(?!\d)")


def _parse_spec_number(filename: str) -> str:
    """'29502-i40.md' → '29.502'. 매칭 실패 시 ''."""
    stem = Path(filename).stem
    m = _SPEC_FNAME_RE.match(stem)
    if not m:
        return ""
    return f"{m.group(1)}.{m.group(2)}"


_ACRONYMS_PATH = Path(__file__).parent.parent / "dictionary" / "acronyms.json"
_ACRONYMS: dict = load_acronyms(_ACRONYMS_PATH) if _ACRONYMS_PATH.exists() else {}
_ENTITIES_PATH = Path(__file__).parent.parent / "dictionary" / "samsung_entities.json"
_ENTITIES: dict = load_entity_glossary(_ENTITIES_PATH)
_KEYWORDS: set[str] = get_all_keywords(_ACRONYMS, _ENTITIES)
# NOTE: _KEYWORDS is computed once at import. If _update_acronyms() later extends
# _ACRONYMS, _KEYWORDS will not reflect the new entries for that run.


def _update_acronyms(text: str) -> None:
    """문서 텍스트에서 약어 추출 후 전역 사전에 병합."""
    global _ACRONYMS
    section = find_abbreviations_section(text)
    if section is None:
        return
    entries = parse_entries(section)
    if not entries:
        return
    schema = to_dict_schema(entries)
    before = len(_ACRONYMS.get("global", {}))
    _ACRONYMS = merge_into(_ACRONYMS, schema)
    added = len(_ACRONYMS.get("global", {})) - before
    if added:
        print(f"  acronyms: +{added} new entries (total {len(_ACRONYMS['global'])})")


def _save_acronyms() -> None:
    """갱신된 약어 사전을 파일에 저장."""
    _ACRONYMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ACRONYMS_PATH.write_text(
        _json.dumps(_ACRONYMS, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"acronyms saved: {len(_ACRONYMS.get('global', {}))} entries → {_ACRONYMS_PATH}")


def _find_chunk_keywords(text: str, keywords: set[str]) -> list[str]:
    return extract_terms(text, keywords)[:50]


def read_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise SystemExit(
            f"ERROR: '{suffix}' 형식은 본 명령이 처리하지 않음.\n"
            f"  PDF 등은 먼저 변환: python scripts/convert_pdf_to_md.py "
            f"--input-file {file_path} --doc-type <type>"
        )
    return file_path.read_text(encoding="utf-8")


def embed_rows(rows: list[dict[str, Any]], *, dry_run: bool) -> list[dict[str, Any]]:
    if dry_run:
        # dry-run: 빈 벡터 (Milvus 미접속, 형식만 검증)
        for r in rows:
            r["embedding"] = [0.0] * EMBED_DIM
        return rows

    from spar.ingest.embedder import Embedder
    embedder = Embedder()
    vectors = embedder.encode([r["text"] for r in rows], verbose=True)
    for r, vec in zip(rows, vectors, strict=True):
        r["embedding"] = vec
    return rows


def ingest_file(
    client: SparMilvusClient | None,
    file_path: Path,
    doc_type: str,
    *,
    force: bool,
    dry_run: bool,
    intro_only: bool = False,
    update_acronyms: bool = True,
    llm_client: Any = None,
    llm_model: str = "google/gemma-4-E4B-it",
) -> int:
    source_doc = file_path.name
    spec_number = _parse_spec_number(source_doc) if intro_only else ""
    print(f"Processing: {file_path}  [doc_type={doc_type}]")

    text = read_text(file_path)

    # 약어 사전 갱신 (directory 모드에서는 pre-pass 완료 후라 중복이지만 안전)
    if update_acronyms:
        _update_acronyms(text)

    # 약어 매핑 — chunking 직전 (병기 확장)
    if _ACRONYMS:
        text = map_abbreviations(text, _ACRONYMS, llm_client=llm_client, model=llm_model)

    chunks = chunk_dispatch(text, source_doc=source_doc, doc_type=doc_type)
    # doc_type 강제 (chunkers.dispatch가 spec 청크에 'spec' 박지만, 명시 보장)
    for c in chunks:
        c["doc_type"] = doc_type
        c["keywords"] = _find_chunk_keywords(c["text"], _KEYWORDS)
    if spec_number:
        for c in chunks:
            if "spec_number" not in c:
                c["spec_number"] = spec_number
    print(f"  parsed: {len(text)} chars  →  {len(chunks)} chunks")
    if not chunks:
        return 0

    rows = embed_rows(chunks, dry_run=dry_run)

    if dry_run:
        print(f"  [DRY RUN] would insert {len(rows)} chunks — skipping Milvus write")
        if spec_number:
            print(f"  spec_number={spec_number!r} (dynamic field)")
        for r in rows[:2]:
            preview = r["text"][:80].replace("\n", " ")
            kw_preview = r.get("keywords", [])[:5]
            print(f"    chunk_id={r['chunk_id']}  section={r['section']!r}  keywords={kw_preview}  text={preview!r}")
        return len(rows)

    if client is None:
        raise RuntimeError("BUG: client is None after dry-run check")
    if force:
        client.delete_by_source(doc_type, source_doc)
        print(f"  deleted existing chunks for source_doc={source_doc!r}")
    client.insert(doc_type, rows)
    print(f"  inserted: {len(rows)} chunks → spar_{doc_type}")
    return len(rows)


def ingest_excel_file(
    client: "SparMilvusClient | None",
    file_path: Path,
    doc_type: str,
    *,
    force: bool,
    dry_run: bool,
) -> int:
    from spar.ingest.chunkers import dispatch_records
    from spar.parsers.alarm_ref_parser import parse_alarm_ref_excel
    from spar.parsers.counter_ref_parser import parse_counter_ref_excel
    from spar.parsers.parameter_ref_parser import parse_parameter_ref_excel

    _PARSER_MAP = {
        "parameter_ref": parse_parameter_ref_excel,
        "counter_ref": parse_counter_ref_excel,
        "alarm_ref": parse_alarm_ref_excel,
    }
    if doc_type not in _PARSER_MAP:
        raise SystemExit(
            f"ERROR: doc_type '{doc_type}' not supported for Excel ingest. "
            f"Use one of: {sorted(_PARSER_MAP)}"
        )

    source_doc = file_path.name
    print(f"Processing: {file_path}  [doc_type={doc_type}]")

    result = _PARSER_MAP[doc_type](file_path)
    records = result.records
    print(f"  parsed: {len(records)} records")

    chunks = dispatch_records(records, source_doc=source_doc, doc_type=doc_type)
    for c in chunks:
        c["doc_type"] = doc_type
        c["keywords"] = _find_chunk_keywords(c["text"], _KEYWORDS)
    print(f"  chunked: {len(chunks)} chunks")

    if not chunks:
        return 0

    rows = embed_rows(chunks, dry_run=dry_run)

    if dry_run:
        print(f"  [DRY RUN] would insert {len(rows)} chunks — skipping Milvus write")
        for r in rows[:2]:
            preview = r["text"][:80].replace("\n", " ")
            kw_preview = r.get("keywords", [])[:5]
            print(f"    chunk_id={r['chunk_id']}  section={r['section']!r}  keywords={kw_preview}  text={preview!r}")
        return len(rows)

    if client is None:
        raise RuntimeError("BUG: client is None after dry-run check")
    if force:
        client.delete_by_source(doc_type, source_doc)
        print(f"  deleted existing chunks for source_doc={source_doc!r}")
    client.insert(doc_type, rows)
    print(f"  inserted: {len(rows)} chunks → spar_{doc_type}")
    return len(rows)


def ingest_directory(
    client: SparMilvusClient | None,
    input_dir: Path,
    doc_type: str,
    *,
    force: bool,
    dry_run: bool,
    intro_only: bool = False,
    llm_client: Any = None,
    llm_model: str = "google/gemma-4-E4B-it",
) -> None:
    # TODO(Pass B): xlsx files in input_dir are not routed to ingest_excel_file — add when needed
    md_files = sorted(input_dir.rglob("*.md"))
    if not md_files:
        print(f"No .md files found under {input_dir}")
        return
    print(f"Found {len(md_files)} md files under {input_dir}")

    # Phase 1: 약어 pre-pass — 모든 문서 순회 후 acronyms.json 갱신
    print("\nPhase 1: collecting acronyms...")
    for f in md_files:
        try:
            _update_acronyms(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  WARN acronym extraction {f.name}: {e}", file=sys.stderr)
    _save_acronyms()

    # Phase 2: 실제 ingest (약어 사전 이미 갱신됨 — 중복 update 생략)
    n_files = len(md_files)
    print(f"\nPhase 2: ingesting {n_files} files...")
    total = 0
    for idx, f in enumerate(md_files, 1):
        print(f"[{idx}/{n_files}]", end=" ", flush=True)
        try:
            total += ingest_file(
                client, f, doc_type,
                force=force, dry_run=dry_run, intro_only=intro_only,
                update_acronyms=False, llm_client=llm_client, llm_model=llm_model,
            )
        except Exception as e:
            print(f"  ERROR processing {f}: {type(e).__name__}: {e}", file=sys.stderr)
            # continue to next file
    print(f"\nDone. Total chunks: {total}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="md 파일을 청크/임베딩하여 Milvus에 삽입"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-file", type=Path, metavar="FILE")
    src.add_argument("--input-dir", type=Path, metavar="DIR")
    parser.add_argument("--doc-type", required=True, choices=DOC_TYPES)
    parser.add_argument("--force", action="store_true",
                        help="동일 source_doc 기존 청크 삭제 후 재삽입")
    parser.add_argument("--dry-run", action="store_true",
                        help="Milvus 미접속 — 청킹/포맷만 검증")
    parser.add_argument(
        "--intro-only",
        action="store_true",
        help="파일명에서 spec_number 파싱 후 청크 dynamic field로 부착 (spec doc_type 전용)",
    )
    parser.add_argument(
        "--llm-url",
        default=None,
        metavar="URL",
        help="vLLM 서버 URL (예: http://localhost:8000/v1). 지정 시 약어 충돌을 LLM으로 분류.",
    )
    parser.add_argument(
        "--llm-model",
        default="google/gemma-4-E4B-it",
        metavar="MODEL",
        help="약어 충돌 분류에 사용할 모델명 (기본: google/gemma-4-E4B-it)",
    )

    args = parser.parse_args()
    if args.input_file and not args.input_file.exists():
        print(f"ERROR: file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    if args.input_dir and not args.input_dir.is_dir():
        print(f"ERROR: not a directory: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    llm_client = None
    if args.llm_url:
        try:
            from openai import OpenAI
            llm_client = OpenAI(base_url=args.llm_url, api_key="EMPTY")
            print(f"LLM conflict resolver: {args.llm_url}  model={args.llm_model}")
        except ImportError:
            print("WARN: openai 패키지 없음 — LLM 충돌 분류 비활성", file=sys.stderr)

    client_ctx: ContextManager[SparMilvusClient | None]
    if args.dry_run:
        print("[DRY RUN] Milvus connection skipped\n")
        client_ctx = _NullCtx()  # type: ignore[assignment]
    else:
        client_ctx = SparMilvusClient()

    with client_ctx as client:
        if args.input_file:
            if args.input_file.suffix.lower() == ".xlsx":
                # Excel ingest: no acronym extraction, so _save_acronyms() is intentionally omitted
                ingest_excel_file(
                    client, args.input_file, args.doc_type,
                    force=args.force, dry_run=args.dry_run,
                )
            else:
                ingest_file(client, args.input_file, args.doc_type,
                            force=args.force, dry_run=args.dry_run, intro_only=args.intro_only,
                            llm_client=llm_client, llm_model=args.llm_model)
                _save_acronyms()
        else:
            ingest_directory(client, args.input_dir, args.doc_type,
                             force=args.force, dry_run=args.dry_run, intro_only=args.intro_only,
                             llm_client=llm_client, llm_model=args.llm_model)


class _NullCtx:
    def __enter__(self) -> None: return None
    def __exit__(self, *_: object) -> None: pass


if __name__ == "__main__":
    main()
