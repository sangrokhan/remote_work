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
import sys
from pathlib import Path
from typing import Any, ContextManager

# src/ 레이아웃 — 설치 없이 직접 실행 시 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# .env 자동 로드 (python-dotenv 설치 시)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from spar.ingest.chunkers import dispatch as chunk_dispatch
from spar.preprocessing.abbrev_mapper import load_acronyms, map_abbreviations
from spar.retrieval.milvus_client import DOC_TYPES, EMBED_DIM, SparMilvusClient

ALLOWED_SUFFIXES = {".md", ".txt"}

import re as _re

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
    vectors = embedder.encode([r["text"] for r in rows])
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
) -> int:
    source_doc = file_path.name
    spec_number = _parse_spec_number(source_doc) if intro_only else ""
    print(f"Processing: {file_path}  [doc_type={doc_type}]")

    text = read_text(file_path)

    # 약어 매핑 — chunking 직전 (병기 확장)
    if _ACRONYMS:
        text = map_abbreviations(text, _ACRONYMS)

    chunks = chunk_dispatch(text, source_doc=source_doc, doc_type=doc_type)
    # doc_type 강제 (chunkers.dispatch가 spec 청크에 'spec' 박지만, 명시 보장)
    for c in chunks:
        c["doc_type"] = doc_type
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
            print(f"    chunk_id={r['chunk_id']}  section={r['section']!r}  text={preview!r}")
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
) -> None:
    md_files = sorted(input_dir.rglob("*.md"))
    if not md_files:
        print(f"No .md files found under {input_dir}")
        return
    print(f"Found {len(md_files)} md files under {input_dir}")
    total = 0
    for f in md_files:
        try:
            total += ingest_file(client, f, doc_type, force=force, dry_run=dry_run, intro_only=intro_only)
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

    args = parser.parse_args()
    if args.input_file and not args.input_file.exists():
        print(f"ERROR: file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    if args.input_dir and not args.input_dir.is_dir():
        print(f"ERROR: not a directory: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    client_ctx: ContextManager[SparMilvusClient | None]
    if args.dry_run:
        print("[DRY RUN] Milvus connection skipped\n")
        client_ctx = _NullCtx()  # type: ignore[assignment]
    else:
        client_ctx = SparMilvusClient()

    with client_ctx as client:
        if args.input_file:
            ingest_file(client, args.input_file, args.doc_type,
                        force=args.force, dry_run=args.dry_run, intro_only=args.intro_only)
        else:
            ingest_directory(client, args.input_dir, args.doc_type,
                             force=args.force, dry_run=args.dry_run, intro_only=args.intro_only)


class _NullCtx:
    def __enter__(self) -> None: return None
    def __exit__(self, *_: object) -> None: pass


if __name__ == "__main__":
    main()
