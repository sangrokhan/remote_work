#!/usr/bin/env python3
"""SPAR 문서 전처리 파이프라인 entrypoint.

파이프라인: 파일 입력 → 파싱 → 청킹 → 임베딩 → Milvus 삽입

Usage:
    python scripts/run_ingest.py --input-file data/raw/foo.txt --doc-type parameter_ref
    python scripts/run_ingest.py --input-dir data/raw/ --doc-type mop
    python scripts/run_ingest.py --input-file ... --doc-type ... --force
    python scripts/run_ingest.py --input-file ... --doc-type ... --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

# src/ 레이아웃 — 설치 없이 직접 실행 시 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spar.retrieval.milvus_client import DOC_TYPES, EMBED_DIM, SparMilvusClient

# ---------------------------------------------------------------------------
# 스텁: 파서 (Task 1.1 완료 후 교체)
# ---------------------------------------------------------------------------


def parse_document(file_path: Path, doc_type: str) -> str:  # noqa: ARG001
    """문서를 텍스트로 변환.

    현재: .txt/.md 직독, 나머지는 스텁 반환.
    TODO(Task 1.1): doc_type별 전용 파서(pdfplumber/unstructured)로 교체.
    """
    suffix = file_path.suffix.lower()
    if suffix in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8")
    # PDF 등 미지원 포맷 — 파서 구현 전 임시 처리
    print(f"  [WARN] Unsupported format '{suffix}', using stub text")
    return f"[STUB] placeholder content for {file_path.name}"


# ---------------------------------------------------------------------------
# 스텁: 청커 (Task 1.3 완료 후 교체)
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    doc_type: str,  # noqa: ARG001
    source_doc: str,
) -> list[dict[str, Any]]:
    """텍스트를 Milvus 스키마 호환 청크 목록으로 분할.

    현재: 단어 기준 고정 크기 분할(500 단어).
    TODO(Task 1.3): doc_type별 전략으로 교체.
      - parameter_ref: 항목(parameter) 1개 = 청크 1개
      - mop/install_guide: 절차 헤더 + 단계 묶음
      - feature_desc: 섹션 단위 + 부모 컨텍스트
      - release_notes: 변경 항목 단위
    """
    _CHUNK_WORDS = 500  # 임시 고정값 — Task 1.3에서 유형별로 교체

    words = text.split()
    if not words:
        return []

    chunks: list[dict[str, Any]] = []
    for idx, start in enumerate(range(0, len(words), _CHUNK_WORDS)):
        chunk_words = words[start : start + _CHUNK_WORDS]
        chunk_id = hashlib.sha1(f"{source_doc}::{idx}".encode()).hexdigest()[:24]
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_type": doc_type,
                "product": "",           # TODO: 파서에서 추출 (LTE | NR | both)
                "release": "",           # TODO: 파서에서 추출 (예: v6.0)
                "deployment_type": "",   # TODO: 파서에서 추출
                "mo_name": "",           # TODO: parameter_ref 전용
                "source_doc": source_doc,
                "section": "",           # TODO: 파서에서 섹션 헤더 추출
                "page": 0,               # TODO: 파서에서 페이지 번호 추출
                "text": " ".join(chunk_words),
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# 스텁: 임베더 (Task 1.4 완료 후 교체)
# ---------------------------------------------------------------------------


def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """각 청크 텍스트를 밀집 벡터로 변환.

    현재: 랜덤 1024차원 벡터 (스텁).
    TODO(Task 1.4): sentence-transformers BGE-large-en-v1.5 또는 E5-large-v2로 교체.
      from sentence_transformers import SentenceTransformer
      model = SentenceTransformer("BAAI/bge-large-en-v1.5")
      vecs = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    """
    import random

    print(
        f"  [WARN] Using random stub embeddings (dim={EMBED_DIM})"
        " — replace with real model in Task 1.4"
    )
    result = []
    for chunk in chunks:
        row = dict(chunk)
        row["embedding"] = [random.gauss(0, 0.1) for _ in range(EMBED_DIM)]
        result.append(row)
    return result


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------


def ingest_file(
    client: SparMilvusClient | None,
    file_path: Path,
    doc_type: str,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """단일 파일을 전처리하여 Milvus에 삽입.

    Returns:
        삽입(또는 dry-run 시 생성)된 청크 수.
    """
    source_doc = file_path.name
    print(f"Processing: {file_path}  [doc_type={doc_type}]")

    # 1단계: 파싱
    text = parse_document(file_path, doc_type)
    print(f"  parsed: {len(text)} chars")

    # 2단계: 청킹
    chunks = chunk_text(text, doc_type, source_doc)
    print(f"  chunked: {len(chunks)} chunks")
    if not chunks:
        print("  skipped: no chunks generated")
        return 0

    # 3단계: 임베딩
    rows = embed_chunks(chunks)

    # dry-run: Milvus 쓰기 없이 미리보기만
    if dry_run:
        print(f"  [DRY RUN] would insert {len(rows)} chunks — skipping Milvus write")
        for row in rows[:2]:
            preview = row["text"][:80].replace("\n", " ")
            print(f"    chunk_id={row['chunk_id']}  text={preview!r}...")
        return len(rows)

    assert client is not None

    # 4단계: 기존 청크 삭제 (--force 시)
    if force:
        client.delete_by_source(doc_type, source_doc)
        print(f"  deleted existing chunks for source_doc={source_doc!r}")

    # 5단계: Milvus 삽입
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
) -> None:
    """디렉토리 내 모든 파일을 순서대로 전처리."""
    files = sorted(f for f in input_dir.iterdir() if f.is_file() and not f.name.startswith("."))
    if not files:
        print(f"No files found in {input_dir}")
        return

    total = 0
    for f in files:
        total += ingest_file(client, f, doc_type, force=force, dry_run=dry_run)

    print(f"\nDone. Total chunks: {total}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SPAR 문서 전처리 파이프라인 (파싱 → 청킹 → 임베딩 → Milvus 삽입)"
    )

    # 입력 소스 (둘 중 하나 필수)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-file", type=Path, metavar="FILE", help="처리할 단일 파일")
    input_group.add_argument(
        "--input-dir", type=Path, metavar="DIR", help="처리할 디렉토리 (비재귀)"
    )

    parser.add_argument(
        "--doc-type",
        required=True,
        choices=DOC_TYPES,
        metavar="DOC_TYPE",
        help=f"문서 유형 (선택: {', '.join(DOC_TYPES)})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="동일 source_doc의 기존 청크를 삭제 후 재삽입",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Milvus 연결/삽입 없이 파이프라인만 실행 (파싱·청킹·임베딩 확인용)",
    )

    args = parser.parse_args()

    # 경로 유효성 검사
    if args.input_file and not args.input_file.exists():
        print(f"ERROR: file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    if args.input_dir and not args.input_dir.is_dir():
        print(f"ERROR: not a directory: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("[DRY RUN] Milvus connection skipped\n")
        client_ctx = _NullContext()
    else:
        client_ctx = SparMilvusClient()

    with client_ctx as client:
        if args.input_file:
            ingest_file(
                client,
                args.input_file,
                args.doc_type,
                force=args.force,
                dry_run=args.dry_run,
            )
        else:
            ingest_directory(
                client,
                args.input_dir,
                args.doc_type,
                force=args.force,
                dry_run=args.dry_run,
            )


class _NullContext:
    """dry-run용 no-op 컨텍스트 매니저."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        pass


if __name__ == "__main__":
    main()
