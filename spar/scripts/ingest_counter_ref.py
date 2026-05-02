#!/usr/bin/env python3
"""CLI: Counter Reference Excel → Milvus ingest.

Usage:
    python scripts/ingest_counter_ref.py --file data/counter_ref/NR_counters.xlsx
    python scripts/ingest_counter_ref.py --file ... --sheet "LTE Counters" --product LTE
    python scripts/ingest_counter_ref.py --file ... --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from spar.parsers.counter_ref_parser import parse_counter_ref_excel
from spar.retrieval.milvus_client import EMBED_DIM, SparMilvusClient

DOC_TYPE = "counter_ref"


def _chunk_id(source_doc: str, counter_name: str) -> str:
    key = f"{source_doc}::{counter_name}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def _build_rows(
    path: Path,
    sheet_name: str | None,
    product: str,
    release: str,
) -> list[dict[str, Any]]:
    result = parse_counter_ref_excel(path, sheet_name=sheet_name)
    if result.warnings:
        for w in result.warnings[:10]:
            print(f"  [warn] {w}")
        if len(result.warnings) > 10:
            print(f"  ... 외 {len(result.warnings) - 10}개 경고")

    source_doc = path.name
    rows = []
    for rec in result.records:
        text = rec.to_chunk_text()
        rows.append({
            "chunk_id": _chunk_id(source_doc, rec.counter_name),
            "embedding": [],  # 임시 — embed_rows()에서 채움
            "doc_type": DOC_TYPE,
            "product": product,
            "release": release,
            "deployment_type": "",
            "mo_name": "",
            "source_doc": source_doc,
            "section": rec.mid_group,
            "page": 0,
            "section_num": rec.mid_group_id,
            "section_title": rec.mid_group,
            "section_depth": 2,
            "chunk_index": 0,
            "chunk_index_in_section": 0,
            "parent_sections": [rec.large_group] if rec.large_group else [],
            "keywords": [rec.counter_name] + ([rec.mid_group_id] if rec.mid_group_id else []),
            "text": text,
            # dynamic fields — counter-specific
            "large_group": rec.large_group,
            "mid_group": rec.mid_group,
            "mid_group_id": rec.mid_group_id,
            "counter_name": rec.counter_name,
            "period": rec.period,
            "unit": rec.unit,
            "min_val": rec.min_val,
            "max_val": rec.max_val,
        })
    return rows, result.skipped_rows


def embed_rows(rows: list[dict[str, Any]], *, dry_run: bool) -> list[dict[str, Any]]:
    if dry_run:
        for r in rows:
            r["embedding"] = [0.0] * EMBED_DIM
        return rows

    from spar.ingest.embedder import Embedder
    embedder = Embedder()
    vectors = embedder.encode([r["text"] for r in rows], verbose=True)
    for r, vec in zip(rows, vectors):
        r["embedding"] = vec
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Counter Reference Excel → Milvus ingest")
    parser.add_argument("--file", required=True, help="Excel 파일 경로 (.xlsx)")
    parser.add_argument("--sheet", default=None, help="시트명 (기본값: 첫 번째 시트)")
    parser.add_argument("--product", default="NR", choices=["NR", "LTE", "both"],
                        help="제품 구분 (기본값: NR)")
    parser.add_argument("--release", default="", help="릴리스 버전 (예: v7.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Milvus 미연결 — 파싱/임베딩만 수행")
    parser.add_argument("--force", action="store_true",
                        help="컬렉션 재생성 (기존 데이터 삭제)")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"파일 없음: {path}")

    print(f"파싱: {path}  (sheet={args.sheet or 'first'})")
    rows, skipped = _build_rows(path, args.sheet, args.product, args.release)
    print(f"  레코드: {len(rows)}개  스킵: {skipped}행")

    if not rows:
        sys.exit("파싱된 카운터 없음. 헤더 컬럼명 확인 필요.")

    print("임베딩 중...")
    rows = embed_rows(rows, dry_run=args.dry_run)

    if args.dry_run:
        print("[DRY RUN] Milvus 미연결 — 샘플 출력:")
        for r in rows[:3]:
            preview = r["text"][:80].replace("\n", " ")
            print(f"  chunk_id={r['chunk_id']}  counter={r['counter_name']}  text={preview!r}")
        return

    with SparMilvusClient() as client:
        if args.force:
            client.create_collection(DOC_TYPE, drop_if_exists=True)
            print(f"컬렉션 재생성: spar_{DOC_TYPE}")
        elif not client.collection_exists(DOC_TYPE):
            client.create_collection(DOC_TYPE)
            print(f"컬렉션 생성: spar_{DOC_TYPE}")

        client.insert(DOC_TYPE, rows)
        print(f"삽입 완료: {len(rows)}개 → spar_{DOC_TYPE}")


if __name__ == "__main__":
    main()
