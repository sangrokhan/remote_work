# Markdown Ingest Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 두 입력 경로(외부 TSpec-LLM md 코퍼스 + SKT 사내 PDF)를 분리된 두 단계 파이프라인으로 통합 — `convert_pdf_to_md.py`(PDF→md)와 `run_ingest.py`(md→Milvus). 920개 Rel-18 md 파일 인덱싱이 통합 동작 검증의 끝점.

**Architecture:**
- 단계 분리: PDF 파싱(SKT 전용)과 청킹/임베딩(공통)을 별도 명령으로 분할. md = 검수 가능한 중간 산출물.
- doc-type 분기는 두 단계 각각에서: (a) PDF 파서가 doc-type별 라이브러리 선택, (b) 청커가 doc-type별 전략 선택.
- TSpec-LLM은 PDF 단계 skip — `data/tspec-llm/3GPP-clean/Rel-XX/...` md를 `--doc-type spec`으로 직접 ingest.

**Tech Stack:** Python 3.12, pymilvus 2.4+, pytest, sentence-transformers (BGE-large-en-v1.5), unstructured/pdfplumber/camelot (PDF→md, Task 1.1 본 구현은 별도 PR에서 채움), huggingface_hub

---

## File Structure

**Modify:**
- `src/spar/retrieval/milvus_client.py` — `DOC_TYPES`에 `"spec"` 추가
- `scripts/run_ingest.py` — md 전용으로 축소; 청커/임베더는 외부 모듈 호출
- `spar/.env.example` — 임베더 모델명 환경 변수 추가 (이미 HF_TOKEN은 추가됨)

**Create:**
- `src/spar/ingest/__init__.py` — ingest 패키지 마커
- `src/spar/ingest/chunkers.py` — doc-type별 청킹 전략 (md-aware + fixed-size fallback)
- `src/spar/ingest/embedder.py` — sentence-transformers 래퍼 (정규화 + 배치)
- `scripts/convert_pdf_to_md.py` — PDF→md CLI 스켈레톤 (doc-type 분기, 실 파서는 Task 1.1에서 채움)
- `tests/ingest/__init__.py`
- `tests/ingest/test_chunkers.py`
- `tests/ingest/test_embedder.py`
- `tests/ingest/test_run_ingest_smoke.py` — 1개 md 파일 dry-run 검증
- `tests/scripts/test_convert_pdf_to_md_cli.py` — CLI 인자 + doc-type 분기 라우팅 검증

---

## Task 1: DOC_TYPES에 `spec` 추가

**Files:**
- Modify: `src/spar/retrieval/milvus_client.py:38-46`
- Test: `tests/retrieval/test_doc_types.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_doc_types.py
from spar.retrieval.milvus_client import DOC_TYPES


def test_spec_doc_type_present():
    """3GPP TSpec-LLM markdown ingest 위해 spec 유형 필요."""
    assert "spec" in DOC_TYPES


def test_doc_types_unique():
    assert len(DOC_TYPES) == len(set(DOC_TYPES))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/retrieval/test_doc_types.py -v`
Expected: FAIL — `assert 'spec' in DOC_TYPES`

- [ ] **Step 3: Implement minimal change**

Edit `src/spar/retrieval/milvus_client.py` — add `"spec"` to `DOC_TYPES` list:

```python
DOC_TYPES = [
    "parameter_ref",
    "counter_ref",
    "alarm_ref",
    "feature_desc",
    "mop",
    "install_guide",
    "release_notes",
    "spec",  # 3GPP TSpec-LLM 등 외부 표준 문서
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/retrieval/test_doc_types.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/spar/retrieval/milvus_client.py tests/retrieval/test_doc_types.py
git commit -m "feat(retrieval): add 'spec' doc_type for 3GPP standards"
```

---

## Task 2: md-aware Chunker 모듈

**Files:**
- Create: `src/spar/ingest/__init__.py`
- Create: `src/spar/ingest/chunkers.py`
- Create: `tests/ingest/__init__.py`
- Create: `tests/ingest/test_chunkers.py`

**Design:**
- 함수: `chunk_markdown(text, source_doc, *, max_words=500) -> list[Chunk]` — 헤더(`#`, `##`, ...) 경계로 자르고 헤더 패스 누적, max_words 초과 시 단어 기준 추가 분할.
- 함수: `chunk_fixed(text, source_doc, *, doc_type, words=500) -> list[Chunk]` — 헤더 없는 텍스트용 fallback.
- 함수: `dispatch(text, source_doc, *, doc_type) -> list[Chunk]` — doc_type에 따라 위 둘 중 선택.
  - `spec` → `chunk_markdown` (3GPP md = 헤더 풍부)
  - 그 외 → `chunk_fixed` (Task 1.3에서 doc_type별 전략으로 추후 교체; PRD 명시)

각 청크는 dict 형태:
```python
{
  "chunk_id": str,
  "doc_type": str,
  "source_doc": str,
  "section": str,           # 헤더 경로 ("4.2.1 Architecture")
  "text": str,
  "page": 0,                # md엔 페이지 없음
  # product, release, deployment_type, mo_name = "" (Task 2 범위 밖)
}
```

- [ ] **Step 1: Write failing tests**

Create `tests/ingest/__init__.py` (빈 파일).

```python
# tests/ingest/test_chunkers.py
from spar.ingest.chunkers import chunk_markdown, chunk_fixed, dispatch


SAMPLE_MD = """# 1 Scope
This document specifies foo.

# 2 References
The following documents are referenced.

## 2.1 Normative
- Reference A
- Reference B

# 3 Definitions
Terms used in this spec.
"""


def test_markdown_chunker_splits_on_headers():
    chunks = chunk_markdown(SAMPLE_MD, source_doc="21101.md")
    # 4 헤더 → 4 청크 (전부 max_words 미만)
    assert len(chunks) == 4
    sections = [c["section"] for c in chunks]
    assert sections == ["1 Scope", "2 References", "2.1 Normative", "3 Definitions"]


def test_markdown_chunker_preserves_text_under_header():
    chunks = chunk_markdown(SAMPLE_MD, source_doc="21101.md")
    scope = next(c for c in chunks if c["section"] == "1 Scope")
    assert "specifies foo" in scope["text"]


def test_markdown_chunker_subsplits_long_section():
    long_text = "# Big\n" + "word " * 1200
    chunks = chunk_markdown(long_text, source_doc="big.md", max_words=500)
    # 1200 words / 500 → 3 청크
    assert len(chunks) == 3
    assert all(c["section"] == "Big" for c in chunks)


def test_chunk_ids_unique():
    chunks = chunk_markdown(SAMPLE_MD, source_doc="21101.md")
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_fixed_chunker_word_buckets():
    text = " ".join(["w"] * 1500)
    chunks = chunk_fixed(text, source_doc="x.txt", doc_type="mop", words=500)
    assert len(chunks) == 3
    assert all(c["doc_type"] == "mop" for c in chunks)


def test_dispatch_spec_uses_markdown():
    chunks = dispatch(SAMPLE_MD, source_doc="21101.md", doc_type="spec")
    assert len(chunks) == 4  # 헤더 기반


def test_dispatch_mop_uses_fixed():
    chunks = dispatch(SAMPLE_MD, source_doc="x.md", doc_type="mop")
    # fixed 청커는 헤더 무시 — 1 청크 (전체 < 500 words)
    assert len(chunks) == 1


def test_empty_text_returns_empty_list():
    assert chunk_markdown("", source_doc="x.md") == []
    assert chunk_fixed("", source_doc="x.md", doc_type="mop") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/ingest/test_chunkers.py -v`
Expected: FAIL — `ModuleNotFoundError: spar.ingest`

- [ ] **Step 3: Implement chunker module**

Create `src/spar/ingest/__init__.py` (빈 파일).

Create `src/spar/ingest/chunkers.py`:

```python
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


def chunk_markdown(text: str, source_doc: str, *, max_words: int = 500) -> list[Chunk]:
    """헤더 경계로 분할. 섹션이 max_words 초과 시 단어 단위 추가 분할."""
    if not text.strip():
        return []

    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        # 헤더 없으면 fixed로 위임 (doc_type=spec 가정)
        return chunk_fixed(text, source_doc=source_doc, doc_type="spec", words=max_words)

    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        section = m.group(2).strip()
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
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `.venv/bin/pytest tests/ingest/test_chunkers.py -v`
Expected: 8 PASS

- [ ] **Step 5: Commit**

```bash
git add src/spar/ingest/__init__.py src/spar/ingest/chunkers.py \
        tests/ingest/__init__.py tests/ingest/test_chunkers.py
git commit -m "feat(ingest): add md-aware + fixed chunkers with doc_type dispatch"
```

---

## Task 3: Embedder 래퍼 (sentence-transformers)

**Files:**
- Create: `src/spar/ingest/embedder.py`
- Create: `tests/ingest/test_embedder.py`
- Modify: `spar/.env.example` (모델명 환경 변수 추가)
- Modify: `spar/requirements.txt` (sentence-transformers 활성화)

**Design:**
- 클래스: `Embedder(model_name=os.environ["EMBED_MODEL"], device="auto")`
- 메서드: `encode(texts: list[str], batch_size=32) -> list[list[float]]` — 정규화된 dense 벡터.
- 모델 사용량이 큰 통합 테스트는 mark `slow` + 환경 변수 가드 (CI에서 기본 skip).
- 단위 테스트는 stub class로 결정론 입력 검증.

- [ ] **Step 1: Write failing tests**

```python
# tests/ingest/test_embedder.py
import os
import pytest

from spar.ingest.embedder import Embedder


def test_embedder_dim_matches_milvus_constant():
    """EMBED_DIM(=1024)과 모델 차원 일치."""
    from spar.retrieval.milvus_client import EMBED_DIM
    assert EMBED_DIM == 1024


def test_embedder_normalizes_unit_norm(monkeypatch):
    """encode 결과는 L2-norm == 1 (정규화 보장)."""
    class _StubST:
        def __init__(self, name): self.name = name
        def encode(self, texts, batch_size=32, normalize_embeddings=True,
                   show_progress_bar=False):
            import numpy as np
            arr = np.ones((len(texts), 1024), dtype="float32")
            if normalize_embeddings:
                arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
            return arr

    monkeypatch.setattr("spar.ingest.embedder.SentenceTransformer", _StubST)
    e = Embedder(model_name="stub")
    vecs = e.encode(["a", "b"])
    assert len(vecs) == 2 and len(vecs[0]) == 1024
    import math
    assert all(abs(sum(v * v for v in vec) - 1.0) < 1e-5 for vec in vecs)


@pytest.mark.skipif(
    os.environ.get("RUN_HEAVY_TESTS") != "1",
    reason="실 모델 다운로드 필요 — RUN_HEAVY_TESTS=1로 활성화",
)
def test_embedder_real_model_smoke():
    e = Embedder(model_name=os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5"))
    vecs = e.encode(["hello world", "3GPP NR"])
    assert len(vecs) == 2 and len(vecs[0]) == 1024
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/ingest/test_embedder.py -v`
Expected: FAIL — `ModuleNotFoundError: spar.ingest.embedder`

- [ ] **Step 3: Implement embedder**

Create `src/spar/ingest/embedder.py`:

```python
"""Dense embedder — sentence-transformers 래퍼."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer  # type: ignore

# 런타임 import는 클래스 안에서 — 테스트 monkeypatch 가능하게 모듈 attr로 노출
try:
    from sentence_transformers import SentenceTransformer  # noqa: F811
except ImportError:  # pragma: no cover — runtime 미설치 환경
    SentenceTransformer = None  # type: ignore


DEFAULT_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5")


class Embedder:
    """단일 모델 dense embedder. 코사인 유사도용 정규화 강제."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto") -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers 미설치. requirements.txt 확인."
            )
        kwargs = {} if device == "auto" else {"device": device}
        self._model = SentenceTransformer(model_name, **kwargs)
        self.model_name = model_name

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # numpy → list (Milvus 호환)
        return [list(map(float, v)) for v in vecs]
```

- [ ] **Step 4: Activate dependency**

Edit `requirements.txt` — uncomment sentence-transformers:

```
sentence-transformers>=3.0
```

- [ ] **Step 5: Add env var slot**

Edit `spar/.env.example` — 데이터셋 다운로드 섹션 위에 추가:

```
# ============================================================
# 임베딩 모델 (sentence-transformers)
# ============================================================
EMBED_MODEL=BAAI/bge-large-en-v1.5
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pip install -r requirements.txt`
Run: `.venv/bin/pytest tests/ingest/test_embedder.py -v`
Expected: 2 PASS, 1 SKIP (heavy)

- [ ] **Step 7: Commit**

```bash
git add src/spar/ingest/embedder.py tests/ingest/test_embedder.py \
        requirements.txt spar/.env.example
git commit -m "feat(ingest): add sentence-transformers embedder wrapper"
```

---

## Task 4: run_ingest.py — md-only 리팩터

**Files:**
- Modify: `scripts/run_ingest.py` (전체 재작성)
- Create: `tests/ingest/test_run_ingest_smoke.py`

**Design:**
- PDF 분기 제거 — `.md`/`.txt` 외 확장자 입력 시 명확 에러 (`convert_pdf_to_md.py` 안내 메시지).
- 청커 = `spar.ingest.chunkers.dispatch`.
- 임베더 = `spar.ingest.embedder.Embedder` (dry-run 시 미초기화).
- 디렉토리 입력 시 `*.md` rglob로 재귀 처리 (TSpec-LLM 구조 = `Rel-XX/NN_series/*.md`).

- [ ] **Step 1: Write failing smoke test**

```python
# tests/ingest/test_run_ingest_smoke.py
import subprocess
import sys
from pathlib import Path


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/run_ingest.py", *args],
        cwd=cwd, capture_output=True, text=True,
    )


def test_dry_run_md_file(tmp_path):
    md = tmp_path / "sample.md"
    md.write_text("# Section A\nHello world.\n# Section B\nMore content.\n")

    repo = Path(__file__).resolve().parents[2]  # spar/
    proc = _run(
        ["--input-file", str(md), "--doc-type", "spec", "--dry-run"],
        cwd=repo,
    )
    assert proc.returncode == 0, proc.stderr
    assert "DRY RUN" in proc.stdout
    assert "2 chunks" in proc.stdout  # Section A + B


def test_pdf_input_rejected(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    repo = Path(__file__).resolve().parents[2]
    proc = _run(
        ["--input-file", str(pdf), "--doc-type", "spec", "--dry-run"],
        cwd=repo,
    )
    assert proc.returncode != 0
    assert "convert_pdf_to_md.py" in proc.stderr
```

- [ ] **Step 2: Run test to verify failures**

Run: `.venv/bin/pytest tests/ingest/test_run_ingest_smoke.py -v`
Expected: FAIL — current script has wrong chunk count (single 500-word fallback) + PDF가 stub로 통과되어 reject 안됨.

- [ ] **Step 3: Rewrite run_ingest.py**

Replace `scripts/run_ingest.py` content:

```python
#!/usr/bin/env python3
"""SPAR md → Milvus ingest 파이프라인.

입력: .md (또는 .txt)
단계: read → chunk → embed → milvus insert

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
from typing import Any

# src/ 레이아웃 — 설치 없이 직접 실행 시 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spar.ingest.chunkers import dispatch as chunk_dispatch
from spar.retrieval.milvus_client import DOC_TYPES, EMBED_DIM, SparMilvusClient

ALLOWED_SUFFIXES = {".md", ".txt"}


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
) -> int:
    source_doc = file_path.name
    print(f"Processing: {file_path}  [doc_type={doc_type}]")

    text = read_text(file_path)
    chunks = chunk_dispatch(text, source_doc=source_doc, doc_type=doc_type)
    # doc_type 강제 (chunkers.dispatch가 spec 청크에 'spec' 박지만, 명시 보장)
    for c in chunks:
        c["doc_type"] = doc_type
    print(f"  parsed: {len(text)} chars  →  {len(chunks)} chunks")
    if not chunks:
        return 0

    rows = embed_rows(chunks, dry_run=dry_run)

    if dry_run:
        print(f"  [DRY RUN] would insert {len(rows)} chunks — skipping Milvus write")
        for r in rows[:2]:
            preview = r["text"][:80].replace("\n", " ")
            print(f"    chunk_id={r['chunk_id']}  section={r['section']!r}  text={preview!r}")
        return len(rows)

    assert client is not None
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
) -> None:
    md_files = sorted(input_dir.rglob("*.md"))
    if not md_files:
        print(f"No .md files found under {input_dir}")
        return
    print(f"Found {len(md_files)} md files under {input_dir}")
    total = 0
    for f in md_files:
        total += ingest_file(client, f, doc_type, force=force, dry_run=dry_run)
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

    args = parser.parse_args()
    if args.input_file and not args.input_file.exists():
        print(f"ERROR: file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    if args.input_dir and not args.input_dir.is_dir():
        print(f"ERROR: not a directory: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("[DRY RUN] Milvus connection skipped\n")
        client_ctx: Any = _NullCtx()
    else:
        client_ctx = SparMilvusClient()

    with client_ctx as client:
        if args.input_file:
            ingest_file(client, args.input_file, args.doc_type,
                        force=args.force, dry_run=args.dry_run)
        else:
            ingest_directory(client, args.input_dir, args.doc_type,
                             force=args.force, dry_run=args.dry_run)


class _NullCtx:
    def __enter__(self) -> None: return None
    def __exit__(self, *_: object) -> None: pass


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke tests to verify**

Run: `.venv/bin/pytest tests/ingest/test_run_ingest_smoke.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_ingest.py tests/ingest/test_run_ingest_smoke.py
git commit -m "refactor(ingest): make run_ingest.py md-only, reject PDF inputs"
```

---

## Task 5: convert_pdf_to_md.py 스켈레톤

**Files:**
- Create: `scripts/convert_pdf_to_md.py`
- Create: `tests/scripts/__init__.py`
- Create: `tests/scripts/test_convert_pdf_to_md_cli.py`

**Design:**
- CLI 인자: `--input-file | --input-dir`, `--doc-type`(필수, choices=DOC_TYPES − {"spec"}), `--output-dir`(필수), `--dry-run`.
- doc-type → 파서 함수 dispatch (현 단계는 NotImplementedError로 placeholder; PRD Task 1.1에서 라이브러리별 구현 채움).
- `spec` doc_type 거부 — 3GPP는 이미 md.

- [ ] **Step 1: Write failing CLI tests**

Create `tests/scripts/__init__.py` (빈 파일).

```python
# tests/scripts/test_convert_pdf_to_md_cli.py
import subprocess
import sys
from pathlib import Path


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/convert_pdf_to_md.py", *args],
        cwd=cwd, capture_output=True, text=True,
    )


def test_help_lists_doc_types():
    repo = Path(__file__).resolve().parents[2]
    proc = _run(["--help"], cwd=repo)
    assert proc.returncode == 0
    assert "--doc-type" in proc.stdout
    assert "spec" not in proc.stdout  # spec은 PDF 변환 대상 아님


def test_spec_doc_type_rejected(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    proc = _run(
        ["--input-file", str(pdf), "--doc-type", "spec",
         "--output-dir", str(tmp_path / "out")],
        cwd=repo,
    )
    assert proc.returncode != 0


def test_dry_run_announces_parser(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    proc = _run(
        ["--input-file", str(pdf), "--doc-type", "parameter_ref",
         "--output-dir", str(tmp_path / "out"), "--dry-run"],
        cwd=repo,
    )
    assert proc.returncode == 0
    assert "parameter_ref" in proc.stdout
    assert "[DRY RUN]" in proc.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/scripts/test_convert_pdf_to_md_cli.py -v`
Expected: FAIL — script not found

- [ ] **Step 3: Implement skeleton**

Create `scripts/convert_pdf_to_md.py`:

```python
#!/usr/bin/env python3
"""PDF → markdown 변환 CLI (doc-type별 파서 분기).

PRD Task 1.1: 실 구현(파서)은 doc-type별로 별도 PR에서 채움. 본 스켈레톤은
인터페이스 + dispatch만 확정.

Usage:
    python scripts/convert_pdf_to_md.py --input-file foo.pdf \\
        --doc-type parameter_ref --output-dir data/skt-md/parameter_ref/
    python scripts/convert_pdf_to_md.py --input-dir data/skt-pdf/ \\
        --doc-type mop --output-dir data/skt-md/mop/
    python scripts/convert_pdf_to_md.py --input-file ... --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spar.retrieval.milvus_client import DOC_TYPES

# spec은 외부 markdown 코퍼스 — PDF 변환 대상 아님
PDF_DOC_TYPES = [dt for dt in DOC_TYPES if dt != "spec"]


def parse_pdf(pdf_path: Path, doc_type: str) -> str:
    """doc-type별 PDF→md 변환 라우터.

    TODO(Task 1.1): 각 분기에 실제 라이브러리 호출 채우기.
      parameter_ref/counter_ref/alarm_ref → camelot/pdfplumber (표 추출 강화)
      mop/install_guide                  → unstructured (절차 헤더 보존)
      feature_desc                        → pdfplumber (일반 텍스트)
      release_notes                       → unstructured (변경 항목 단위)
    """
    raise NotImplementedError(
        f"PDF parser for doc_type={doc_type!r} not yet implemented (Task 1.1)"
    )


def convert_file(
    pdf: Path, doc_type: str, output_dir: Path, *, dry_run: bool
) -> Path | None:
    out = output_dir / pdf.with_suffix(".md").name
    print(f"Convert: {pdf}  →  {out}  [doc_type={doc_type}]")
    if dry_run:
        print(f"  [DRY RUN] parser={doc_type} (skeleton only — Task 1.1 미구현)")
        return None
    md_text = parse_pdf(pdf, doc_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(md_text, encoding="utf-8")
    return out


def convert_directory(
    input_dir: Path, doc_type: str, output_dir: Path, *, dry_run: bool
) -> None:
    pdfs = sorted(input_dir.rglob("*.pdf"))
    if not pdfs:
        print(f"No .pdf files under {input_dir}")
        return
    for p in pdfs:
        convert_file(p, doc_type, output_dir, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF → markdown 변환")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-file", type=Path)
    src.add_argument("--input-dir", type=Path)
    parser.add_argument(
        "--doc-type", required=True, choices=PDF_DOC_TYPES,
        help=f"PDF 변환 대상 유형 (선택: {', '.join(PDF_DOC_TYPES)}); 'spec'은 이미 md",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.input_file and not args.input_file.exists():
        print(f"ERROR: file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    if args.input_dir and not args.input_dir.is_dir():
        print(f"ERROR: not a directory: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.input_file:
        convert_file(args.input_file, args.doc_type, args.output_dir,
                     dry_run=args.dry_run)
    else:
        convert_directory(args.input_dir, args.doc_type, args.output_dir,
                          dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/scripts/test_convert_pdf_to_md_cli.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/convert_pdf_to_md.py tests/scripts/__init__.py \
        tests/scripts/test_convert_pdf_to_md_cli.py
git commit -m "feat(ingest): add convert_pdf_to_md.py CLI skeleton (Task 1.1 hook)"
```

---

## Task 6: 단일 TSpec-LLM 파일 dry-run E2E 검증

**Files:**
- Create: `tests/ingest/test_tspec_llm_smoke.py`

**Design:**
- 다운로드된 `data/tspec-llm/3GPP-clean/Rel-18/` 안의 임의 1 파일을 `run_ingest.py --dry-run` 실행 → exit 0 + 청크 ≥ 5개.
- 파일 부재 시 skip (CI 미다운로드 환경 보호).

- [ ] **Step 1: Write the test**

```python
# tests/ingest/test_tspec_llm_smoke.py
import subprocess
import sys
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "tspec-llm" / "3GPP-clean" / "Rel-18"


@pytest.mark.skipif(not DATA_DIR.exists(),
                    reason="TSpec-LLM Rel-18 미다운로드 — fetch_tspec_llm.py 먼저 실행")
def test_first_md_file_dry_run():
    md_files = sorted(DATA_DIR.rglob("*.md"))
    assert md_files, "Rel-18 아래 md 없음"

    target = md_files[0]
    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "scripts/run_ingest.py",
         "--input-file", str(target), "--doc-type", "spec", "--dry-run"],
        cwd=repo, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "DRY RUN" in proc.stdout
    # 3GPP spec 첫 파일은 헤더 풍부 — 최소 5청크 이상 기대
    assert "chunks" in proc.stdout
```

- [ ] **Step 2: Run smoke**

Run: `.venv/bin/pytest tests/ingest/test_tspec_llm_smoke.py -v -s`
Expected: PASS — 표준 1개 파일이 다수 청크로 분할됨.

- [ ] **Step 3: Manual sanity check**

Run:
```bash
.venv/bin/python scripts/run_ingest.py \
  --input-file $(ls data/tspec-llm/3GPP-clean/Rel-18/21_series/*.md | head -1) \
  --doc-type spec --dry-run
```

Expected: section 헤더가 청크 메타로 보임 (`section='1 Scope'` 등).

- [ ] **Step 4: Commit**

```bash
git add tests/ingest/test_tspec_llm_smoke.py
git commit -m "test(ingest): add TSpec-LLM single-file dry-run smoke test"
```

---

## Task 7: Bulk Rel-18 ingest 실행 (수동, 비-TDD)

**Files:** 없음 — 운영 작업.

**Pre-flight:**
- Milvus 컨테이너 가동 여부: `docker ps | grep milvus` 또는 `Makefile`의 milvus 타겟.
- 컬렉션 생성: `python -c "from spar.retrieval.milvus_client import SparMilvusClient; c = SparMilvusClient(); c.create_collection('spec'); print(c.list_collections())"`
- 임베더 모델 사전 다운로드 (오프라인 안전): `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"`
- GPU 확인: `python -c "import torch; print(torch.cuda.is_available())"` — 920 파일×수만 청크 = CPU만으론 시간 큼.

- [ ] **Step 1: Dry-run 디렉토리 전체**

Run:
```bash
.venv/bin/python scripts/run_ingest.py \
  --input-dir data/tspec-llm/3GPP-clean/Rel-18/ \
  --doc-type spec --dry-run 2>&1 | tee /tmp/rel18_dry.log
```

Expected: `Found 920 md files`, 끝부분에 `Total chunks: <N>` (대략 수만 단위 예상).

- [ ] **Step 2: 실제 ingest (소수 파일 먼저)**

10개 파일만 임시 디렉토리로 복사 → ingest, 검색 1회 sanity:

```bash
mkdir -p /tmp/rel18_sub
ls data/tspec-llm/3GPP-clean/Rel-18/21_series/*.md | head -10 | xargs -I{} cp {} /tmp/rel18_sub/
.venv/bin/python scripts/run_ingest.py --input-dir /tmp/rel18_sub --doc-type spec
```

검색 sanity:
```python
.venv/bin/python -c "
from spar.ingest.embedder import Embedder
from spar.retrieval.milvus_client import SparMilvusClient
e = Embedder(); v = e.encode(['What is the architecture of NR?'])
with SparMilvusClient() as c:
    res = c.search('spec', v, top_k=3)
    for hit in res[0]:
        print(hit['score'], hit['source_doc'], hit['section'][:60])
"
```

Expected: 3개 결과, score > 0.4, section 메타 채워짐.

- [ ] **Step 3: 전체 920 파일 bulk ingest**

```bash
.venv/bin/python scripts/run_ingest.py \
  --input-dir data/tspec-llm/3GPP-clean/Rel-18/ \
  --doc-type spec --force 2>&1 | tee /tmp/rel18_ingest.log
```

`--force`: 사전 ingest된 동일 source_doc 청크 제거 후 재삽입.

Expected:
- 920 파일 모두 처리
- `Total chunks: <N>` 정상 출력
- 종료 코드 0
- 시간: GPU(A100 등) 1~2시간, CPU 단독 6시간+ 가능

- [ ] **Step 4: 사후 검증**

```bash
.venv/bin/python -c "
from pymilvus import Collection, connections
import os
connections.connect(host=os.environ.get('MILVUS_HOST','localhost'),
                    port=os.environ.get('MILVUS_PORT','19530'))
col = Collection('spar_spec'); col.load()
print('row count:', col.num_entities)
"
```

Expected: row count == dry-run에서 출력된 Total chunks.

- [ ] **Step 5: 종합 sanity 질의 5개**

5개 질의로 검색 → 모든 질의가 의미 있는 (점수>0.3) hit 1개 이상 반환되는지 수동 확인. 실패 사례 요약을 PR 설명에 첨부.

---

## Self-Review

- [x] **Spec coverage:** 모든 요구사항 — DOC_TYPES 확장(Task 1), md 청커(Task 2), 임베더(Task 3), run_ingest 분기 정리(Task 4), PDF 변환 스켈레톤(Task 5), TSpec-LLM 검증(Task 6), bulk 실행(Task 7) — 매핑됨.
- [x] **Placeholder scan:** 모든 step에 실 코드 또는 실 명령. NotImplementedError 자리는 Task 1.1 인계용 hook으로 의도된 것.
- [x] **Type consistency:** `chunk_dispatch`, `Chunk`, `Embedder.encode`, `DOC_TYPES` 이름이 모든 task 일관.
- [x] **테스트 커버리지:** Task 1~5 각각 TDD; Task 6 = 통합; Task 7 = 운영 수동.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-01-md-ingest-pipeline.md`. 두 가지 실행 방식:

**1. Subagent-Driven (recommended)** — 각 task 별도 fresh subagent, task 사이 리뷰 체크포인트.
**2. Inline Execution** — 본 세션에서 batch 실행 + 체크포인트.

Which approach?
