#!/usr/bin/env python3
"""TSpec-LLM 데이터셋 다운로드 (HuggingFace Hub).

대상: rasoul-nikbakht/TSpec-LLM
  - 3GPP 전 release(.docx → markdown 변환본)
  - gated dataset (CC-BY-NC-4.0): HF 계정 + 접근 승인 + HF_TOKEN 필요
  - 규모: 30,976 files (Rel-8 ~ Rel-19), repo 구조: 3GPP-clean/Rel-XX/NN_series/*.{docx,md}

Usage:
    python scripts/fetch_tspec_llm.py                          # md만, 전 release
    python scripts/fetch_tspec_llm.py --include-docx           # docx도 포함
    python scripts/fetch_tspec_llm.py --release Rel-18
    python scripts/fetch_tspec_llm.py --release Rel-18 --release Rel-17
    HF_TOKEN=hf_xxx python scripts/fetch_tspec_llm.py --token-from-env
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "rasoul-nikbakht/TSpec-LLM"
# spar/scripts/ → spar/data/tspec-llm (CWD 무관)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "data" / "tspec-llm"
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(path: Path) -> None:
    """경량 .env 파서 — 외부 의존성 없이 KEY=VALUE 만 처리."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def download(
    local_dir: Path,
    *,
    token: str | None,
    releases: list[str] | None,
    revision: str | None,
    include_docx: bool,
) -> Path:
    """TSpec-LLM 스냅샷을 local_dir 로 다운로드.

    Args:
        local_dir: 다운로드 대상 경로
        token: HF 토큰 (None이면 캐시된 로그인 사용)
        releases: 특정 release 만 받을 때 prefix 목록 (예: ["Rel-18"])
        revision: dataset revision/branch (None이면 main)
        include_docx: True면 .docx 도 포함, False면 .md 만 받음

    Returns:
        실제 다운로드 경로
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "ERROR: huggingface_hub 미설치. 설치:\n"
            "  pip install 'huggingface_hub[cli]>=0.24'",
            file=sys.stderr,
        )
        sys.exit(1)

    local_dir.mkdir(parents=True, exist_ok=True)

    # repo 구조: 3GPP-clean/Rel-XX/NN_series/*.md
    rel_globs = [f"3GPP-clean/{r}" for r in releases] if releases else ["3GPP-clean"]
    exts = ["md", "docx"] if include_docx else ["md"]
    allow_patterns = [f"{rg}/**/*.{ext}" for rg in rel_globs for ext in exts]
    print(f"Filter: {allow_patterns}")

    print(f"Downloading {REPO_ID} → {local_dir}")
    if revision:
        print(f"  revision: {revision}")

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=token,
        allow_patterns=allow_patterns,
        revision=revision,
    )
    return Path(path)


def summarize(root: Path) -> None:
    """다운로드 결과 요약 출력 (md 파일 수, 총 용량)."""
    md_files = list(root.rglob("*.md"))
    total_bytes = sum(f.stat().st_size for f in md_files)
    print("\nSummary:")
    print(f"  path:       {root}")
    print(f"  md files:   {len(md_files):,}")
    print(f"  total size: {total_bytes / (1024**3):.2f} GiB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TSpec-LLM (3GPP 전 release markdown) 다운로드"
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR,
        metavar="DIR",
        help=f"다운로드 경로 (기본: {DEFAULT_LOCAL_DIR})",
    )
    parser.add_argument(
        "--release",
        action="append",
        metavar="REL",
        help="특정 release prefix만 받기 (예: --release Rel-18). 중복 가능.",
    )
    parser.add_argument(
        "--revision",
        metavar="REV",
        help="dataset revision/branch (기본: main)",
    )
    parser.add_argument(
        "--token-from-env",
        action="store_true",
        help="HF_TOKEN 환경 변수에서 토큰 로드",
    )
    parser.add_argument(
        "--include-docx",
        action="store_true",
        help="원본 .docx도 함께 다운로드 (기본: .md만)",
    )

    args = parser.parse_args()

    # .env 자동 로드 (이미 export 된 값은 덮어쓰지 않음)
    load_env_file(ENV_FILE)

    # HF_TOKEN 자동 사용 — 환경/.env에 있으면 명시 플래그 없어도 적용
    token: str | None = os.environ.get("HF_TOKEN") or None
    if args.token_from_env and not token:
        print("ERROR: HF_TOKEN 비어있음 (환경 변수 또는 .env 확인)", file=sys.stderr)
        sys.exit(1)

    try:
        path = download(
            args.local_dir,
            token=token,
            releases=args.release,
            revision=args.revision,
            include_docx=args.include_docx,
        )
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            print(
                "ERROR: gated dataset 접근 거부.\n"
                "  1) https://hf.co/datasets/rasoul-nikbakht/TSpec-LLM 에서 접근 승인 확인\n"
                "  2) `huggingface-cli login` 또는 HF_TOKEN 설정 후 재시도",
                file=sys.stderr,
            )
        else:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    summarize(path)


if __name__ == "__main__":
    main()
