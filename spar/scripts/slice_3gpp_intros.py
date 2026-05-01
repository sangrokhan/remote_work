#!/usr/bin/env python3
"""각 3GPP .md 파일의 앞 1000줄을 /tmp/3gpp_intros/<series>/<filename>으로 복사."""

from __future__ import annotations

import re
import sys
from pathlib import Path

SPAR_ROOT = Path(__file__).parent.parent
DATA_DIR = SPAR_ROOT / "data" / "tspec-llm" / "3GPP-clean" / "Rel-18"
OUT_DIR = Path("/tmp/3gpp_intros")
LINE_LIMIT = 1000

_SPEC_FNAME_RE = re.compile(r"^(\d{2})(\d{3})")


def parse_spec_number(filename: str) -> str:
    """'29502-i40.md' → '29.502'. 매칭 실패 시 ''."""
    stem = Path(filename).stem
    m = _SPEC_FNAME_RE.match(stem)
    if not m:
        return ""
    return f"{m.group(1)}.{m.group(2)}"


def slice_file(src: Path, dst: Path, line_limit: int = LINE_LIMIT) -> None:
    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    dst.write_text("".join(lines[:line_limit]), encoding="utf-8")


def main() -> None:
    md_files = sorted(DATA_DIR.rglob("*.md"))
    if not md_files:
        print(f"ERROR: .md 파일 없음: {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    count = 0
    skipped = 0
    for src in md_files:
        series = src.parent.name
        dst_dir = OUT_DIR / series
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        slice_file(src, dst)
        spec_num = parse_spec_number(src.name)
        label = spec_num if spec_num else "(unknown spec_number)"
        print(f"  {src.name} → {dst}  [{label}]")
        if spec_num:
            count += 1
        else:
            skipped += 1

    print(f"\n완료: {count}개 파싱 성공, {skipped}개 spec_number 파싱 실패 → {OUT_DIR}")


if __name__ == "__main__":
    main()
