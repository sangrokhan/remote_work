#!/usr/bin/env python3
"""3GPP md 파일의 Abbreviations 섹션에서 약어 사전 추출.

3GPP 표준 문서는 보통 `## 3.x Abbreviations` 섹션 아래
`<ACRONYM> <Expansion>` 형식으로 약어를 정의한다. 본 스크립트는 그 섹션을
파싱해 `dictionary/acronyms.json` 형식의 사전을 생성한다.

Output schema (abbrev-mapping 플랜과 호환):
    {
      "global": {
        "<ACRONYM>": {"expansion": "<Expansion>", "variants": []}
      }
    }

Usage:
    python scripts/extract_acronyms.py --input-file foo.md
    python scripts/extract_acronyms.py --input-file foo.md --output dictionary/acronyms.json
    python scripts/extract_acronyms.py --input-file foo.md --merge dictionary/acronyms.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "dictionary" / "acronyms.json"

# Abbreviations 섹션 헤더 — pandoc-md 두 형태 지원
#   1) Setext-style: `3.3 Abbreviations\n---------\n`
#   2) ATX-style:    `## 3.3 Abbreviations\n`
_SECTION_RE = re.compile(
    r"(?:^#{1,6}\s+[\d\.]*\s*[Aa]bbreviations\s*$"     # ATX
    r"|^[\d\.]*\s*[Aa]bbreviations\s*\n[-=]{3,}\s*$)", # Setext
    re.MULTILINE,
)

# 다음 섹션 헤더 (종료 경계)
_NEXT_SECTION_RE = re.compile(
    r"(?:^#{1,6}\s+\S"           # ATX 헤더
    r"|^\S.*\n[-=]{3,}\s*$)",     # Setext 헤더
    re.MULTILINE,
)

# 약어 항목 — 줄 시작 (선택적 `> ` 인용), 대문자/숫자 시작, 이후 공백 + 본문
# 약어: 2~20자, 대문자/숫자/`-`/`_`/이스케이프된 `\_`/소문자(예: gNB) 허용
_ENTRY_RE = re.compile(
    r"^\s*>?\s*"                           # 선택적 blockquote prefix
    r"([A-Za-z][A-Za-z0-9_\\\-]{1,19})"    # 약어 캡처
    r"[ \t]+"                              # 공백 (탭/스페이스)
    r"(.+?)\s*$",                          # expansion (행 끝)
)


def _normalize_acronym(raw: str) -> str:
    """파서 산출물의 escape(`\\_`)를 정규화."""
    return raw.replace("\\_", "_").replace("\\-", "-").strip()


def _is_likely_acronym(token: str) -> bool:
    """약어 형태 휴리스틱 — 일반 단어 제외."""
    if len(token) < 2 or len(token) > 20:
        return False
    # 적어도 1개 대문자 필요
    if not any(c.isupper() for c in token):
        return False
    # 모두 소문자만이면 일반 단어
    if token.islower():
        return False
    # 흔한 영어 단어 (For, The, This 등) 차단 — 첫글자만 대문자 + 모두 알파벳
    if token[0].isupper() and token[1:].islower() and token.isalpha():
        return False
    return True


def find_abbreviations_section(text: str) -> str | None:
    """Abbreviations 섹션 본문 추출 (헤더 다음 ~ 다음 헤더 직전)."""
    m = _SECTION_RE.search(text)
    if not m:
        return None
    body_start = m.end()
    # 다음 헤더 찾기 (현재 섹션 헤더 이후)
    next_m = _NEXT_SECTION_RE.search(text, pos=body_start)
    body_end = next_m.start() if next_m else len(text)
    return text[body_start:body_end]


def parse_entries(section_body: str) -> dict[str, str]:
    """섹션 본문에서 약어 → 확장 매핑 추출.

    멀티라인 entry는 후속 들여쓰기 있는 줄을 합쳐 처리.
    """
    entries: dict[str, str] = {}
    current_acronym: str | None = None
    current_expansion_parts: list[str] = []

    def _commit() -> None:
        nonlocal current_acronym, current_expansion_parts
        if current_acronym and current_expansion_parts:
            full = " ".join(current_expansion_parts).strip()
            # 첫 등장이 우선 (중복 시 무시)
            entries.setdefault(current_acronym, full)
        current_acronym = None
        current_expansion_parts = []

    for raw_line in section_body.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            _commit()
            continue
        m = _ENTRY_RE.match(line)
        if m:
            candidate = _normalize_acronym(m.group(1))
            if _is_likely_acronym(candidate):
                _commit()
                current_acronym = candidate
                current_expansion_parts = [m.group(2).strip()]
                continue
        # 들여쓰기/연속 줄 — 현재 entry expansion에 append
        if current_acronym and (raw_line.startswith((" ", "\t")) or line.strip()):
            current_expansion_parts.append(line.strip())

    _commit()
    return entries


def to_dict_schema(entries: dict[str, str]) -> dict[str, Any]:
    """abbrev-mapping plan 스키마({"global": {ACRO: {expansion, variants}}})로 변환."""
    return {
        "global": {
            acro: {"expansion": exp, "variants": []}
            for acro, exp in sorted(entries.items())
        }
    }


def merge_into(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """기존 사전에 신규 항목 병합 — 기존 값 우선 (수동 큐레이션 보호)."""
    g_existing = existing.setdefault("global", {})
    for acro, payload in new.get("global", {}).items():
        if acro not in g_existing:
            g_existing[acro] = payload
    return existing


def main() -> None:
    ap = argparse.ArgumentParser(description="3GPP md → acronyms.json 추출")
    ap.add_argument("--input-file", type=Path, required=True, metavar="MD")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"출력 경로 (기본: {DEFAULT_OUTPUT})")
    ap.add_argument("--merge", type=Path, metavar="JSON",
                    help="기존 사전과 병합 (기존 값 우선)")
    ap.add_argument("--dry-run", action="store_true",
                    help="파일 쓰기 없이 stdout으로만 출력")
    args = ap.parse_args()

    if not args.input_file.exists():
        print(f"ERROR: {args.input_file} 없음", file=sys.stderr)
        sys.exit(1)

    text = args.input_file.read_text(encoding="utf-8")
    section = find_abbreviations_section(text)
    if section is None:
        print(f"ERROR: 'Abbreviations' 섹션 미발견 in {args.input_file}", file=sys.stderr)
        sys.exit(2)

    entries = parse_entries(section)
    print(f"추출: {len(entries)} entries from {args.input_file.name}")
    payload = to_dict_schema(entries)

    if args.merge and args.merge.exists():
        existing = json.loads(args.merge.read_text(encoding="utf-8"))
        before = len(existing.get("global", {}))
        merged = merge_into(existing, payload)
        after = len(merged.get("global", {}))
        print(f"merge: {before} → {after} entries (added {after - before})")
        payload = merged

    out_text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    if args.dry_run:
        print("--- DRY RUN ---")
        print(out_text[:1500])
        print(f"... (total {len(out_text)} chars)")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(out_text + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
