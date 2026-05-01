#!/usr/bin/env python3
"""3GPP .md 문서를 읽어 Codex CLI로 QA 세트를 생성하고 JSONL로 저장.

사용법:
    python scripts/gen_goldset_qa.py --input-file data/tspec-llm/.../29502-i40.md
    python scripts/gen_goldset_qa.py --input-dir data/tspec-llm/3GPP-clean/Rel-18/29_series
    python scripts/gen_goldset_qa.py --input-dir data/tspec-llm/3GPP-clean/Rel-18 --output data/goldsets/my_goldset.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SPAR_ROOT = Path(__file__).parent.parent
DEFAULT_OUT = SPAR_ROOT / "data" / "goldsets" / "retrieval_goldset.jsonl"

_SPEC_FNAME_RE = re.compile(r"^(\d{2})(\d{3})")

QA_COUNTS = {
    "terminology": 3,
    "technology": 3,
    "behavior": 4,
}

PROMPT_TEMPLATE = """\
아래 3GPP 기술 규격 문서의 일부를 읽고, 실제 통신 엔지니어가 업무 중 물어볼 법한 질문-답변 세트를 생성하라.

## 문서 메타데이터
- spec_number: {spec_number}
- source_doc: {source_doc}
- release: {release}

## 질문 관점별 최소 생성 개수
- terminology: {cnt_terminology}개 — 특정 명칭 관점. 용어·약어·개념의 정의와 의미.
  예) "3GPP TS {spec_number}에서 S-NSSAI란 무엇인가?", "TS {spec_number}에서 SMF가 하는 역할은?"
- technology: {cnt_technology}개 — 기술 관점. 프로토콜·아키텍처·메커니즘의 작동 원리.
  예) "TS {spec_number} {release}에서 AMF와 SMF 간 N11 인터페이스는 어떤 방식으로 PDU 세션 컨텍스트를 전달하는가?"
- behavior: {cnt_behavior}개 — 동작 관점. 특정 조건·이벤트·절차에서 네트워크 요소의 구체적 행동.
  예) "TS {spec_number}에서 UE가 handover 중 PDU 세션 재개에 실패하면 AMF는 어떻게 처리하는가?"

## 질문 작성 규칙
- 질문 앞에 반드시 문서 식별자 포함: "3GPP TS {spec_number}" 또는 "TS {spec_number} {release}" 형식
- 절(section) 번호는 질문에 포함하지 말 것 (사용자는 섹션 번호를 모름)
- 아래 표현은 절대 사용 금지: "문서에서", "본 문서", "이 규격에서", "명시된", "기술된", "정의된 바에 따르면"
- section 필드: 답변 근거가 되는 챕터 번호를 반드시 기입 (예: "5.2.1", "6.1.3.2"). 빈 문자열 불가

## 출력 규칙
- 반드시 JSON 배열만 출력 (마크다운 코드블록, 설명 텍스트 없이)
- answer는 문서 내용에만 근거 (hallucination 금지)

## 출력 스키마
[
  {{
    "query": "질문 텍스트",
    "answer": "답변 텍스트",
    "type": "terminology",
    "section": "4.1"
  }}
]

## 문서 내용 ({source_doc}, {spec_number}, {release})
---
{doc_content}
---
"""


def parse_spec_number(filename: str) -> str:
    stem = Path(filename).stem
    m = _SPEC_FNAME_RE.match(stem)
    if not m:
        return ""
    return f"{m.group(1)}.{m.group(2)}"


def extract_release(file_path: Path) -> str:
    for part in file_path.parts:
        if part.startswith("Rel-"):
            return part
    return "Rel-18"


def build_prompt(file_path: Path, doc_content: str) -> str:
    spec_number = parse_spec_number(file_path.name)
    release = extract_release(file_path)
    return PROMPT_TEMPLATE.format(
        spec_number=spec_number or "(unknown)",
        source_doc=file_path.name,
        release=release,
        cnt_terminology=QA_COUNTS["terminology"],
        cnt_technology=QA_COUNTS["technology"],
        cnt_behavior=QA_COUNTS["behavior"],
        doc_content=doc_content,
    )


def call_codex(prompt: str, max_lines: int) -> str:
    """Codex exec headless. 프롬프트를 tmpfile로 주입해 ARG_MAX 회피."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", encoding="utf-8", delete=False
    ) as pf:
        pf.write(prompt)
        prompt_path = pf.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8", delete=False
    ) as of:
        output_path = of.name

    try:
        with open(prompt_path, encoding="utf-8") as stdin_f:
            result = subprocess.run(
                [
                    "codex", "exec", "-",
                    "--dangerously-bypass-approvals-and-sandbox",
                    "--ephemeral",
                    "--color", "never",
                    "--output-last-message", output_path,
                ],
                stdin=stdin_f,
                capture_output=True,
                text=True,
                timeout=300,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"codex exec 실패 (exit {result.returncode}):\n{result.stderr[:500]}"
            )
        return Path(output_path).read_text(encoding="utf-8")
    finally:
        Path(prompt_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def parse_qa_output(raw: str, source_doc: str, spec_number: str, release: str, start_id: int) -> list[dict]:
    raw = raw.strip()
    # JSON 배열 추출 (코드블록 감쌀 경우 대비)
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        raise ValueError(f"JSON 배열 없음. raw output:\n{raw[:300]}")
    items = json.loads(m.group(0))
    results = []
    for i, item in enumerate(items):
        section = item.get("section", "")
        if not section:
            print(f"  WARN: section 빈값 (query: {item.get('query', '')[:40]})", file=sys.stderr)
        results.append({
            "query_id": f"Q{start_id + i:04d}",
            "query": item["query"],
            "answer": item["answer"],
            "type": item["type"],
            "section": section,
            "source_doc": source_doc,
            "spec_number": spec_number,
            "release": release,
        })
    return results


def process_file(
    file_path: Path,
    out_file,
    start_id: int,
    max_lines: int,
    dry_run: bool,
) -> int:
    print(f"  처리: {file_path.name}", end="", flush=True)
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
        doc_content = "".join(lines[:max_lines])
    except (UnicodeDecodeError, OSError) as e:
        print(f"  WARN: 읽기 실패 ({e})", file=sys.stderr)
        return 0

    prompt = build_prompt(file_path, doc_content)

    if dry_run:
        print(" [DRY RUN - codex 미호출]")
        return 0

    t0 = time.time()
    try:
        raw = call_codex(prompt, max_lines)
    except (subprocess.TimeoutExpired, RuntimeError) as e:
        print(f"\n  ERROR: {e}", file=sys.stderr)
        return 0

    spec_number = parse_spec_number(file_path.name)
    release = extract_release(file_path)

    try:
        items = parse_qa_output(raw, file_path.name, spec_number, release, start_id)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"\n  ERROR 파싱 실패: {e}", file=sys.stderr)
        return 0

    for item in items:
        out_file.write(json.dumps(item, ensure_ascii=False) + "\n")
    out_file.flush()

    elapsed = time.time() - t0
    print(f" → {len(items)}개 ({elapsed:.1f}s)")
    return len(items)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3GPP .md 문서 → Codex QA 생성 → JSONL 저장"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-file", type=Path, metavar="FILE")
    src.add_argument("--input-dir", type=Path, metavar="DIR")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUT, metavar="FILE",
        help=f"출력 JSONL 경로 (기본: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--max-lines", type=int, default=1500, metavar="N",
        help="파일당 최대 읽을 줄 수 (기본: 1500)",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="출력 파일에 이어쓰기 (기본: 덮어쓰기)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Codex 미호출, 구조만 확인",
    )
    args = parser.parse_args()

    if args.input_file and not args.input_file.exists():
        print(f"ERROR: 파일 없음: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    if args.input_dir and not args.input_dir.is_dir():
        print(f"ERROR: 디렉토리 없음: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.input_file:
        md_files = [args.input_file]
    else:
        md_files = sorted(args.input_dir.rglob("*.md"))
        if not md_files:
            print(f"ERROR: .md 없음: {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"{len(md_files)}개 파일 발견: {args.input_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    if args.dry_run:
        print("[DRY RUN] codex 미호출\n")

    total = 0
    start_id = 1

    if args.append and args.output.exists():
        existing = sum(1 for _ in args.output.open(encoding="utf-8"))
        start_id = existing + 1
        print(f"기존 {existing}개 항목에 이어쓰기. query_id Q{start_id:04d}부터 시작\n")

    with open(args.output, mode, encoding="utf-8") as out_f:
        for md in md_files:
            count = process_file(md, out_f, start_id, args.max_lines, args.dry_run)
            total += count
            start_id += count

    print(f"\n완료: {total}개 QA → {args.output}")


if __name__ == "__main__":
    main()
