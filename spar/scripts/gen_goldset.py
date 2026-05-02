#!/usr/bin/env python3
"""3GPP .md 문서 → 단일 통합 goldset JSONL 생성.

QA 필드(query, answer, type, section, ...)와 router 필드(expected_route, needs_decomposition)를
한 레코드에 함께 저장한다. retrieval 평가와 router 평가 모두 이 파일 하나로 수행 가능.

사용법:
    python scripts/gen_goldset.py --input-file data/tspec-llm/.../29502-i40.md
    python scripts/gen_goldset.py --input-dir data/tspec-llm/3GPP-clean/Rel-18/29_series
    python scripts/gen_goldset.py --input-dir data/tspec-llm/... --output data/goldsets/goldset.jsonl

실행 순서: codex exec → (실패 시) gemini → (gemini 실패 시) 지수 백오프 재시도 (최대 MAX_ATTEMPTS회)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SPAR_ROOT = Path(__file__).parent.parent
DEFAULT_OUT = SPAR_ROOT / "data" / "goldsets" / "goldset.jsonl"

_SPEC_FNAME_RE = re.compile(r"^(\d{2})(\d{3})")

QA_COUNTS = {
    "terminology": 2,
    "technology": 2,
    "behavior": 2,
    "diagnostic": 2,
    "procedural": 2,
    "comparative": 1,
    "lookup": 1,
}

# QA type → router expected_route 매핑
_TYPE_TO_ROUTE: dict[str, str] = {
    "terminology": "definition_explain",
    "technology": "default_rag",
    "behavior": "default_rag",
    "diagnostic": "diagnostic",
    "procedural": "procedural",
    "comparative": "comparative",
    "lookup": "structured_lookup",
}
_TYPE_TO_DECOMPOSE: dict[str, bool] = {
    "terminology": False,
    "technology": False,
    "behavior": False,
    "diagnostic": True,
    "procedural": False,
    "comparative": False,
    "lookup": False,
}

MAX_ATTEMPTS = 10  # gemini 재시도 상한 (무한루프 방지)

PROMPT_TEMPLATE = """\
아래 3GPP 기술 규격 문서의 일부를 읽고, 실제 통신 엔지니어가 업무 중 물어볼 법한 질문-답변 세트를 생성하라.

## 문서 메타데이터
- spec_number: {spec_number}
- source_doc: {source_doc}
- release: {release}

## 질문 관점별 최소 생성 개수
- terminology: {cnt_terminology}개 — 용어/약어/개념 정의 관점. 특정 용어가 무엇인지, 어떤 의미인지.
  예) "3GPP TS {spec_number}에서 S-NSSAI란 무엇인가?", "TS {spec_number}에서 SMF가 하는 역할은?"
- technology: {cnt_technology}개 — 기술 원리 관점. 프로토콜·아키텍처·메커니즘의 작동 방식.
  예) "TS {spec_number} {release}에서 AMF와 SMF 간 N11 인터페이스는 어떤 방식으로 PDU 세션 컨텍스트를 전달하는가?"
- behavior: {cnt_behavior}개 — 동작 관점. 특정 조건·이벤트에서 네트워크 요소의 구체적 행동.
  예) "TS {spec_number}에서 UE가 handover 중 PDU 세션 재개에 실패하면 AMF는 어떻게 처리하는가?"
- diagnostic: {cnt_diagnostic}개 — 장애/문제 진단 관점. 특정 오류·실패 상황의 원인과 해결.
  예) "TS {spec_number}에서 PDU Session Establishment가 실패하는 주요 원인과 각 원인별 처리 절차는?"
- procedural: {cnt_procedural}개 — 절차/단계 관점. 특정 기능을 활성화하거나 수행하는 순서.
  예) "TS {spec_number} {release}에서 UE가 Network Slice를 선택하는 절차는 어떤 순서로 진행되는가?"
- comparative: {cnt_comparative}개 — 비교 관점. 두 개 이상의 메커니즘·옵션·파라미터 차이.
  예) "TS {spec_number}에서 RRC_IDLE과 RRC_INACTIVE 상태의 차이점은 무엇인가?"
- lookup: {cnt_lookup}개 — 특정 값/파라미터 조회 관점. 타이머 값, 임계값, 식별자 등 구체적 수치.
  예) "TS {spec_number}에서 T3512 타이머의 기본값과 적용 조건은?"

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
        cnt_diagnostic=QA_COUNTS["diagnostic"],
        cnt_procedural=QA_COUNTS["procedural"],
        cnt_comparative=QA_COUNTS["comparative"],
        cnt_lookup=QA_COUNTS["lookup"],
        doc_content=doc_content,
    )


def call_codex(prompt: str) -> str:
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


def call_gemini(prompt: str) -> str:
    if shutil.which("gemini") is None:
        raise RuntimeError("gemini CLI PATH에 없음")

    result = subprocess.run(
        ["gemini", "--yolo", "--output-format", "json", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gemini 실패 (exit {result.returncode}):\n{result.stderr[:500]}"
        )
    raw = result.stdout.strip()
    if not raw:
        raise RuntimeError("gemini 빈 출력")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    for key in ("response", "text", "output"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, str) and value:
            return value

    raise RuntimeError(f"gemini 응답 필드 없음: {str(payload)[:200]}")


def call_with_fallback(prompt: str) -> str:
    """codex → gemini → gemini 지수 백오프 재시도 (최대 MAX_ATTEMPTS회)."""
    try:
        return call_codex(prompt)
    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        print(f"\n  WARN: codex 실패, gemini로 전환 — {exc}", file=sys.stderr)

    delay = 2.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = call_gemini(prompt)
            if attempt > 1:
                print(f"\n  gemini {attempt - 1}회 재시도 후 성공", file=sys.stderr)
            return result
        except (subprocess.TimeoutExpired, RuntimeError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"gemini {MAX_ATTEMPTS}회 모두 실패 — 마지막 오류: {exc}"
                ) from exc
            print(
                f"\n  WARN: gemini 실패 (시도 {attempt}/{MAX_ATTEMPTS}), {delay:.0f}s 대기 — {exc}",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay *= 2

    raise RuntimeError("BUG: unreachable")


def _derive_router_fields(qa_type: str) -> tuple[str, bool]:
    route = _TYPE_TO_ROUTE.get(qa_type, "default_rag")
    needs_decomp = _TYPE_TO_DECOMPOSE.get(qa_type, False)
    return route, needs_decomp


def parse_goldset_output(
    raw: str,
    source_doc: str,
    spec_number: str,
    release: str,
    start_id: int,
) -> list[dict]:
    raw = raw.strip()
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        raise ValueError(f"JSON 배열 없음. raw output:\n{raw[:300]}")
    items = json.loads(m.group(0))
    results = []
    for i, item in enumerate(items):
        qa_type = item.get("type", "terminology")
        section = item.get("section", "")
        if not section:
            print(f"  WARN: section 빈값 (query: {item.get('query', '')[:40]})", file=sys.stderr)
        expected_route, needs_decomposition = _derive_router_fields(qa_type)
        results.append({
            "query_id": f"Q{start_id + i:04d}",
            "query": item["query"],
            "answer": item["answer"],
            "type": qa_type,
            "section": section,
            "source_doc": source_doc,
            "spec_number": spec_number,
            "release": release,
            "expected_route": expected_route,
            "needs_decomposition": needs_decomposition,
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
        print(" [DRY RUN - codex/gemini 미호출]")
        return 0

    t0 = time.time()
    raw = call_with_fallback(prompt)

    spec_number = parse_spec_number(file_path.name)
    release = extract_release(file_path)

    try:
        items = parse_goldset_output(raw, file_path.name, spec_number, release, start_id)
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
        description="3GPP .md 문서 → QA + router 통합 goldset JSONL 생성"
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
        help="Codex/gemini 미호출, 구조만 확인",
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
        print("[DRY RUN] codex/gemini 미호출\n")

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

    print(f"\n완료: {total}개 → {args.output}")


if __name__ == "__main__":
    main()
