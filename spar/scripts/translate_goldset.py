#!/usr/bin/env python3
"""goldset.jsonl의 한국어 query/answer를 영어로 번역.

사용법:
    python scripts/translate_goldset.py
    python scripts/translate_goldset.py --input data/goldsets/goldset.jsonl --output data/goldsets/goldset_en.jsonl
    python scripts/translate_goldset.py --batch-size 5 --limit 100 --dry-run

실행 순서: codex exec → (실패 시) gemini → (gemini 실패 시) 지수 백오프 재시도 (최대 MAX_ATTEMPTS회)
번역된 항목은 배치 완료 즉시 파일에 기록 (재시작 시 이어받기 가능).
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
DEFAULT_INPUT = SPAR_ROOT / "data" / "goldsets" / "goldset.jsonl"
DEFAULT_OUTPUT = SPAR_ROOT / "data" / "goldsets" / "goldset_en.jsonl"

MAX_ATTEMPTS = 5
DEFAULT_BATCH = 10

_TERM_RULES = """\
Rules:
- Keep ALL 3GPP spec numbers unchanged (e.g., TS 21.900, TS 22.011, Rel-18, Rel-17)
- Keep ALL technical acronyms and abbreviations unchanged (e.g., UE, AMF, SMF, SUPI, MCPTT, NR, LTE, QoS, PDU, PLMN, NSSAI, MPS, CR, TSG)
- Keep ALL MO names, counter IDs, alarm codes, parameter names unchanged
- Keep "3GPP TS (unknown)" as-is
- Translate ONLY the Korean natural language parts to natural English"""

QUERY_PROMPT_TEMPLATE = """\
Translate each Korean 3GPP question to English.
{rules}
- Preserve question structure and intent (procedural→"In what order/procedure", \
diagnostic→"What causes/Why does", comparative→"What is the difference", \
lookup→"What is the value/How many")

Output EXACTLY {n} lines — one translation per line, same order, no numbering, no extra text.

{numbered_items}"""

ANSWER_PROMPT_TEMPLATE = """\
Translate each Korean 3GPP answer to English.
{rules}
- Keep technical explanations accurate and complete

Output EXACTLY {n} lines — one translation per line, same order, no numbering, no extra text.

{numbered_items}"""


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


def _numbered(texts: list[str]) -> str:
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))


def _parse_lines(raw: str, expected: int, field: str) -> list[str]:
    """Strip optional 'N. ' prefix from each line, return exactly expected lines."""
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    # strip leading "N." or "N)" numbering if LLM added it anyway
    cleaned = [re.sub(r"^\d+[.)]\s*", "", l) for l in lines]
    if len(cleaned) != expected:
        raise ValueError(
            f"{field} 줄 수 불일치: 요청 {expected}개, 반환 {len(cleaned)}개\n"
            f"raw:\n{raw[:300]}"
        )
    return cleaned


def build_query_prompt(batch: list[dict]) -> str:
    return QUERY_PROMPT_TEMPLATE.format(
        rules=_TERM_RULES,
        n=len(batch),
        numbered_items=_numbered([item["query"] for item in batch]),
    )


def build_answer_prompt(batch: list[dict]) -> str:
    return ANSWER_PROMPT_TEMPLATE.format(
        rules=_TERM_RULES,
        n=len(batch),
        numbered_items=_numbered([item["answer"] for item in batch]),
    )


def translate_batch(batch: list[dict]) -> list[dict]:
    query_raw = call_with_fallback(build_query_prompt(batch))
    answer_raw = call_with_fallback(build_answer_prompt(batch))

    queries_en = _parse_lines(query_raw, len(batch), "query")
    answers_en = _parse_lines(answer_raw, len(batch), "answer")

    results = []
    for item, q_en, a_en in zip(batch, queries_en, answers_en):
        merged = dict(item)
        merged["query"] = q_en
        merged["answer"] = a_en
        results.append(merged)
    return results


def load_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["query_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="goldset.jsonl 한→영 번역")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH, metavar="N",
        help=f"LLM 호출당 번역 항목 수 (기본: {DEFAULT_BATCH})",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="처리할 최대 항목 수 (미지정 시 전체)",
    )
    parser.add_argument("--dry-run", action="store_true", help="LLM 미호출, 구조 확인만")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: goldset 없음: {args.input}", file=sys.stderr)
        sys.exit(1)

    all_items = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_items.append(json.loads(line))

    print(f"goldset {len(all_items)}개 로드: {args.input}")

    done_ids = load_done_ids(args.output)
    if done_ids:
        print(f"기존 번역 {len(done_ids)}개 발견 → 이어받기")

    pending = [item for item in all_items if item["query_id"] not in done_ids]
    if args.limit:
        pending = pending[: args.limit]

    if not pending:
        print("번역할 항목 없음.")
        return

    print(f"번역 대상: {len(pending)}개 (배치 크기: {args.batch_size})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_done = len(done_ids)

    with args.output.open("a", encoding="utf-8") as out_f:
        batches = [
            pending[i : i + args.batch_size]
            for i in range(0, len(pending), args.batch_size)
        ]
        total_batches = len(batches)

        for batch_idx, batch in enumerate(batches, start=1):
            ids_str = f"{batch[0]['query_id']}~{batch[-1]['query_id']}"
            print(
                f"  배치 {batch_idx}/{total_batches} ({ids_str}) ...",
                end="",
                flush=True,
            )

            if args.dry_run:
                print(" [DRY RUN]")
                continue

            t0 = time.time()

            try:
                translated = translate_batch(batch)
            except (RuntimeError, ValueError) as exc:
                print(f"\n  ERROR 배치 {batch_idx} 실패, 스킵: {exc}", file=sys.stderr)
                continue

            for item in translated:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            out_f.flush()

            total_done += len(translated)
            elapsed = time.time() - t0
            print(f" {len(translated)}개 완료 ({elapsed:.1f}s) 누계={total_done}")

    if not args.dry_run:
        print(f"\n완료: {total_done}개 → {args.output}")


if __name__ == "__main__":
    main()
