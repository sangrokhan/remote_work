# Abbreviation Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 파싱 직후 RAN 도메인 약어를 병기 확장하고, 쿼리 시 역방향 조회로 약어·전체어 양방향 검색을 보장한다.

**Architecture:** `dictionary/acronyms.json` 단일 사전을 인제스트·쿼리 양쪽에서 공유한다. global 약어는 regex 치환, conflict 약어는 vLLM OpenAI-compatible API로 closed-set 분류한다. 파이프라인 위치: `parse → map_abbreviations → chunk → embed → insert`.

**Tech Stack:** Python 3.12, regex (stdlib), openai>=1.40 (vLLM OpenAI-compatible), pytest

---

## File Map

| 작업 | 파일 |
|------|------|
| Create | `dictionary/acronyms.json` |
| Create | `src/spar/preprocessing/__init__.py` |
| Create | `src/spar/preprocessing/abbrev_mapper.py` |
| Create | `tests/preprocessing/__init__.py` |
| Create | `tests/preprocessing/test_abbrev_mapper.py` |
| Modify | `scripts/run_ingest.py` |

---

## Task 1: 약어 사전 생성

**Files:**
- Create: `dictionary/acronyms.json`

- [ ] **Step 1: `dictionary/` 디렉토리 생성**

```bash
mkdir -p dictionary
```

- [ ] **Step 2: `dictionary/acronyms.json` 작성**

```json
{
  "global": {
    "HO": {
      "expansion": "Handover",
      "variants": ["Hand-Over"]
    },
    "TTT": {
      "expansion": "Time-To-Trigger",
      "variants": []
    },
    "RACH": {
      "expansion": "Random Access Channel",
      "variants": []
    },
    "BWP": {
      "expansion": "Bandwidth Part",
      "variants": []
    },
    "UE": {
      "expansion": "User Equipment",
      "variants": []
    },
    "gNB": {
      "expansion": "Next Generation NodeB",
      "variants": []
    },
    "RRC": {
      "expansion": "Radio Resource Control",
      "variants": []
    },
    "PDCP": {
      "expansion": "Packet Data Convergence Protocol",
      "variants": []
    },
    "RLF": {
      "expansion": "Radio Link Failure",
      "variants": []
    },
    "MRO": {
      "expansion": "Mobility Robustness Optimization",
      "variants": []
    }
  },
  "conflicts": {
    "CA": {
      "candidates": ["Carrier Aggregation", "Cell Activation"],
      "variants": []
    },
    "CR": {
      "candidates": ["Change Request", "Cell Range"],
      "variants": []
    }
  }
}
```

- [ ] **Step 3: 커밋**

```bash
git add dictionary/acronyms.json
git commit -m "feat(abbrev): add initial RAN acronym dictionary"
```

---

## Task 2: 패키지 골격 + global 매핑 구현

**Files:**
- Create: `src/spar/preprocessing/__init__.py`
- Create: `src/spar/preprocessing/abbrev_mapper.py`
- Create: `tests/preprocessing/__init__.py`
- Create: `tests/preprocessing/test_abbrev_mapper.py`

- [ ] **Step 1: 테스트 먼저 작성 (failing)**

`tests/preprocessing/__init__.py` — 빈 파일:
```python
```

`tests/preprocessing/test_abbrev_mapper.py`:
```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from spar.preprocessing.abbrev_mapper import (
    build_reverse_index,
    load_acronyms,
    map_abbreviations,
)

SAMPLE_ACRONYMS = {
    "global": {
        "HO": {"expansion": "Handover", "variants": ["Hand-Over"]},
        "TTT": {"expansion": "Time-To-Trigger", "variants": []},
        "RACH": {"expansion": "Random Access Channel", "variants": []},
    },
    "conflicts": {
        "CA": {"candidates": ["Carrier Aggregation", "Cell Activation"], "variants": []},
    },
}


class TestLoadAcronyms:
    def test_loads_json_from_path(self, tmp_path: Path) -> None:
        f = tmp_path / "acronyms.json"
        f.write_text(json.dumps(SAMPLE_ACRONYMS), encoding="utf-8")
        result = load_acronyms(f)
        assert result["global"]["HO"]["expansion"] == "Handover"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_acronyms(tmp_path / "nonexistent.json")


class TestApplyGlobal:
    def test_expands_abbreviation_with_annotation(self) -> None:
        text = "HO is triggered when TTT expires."
        result = map_abbreviations(text, SAMPLE_ACRONYMS)
        assert "HO(Handover)" in result
        assert "TTT(Time-To-Trigger)" in result

    def test_expands_variant(self) -> None:
        text = "Hand-Over failure occurred."
        result = map_abbreviations(text, SAMPLE_ACRONYMS)
        assert "Hand-Over(Handover)" in result

    def test_word_boundary_not_partial_match(self) -> None:
        text = "RACHAEL sent RACH preamble."
        result = map_abbreviations(text, SAMPLE_ACRONYMS)
        assert "RACHAEL" in result
        assert "RACHAEL(Random Access Channel)" not in result
        assert "RACH(Random Access Channel)" in result

    def test_no_llm_client_conflict_annotates_all_candidates(self) -> None:
        text = "CA is configured between PCell and SCell."
        result = map_abbreviations(text, SAMPLE_ACRONYMS, llm_client=None)
        assert "CA(Carrier Aggregation|Cell Activation)" in result


class TestBuildReverseIndex:
    def test_expansion_maps_to_abbreviation(self) -> None:
        reverse = build_reverse_index(SAMPLE_ACRONYMS)
        assert reverse["Handover"] == "HO"
        assert reverse["handover"] == "HO"

    def test_variant_maps_to_abbreviation(self) -> None:
        reverse = build_reverse_index(SAMPLE_ACRONYMS)
        assert reverse["Hand-Over"] == "HO"
        assert reverse["hand-over"] == "HO"

    def test_hyphen_stripped_form_maps(self) -> None:
        reverse = build_reverse_index(SAMPLE_ACRONYMS)
        assert reverse["handover"] == "HO"
```

- [ ] **Step 2: 테스트 실행 — fail 확인**

```bash
.venv/bin/pytest tests/preprocessing/test_abbrev_mapper.py -v
```

Expected: `ImportError` (모듈 없음)

- [ ] **Step 3: 패키지 골격 생성**

`src/spar/preprocessing/__init__.py` — 빈 파일:
```python
```

- [ ] **Step 4: `abbrev_mapper.py` 구현 (global + reverse, conflict는 Task 3에서)**

`src/spar/preprocessing/abbrev_mapper.py`:
```python
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI


def load_acronyms(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_reverse_index(acronyms: dict) -> dict[str, str]:
    """전체어·variants → abbreviation 역방향 인덱스 빌드."""
    reverse: dict[str, str] = {}
    for abbrev, info in acronyms.get("global", {}).items():
        expansion: str = info["expansion"]
        for form in [expansion, expansion.lower(), expansion.replace("-", "").lower()]:
            reverse[form] = abbrev
        for variant in info.get("variants", []):
            for form in [variant, variant.lower(), variant.replace("-", "").lower()]:
                reverse[form] = abbrev
    return reverse


def _apply_global(text: str, global_dict: dict) -> str:
    for abbrev, info in global_dict.items():
        expansion: str = info["expansion"]
        text = re.sub(rf"\b{re.escape(abbrev)}\b", f"{abbrev}({expansion})", text)
        for variant in info.get("variants", []):
            text = re.sub(rf"\b{re.escape(variant)}\b", f"{variant}({expansion})", text)
    return text


def _apply_conflicts_no_llm(text: str, conflicts: dict) -> str:
    for abbrev, info in conflicts.items():
        candidates: list[str] = info["candidates"]
        expansion = "|".join(candidates)
        text = re.sub(rf"\b{re.escape(abbrev)}\b", f"{abbrev}({expansion})", text)
    return text


def map_abbreviations(
    text: str,
    acronyms: dict,
    llm_client: OpenAI | None = None,
    model: str = "google/gemma-4-E4B-it",
) -> str:
    """인제스트·쿼리 공용 약어 매핑 진입점."""
    text = _apply_global(text, acronyms.get("global", {}))
    conflicts = acronyms.get("conflicts", {})
    if not conflicts:
        return text
    if llm_client is None:
        return _apply_conflicts_no_llm(text, conflicts)
    return _resolve_and_apply_conflicts(text, conflicts, llm_client, model)


def _resolve_and_apply_conflicts(
    text: str,
    conflicts: dict,
    client: OpenAI,
    model: str,
) -> str:
    contexts = _collect_conflict_contexts(text, conflicts)
    if not contexts:
        return text
    resolutions = _llm_batch_classify(contexts, conflicts, client, model)
    return _apply_conflict_resolutions(text, conflicts, resolutions)


def _collect_conflict_contexts(text: str, conflicts: dict) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for abbrev in conflicts:
        for m in re.finditer(rf"\b{re.escape(abbrev)}\b", text):
            start = max(0, m.start() - 200)
            end = min(len(text), m.end() + 200)
            result.setdefault(abbrev, []).append(text[start:end])
    return result


def _llm_batch_classify(
    contexts: dict[str, list[str]],
    conflicts: dict,
    client: OpenAI,
    model: str,
) -> dict[str, str]:
    items = []
    for abbrev, ctxs in contexts.items():
        candidates = conflicts[abbrev]["candidates"]
        items.append(
            f'약어: "{abbrev}"\n후보: {candidates}\n문맥: "...{ctxs[0]}..."'
        )
    prompt = (
        "다음 약어들의 의미를 문맥에 맞는 후보 중 하나로만 분류해줘. "
        "확신이 없으면 \"uncertain\"으로 답해줘.\n"
        'JSON 형식으로만 답해줘: {"약어": "선택한 후보 또는 uncertain"}\n\n'
        + "\n\n".join(items)
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {abbrev: "uncertain" for abbrev in contexts}


def _apply_conflict_resolutions(
    text: str,
    conflicts: dict,
    resolutions: dict[str, str],
) -> str:
    for abbrev, chosen in resolutions.items():
        candidates: list[str] = conflicts[abbrev]["candidates"]
        expansion = chosen if chosen in candidates else "|".join(candidates)
        text = re.sub(rf"\b{re.escape(abbrev)}\b", f"{abbrev}({expansion})", text)
    return text
```

- [ ] **Step 5: 테스트 실행 — pass 확인**

```bash
.venv/bin/pytest tests/preprocessing/test_abbrev_mapper.py -v
```

Expected: 모든 테스트 PASS (conflict LLM 테스트는 Task 3에 추가)

- [ ] **Step 6: 커밋**

```bash
git add src/spar/preprocessing/ tests/preprocessing/
git commit -m "feat(abbrev): implement global mapping and reverse index"
```

---

## Task 3: Conflict LLM 분류 테스트 추가

**Files:**
- Modify: `tests/preprocessing/test_abbrev_mapper.py`

- [ ] **Step 1: LLM mock 테스트 작성**

`tests/preprocessing/test_abbrev_mapper.py` 하단에 추가:

```python
from unittest.mock import MagicMock


class TestConflictLlmResolution:
    def _make_llm_client(self, json_response: str) -> MagicMock:
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = json_response
        client.chat.completions.create.return_value = MagicMock(choices=[choice])
        return client

    def test_llm_high_confidence_single_expansion(self) -> None:
        text = "CA is configured between PCell and SCell for throughput."
        client = self._make_llm_client('{"CA": "Carrier Aggregation"}')
        result = map_abbreviations(text, SAMPLE_ACRONYMS, llm_client=client)
        assert "CA(Carrier Aggregation)" in result
        assert "|" not in result

    def test_llm_uncertain_annotates_all_candidates(self) -> None:
        text = "The CA procedure was initiated."
        client = self._make_llm_client('{"CA": "uncertain"}')
        result = map_abbreviations(text, SAMPLE_ACRONYMS, llm_client=client)
        assert "CA(Carrier Aggregation|Cell Activation)" in result

    def test_llm_json_parse_failure_falls_back_to_all_candidates(self) -> None:
        text = "CA enabled for this UE."
        client = self._make_llm_client("not valid json")
        result = map_abbreviations(text, SAMPLE_ACRONYMS, llm_client=client)
        assert "CA(Carrier Aggregation|Cell Activation)" in result

    def test_no_conflict_in_text_skips_llm_call(self) -> None:
        text = "HO triggered after TTT expires."
        client = self._make_llm_client("{}")
        map_abbreviations(text, SAMPLE_ACRONYMS, llm_client=client)
        client.chat.completions.create.assert_not_called()

    def test_single_llm_call_for_multiple_conflict_occurrences(self) -> None:
        text = "CA is used here. CA is also used there."
        client = self._make_llm_client('{"CA": "Carrier Aggregation"}')
        map_abbreviations(text, SAMPLE_ACRONYMS, llm_client=client)
        assert client.chat.completions.create.call_count == 1
```

- [ ] **Step 2: 테스트 실행 — pass 확인**

```bash
.venv/bin/pytest tests/preprocessing/test_abbrev_mapper.py::TestConflictLlmResolution -v
```

Expected: 모든 5개 테스트 PASS

- [ ] **Step 3: 커밋**

```bash
git add tests/preprocessing/test_abbrev_mapper.py
git commit -m "test(abbrev): add conflict LLM resolution mock tests"
```

---

## Task 4: 쿼리 확장 함수 추가

**Files:**
- Modify: `src/spar/preprocessing/abbrev_mapper.py`
- Modify: `tests/preprocessing/test_abbrev_mapper.py`

- [ ] **Step 1: 테스트 먼저 작성**

`tests/preprocessing/test_abbrev_mapper.py`에 추가:

```python
from spar.preprocessing.abbrev_mapper import expand_query


class TestExpandQuery:
    def test_abbreviation_expanded_in_query(self) -> None:
        reverse = build_reverse_index(SAMPLE_ACRONYMS)
        result = expand_query("HO threshold setting", SAMPLE_ACRONYMS, reverse)
        assert "HO(Handover)" in result

    def test_full_form_adds_abbreviation(self) -> None:
        reverse = build_reverse_index(SAMPLE_ACRONYMS)
        result = expand_query("Handover failure case", SAMPLE_ACRONYMS, reverse)
        assert "HO" in result

    def test_variant_in_query_resolved(self) -> None:
        reverse = build_reverse_index(SAMPLE_ACRONYMS)
        result = expand_query("Hand-Over ping-pong issue", SAMPLE_ACRONYMS, reverse)
        assert "Handover" in result

    def test_unknown_token_passes_through(self) -> None:
        reverse = build_reverse_index(SAMPLE_ACRONYMS)
        result = expand_query("unknown_token query", SAMPLE_ACRONYMS, reverse)
        assert "unknown_token" in result
```

- [ ] **Step 2: 테스트 실행 — fail 확인**

```bash
.venv/bin/pytest tests/preprocessing/test_abbrev_mapper.py::TestExpandQuery -v
```

Expected: `ImportError` (expand_query 없음)

- [ ] **Step 3: `expand_query` 구현**

`src/spar/preprocessing/abbrev_mapper.py` 하단에 추가:

```python
def expand_query(
    query: str,
    acronyms: dict,
    reverse_index: dict[str, str],
    llm_client: OpenAI | None = None,
    model: str = "google/gemma-4-E4B-it",
) -> str:
    """쿼리 약어 정방향 확장 + 역방향 전체어→약어 추가."""
    # 정방향: 약어 → 전체어 병기
    expanded = map_abbreviations(query, acronyms, llm_client=llm_client, model=model)

    # 역방향: 전체어 토큰 → 약어 주입
    tokens = query.split()
    extra: list[str] = []
    for token in tokens:
        clean = token.strip(".,;:?!()")
        abbrev = reverse_index.get(clean) or reverse_index.get(clean.lower())
        if abbrev and abbrev not in expanded:
            extra.append(abbrev)

    if extra:
        expanded = expanded + " " + " ".join(extra)
    return expanded
```

- [ ] **Step 4: 테스트 실행 — pass 확인**

```bash
.venv/bin/pytest tests/preprocessing/test_abbrev_mapper.py::TestExpandQuery -v
```

Expected: 모든 4개 테스트 PASS

- [ ] **Step 5: 커밋**

```bash
git add src/spar/preprocessing/abbrev_mapper.py tests/preprocessing/test_abbrev_mapper.py
git commit -m "feat(abbrev): add expand_query for bidirectional query expansion"
```

---

## Task 5: 인제스트 파이프라인 통합

**Files:**
- Modify: `scripts/run_ingest.py`

- [ ] **Step 1: `run_ingest.py`에 import 추가**

`scripts/run_ingest.py` 상단 import 블록에 추가 (기존 `sys.path.insert` 라인 아래):

```python
from spar.preprocessing.abbrev_mapper import load_acronyms, map_abbreviations

_ACRONYMS_PATH = Path(__file__).parent.parent / "dictionary" / "acronyms.json"
_ACRONYMS: dict = load_acronyms(_ACRONYMS_PATH) if _ACRONYMS_PATH.exists() else {}
```

- [ ] **Step 2: `ingest_file` 함수 내 파싱 직후에 약어 매핑 삽입**

기존 코드:
```python
    # 1단계: 파싱
    text = parse_document(file_path, doc_type)
    print(f"  parsed: {len(text)} chars")

    # 2단계: 청킹
    chunks = chunk_text(text, doc_type, source_doc)
```

변경 후:
```python
    # 1단계: 파싱
    text = parse_document(file_path, doc_type)
    print(f"  parsed: {len(text)} chars")

    # 1.5단계: 약어 매핑
    text = map_abbreviations(text, _ACRONYMS)
    print(f"  abbrev mapped: {len(text)} chars")

    # 2단계: 청킹
    chunks = chunk_text(text, doc_type, source_doc)
```

- [ ] **Step 3: dry-run으로 파이프라인 동작 확인**

테스트용 임시 파일 생성 후 확인:

```bash
echo "HO failure occurred after TTT expired. CA is configured." > /tmp/test_abbrev.txt
python scripts/run_ingest.py --input-file /tmp/test_abbrev.txt --doc-type parameter_ref --dry-run
```

Expected output (일부):
```
  parsed: 58 chars
  abbrev mapped: N chars
  [DRY RUN] would insert ...
    chunk_id=...  text='HO(Handover) failure occurred after TTT(Time-To-Trigger)...'
```

- [ ] **Step 4: 커밋**

```bash
git add scripts/run_ingest.py
git commit -m "feat(ingest): integrate abbreviation mapping after parse step"
```

---

## Task 6: 전체 테스트 실행

- [ ] **Step 1: 전체 테스트 suite 실행**

```bash
.venv/bin/pytest tests/ -v
```

Expected: 모든 테스트 PASS, 실패 0

- [ ] **Step 2: 타입 체크**

```bash
.venv/bin/mypy src/spar/preprocessing/abbrev_mapper.py
```

Expected: `Success: no issues found`

- [ ] **Step 3: 린트**

```bash
.venv/bin/ruff check src/spar/preprocessing/ tests/preprocessing/
```

Expected: 이슈 없음

- [ ] **Step 4: 최종 커밋 (필요 시)**

```bash
git add -p
git commit -m "chore(abbrev): fix lint/type issues"
```
