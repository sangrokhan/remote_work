# RAGAS Metrics (Task 1.7.2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add RAGAS faithfulness + answer_relevancy evaluation to complete Task 1.7.2 answer quality measurement.

**Architecture:** Pre-computed dataset approach — CLI accepts JSONL with (question, answer, contexts), wraps RAGAS 0.2+ evaluate(), outputs per-query scores + aggregate JSON. Decoupled from pipeline stub; LLM judge configurable via env vars (OPENAI_API_KEY + OPENAI_BASE_URL for vllm compatibility).

**Tech Stack:** ragas>=0.2, langchain-openai, datasets, existing `.venv/bin/pytest` + `.venv/bin/pip`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/spar/eval/ragas_metrics.py` | RAGAS evaluate() wrapper — `compute_ragas_metrics()` |
| Create | `src/spar/eval/run_ragas_eval.py` | CLI: load JSONL → evaluate → print + save JSON |
| Create | `tests/eval/test_ragas_metrics.py` | Unit tests with mocked `ragas.evaluate` |
| Create | `src/spar/eval/metrics_dashboard.md` | Dashboard template (산출물) |
| Modify | `requirements.txt` | Add `ragas>=0.2`, `langchain-openai>=0.3` |
| Modify | `docs/prd.md` | Check RAGAS checkbox in 1.7.2 |
| Modify | `AGENTS.md` | Update eval section |

---

## Task 1: Install RAGAS + Update Requirements

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Install packages into venv**

```bash
.venv/bin/pip install "ragas>=0.2" "langchain-openai>=0.3" 2>&1 | tail -5
```

Expected: `Successfully installed ragas-0.2.x ...`

- [ ] **Step 2: Verify import works**

```bash
.venv/bin/python3 -c "from ragas import evaluate; from ragas.metrics import Faithfulness, AnswerRelevancy; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Update requirements.txt**

Open `requirements.txt` and add after the `ragas>=0.1` line:

```
ragas>=0.2
langchain-openai>=0.3
```

Remove the old `ragas>=0.1` line.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore(deps): install ragas>=0.2 and langchain-openai for RAGAS eval"
```

---

## Task 2: Core RAGAS Metrics Module

**Files:**
- Create: `src/spar/eval/ragas_metrics.py`
- Create: `tests/eval/test_ragas_metrics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/eval/test_ragas_metrics.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from spar.eval.ragas_metrics import RagasSample, compute_ragas_metrics, load_ragas_dataset


# ── fixtures ──────────────────────────────────────────────────────────────────

SAMPLE: RagasSample = {
    "query_id": "Q001",
    "question": "What is handover?",
    "answer": "Handover transfers an active call between cells.",
    "contexts": ["Handover transfers an active call between cells.", "X2 handover uses direct eNB interface."],
}

SAMPLE_NO_GROUND_TRUTH: RagasSample = {
    "query_id": "Q002",
    "question": "What is PDCP?",
    "answer": "PDCP handles packet compression.",
    "contexts": ["PDCP performs header compression and ciphering."],
}


# ── load_ragas_dataset ─────────────────────────────────────────────────────────

def test_load_ragas_dataset(tmp_path):
    import json

    lines = [
        json.dumps({"query_id": "Q001", "question": "q1", "answer": "a1", "contexts": ["c1"]}),
        json.dumps({"query_id": "Q002", "question": "q2", "answer": "a2", "contexts": ["c2", "c3"], "ground_truth": "gt2"}),
        "",  # blank lines ignored
    ]
    p = tmp_path / "dataset.jsonl"
    p.write_text("\n".join(lines))

    samples = load_ragas_dataset(p)
    assert len(samples) == 2
    assert samples[0]["query_id"] == "Q001"
    assert "ground_truth" not in samples[0]
    assert samples[1]["ground_truth"] == "gt2"


def test_load_ragas_dataset_missing_field(tmp_path):
    import json

    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"query_id": "Q001", "question": "q1"}))  # missing answer + contexts

    with pytest.raises(ValueError, match="answer"):
        load_ragas_dataset(p)


# ── compute_ragas_metrics ──────────────────────────────────────────────────────

def _mock_ragas_result(scores: dict[str, list[float]]) -> MagicMock:
    """Build a fake ragas EvaluationResult."""
    mock_result = MagicMock()
    mock_result.scores = [
        {metric: val for metric, val in zip(scores.keys(), [v[i] for v in scores.values()])}
        for i in range(len(next(iter(scores.values()))))
    ]
    mock_result.__getitem__ = lambda self, key: sum(scores[key]) / len(scores[key])
    return mock_result


@patch("spar.eval.ragas_metrics.ragas_evaluate")
def test_compute_ragas_metrics_faithfulness(mock_eval):
    mock_eval.return_value = _mock_ragas_result({"faithfulness": [0.8, 0.9]})

    result = compute_ragas_metrics([SAMPLE, SAMPLE_NO_GROUND_TRUTH], metrics=["faithfulness"])

    assert "faithfulness" in result
    assert 0.0 <= result["faithfulness"] <= 1.0
    assert result["n_samples"] == 2
    assert len(result["per_sample"]) == 2


@patch("spar.eval.ragas_metrics.ragas_evaluate")
def test_compute_ragas_metrics_both(mock_eval):
    mock_eval.return_value = _mock_ragas_result({
        "faithfulness": [0.8, 0.9],
        "answer_relevancy": [0.7, 0.85],
    })

    result = compute_ragas_metrics([SAMPLE, SAMPLE_NO_GROUND_TRUTH], metrics=["faithfulness", "answer_relevancy"])

    assert result["faithfulness"] == pytest.approx(0.85, abs=1e-3)
    assert result["answer_relevancy"] == pytest.approx(0.775, abs=1e-3)


@patch("spar.eval.ragas_metrics.ragas_evaluate")
def test_compute_ragas_metrics_per_sample_ids(mock_eval):
    mock_eval.return_value = _mock_ragas_result({"faithfulness": [0.75]})

    result = compute_ragas_metrics([SAMPLE], metrics=["faithfulness"])

    assert result["per_sample"][0]["query_id"] == "Q001"
    assert "faithfulness" in result["per_sample"][0]


def test_compute_ragas_metrics_empty():
    with pytest.raises(ValueError, match="empty"):
        compute_ragas_metrics([], metrics=["faithfulness"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/eval/test_ragas_metrics.py -v 2>&1 | tail -15
```

Expected: `ImportError` or `ModuleNotFoundError: spar.eval.ragas_metrics`

- [ ] **Step 3: Implement ragas_metrics.py**

Create `src/spar/eval/ragas_metrics.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import NotRequired, TypedDict

from ragas import evaluate as ragas_evaluate  # aliased for mocking in tests
from ragas.metrics import AnswerRelevancy, Faithfulness


class RagasSample(TypedDict):
    query_id: str
    question: str
    answer: str
    contexts: list[str]
    ground_truth: NotRequired[str]


_METRIC_MAP = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
}

_REQUIRED_FIELDS = {"query_id", "question", "answer", "contexts"}


def load_ragas_dataset(path: Path) -> list[RagasSample]:
    samples: list[RagasSample] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        missing = _REQUIRED_FIELDS - raw.keys()
        if missing:
            raise ValueError(f"Sample missing fields: {missing} — got keys: {list(raw.keys())}")
        sample: RagasSample = {
            "query_id": raw["query_id"],
            "question": raw["question"],
            "answer": raw["answer"],
            "contexts": raw["contexts"],
        }
        if "ground_truth" in raw:
            sample["ground_truth"] = raw["ground_truth"]
        samples.append(sample)
    return samples


def compute_ragas_metrics(
    samples: list[RagasSample],
    metrics: list[str],
    llm=None,
    embeddings=None,
) -> dict:
    if not samples:
        raise ValueError("Cannot evaluate empty dataset")

    from datasets import Dataset

    metric_objs = [_METRIC_MAP[m]() for m in metrics]

    ragas_rows = [
        {
            "user_input": s["question"],
            "response": s["answer"],
            "retrieved_contexts": s["contexts"],
            **({"reference": s["ground_truth"]} if "ground_truth" in s else {}),
        }
        for s in samples
    ]
    dataset = Dataset.from_list(ragas_rows)

    eval_kwargs: dict = {"dataset": dataset, "metrics": metric_objs}
    if llm is not None:
        eval_kwargs["llm"] = llm
    if embeddings is not None:
        eval_kwargs["embeddings"] = embeddings

    result = ragas_evaluate(**eval_kwargs)

    per_sample = []
    for i, s in enumerate(samples):
        row: dict = {"query_id": s["query_id"]}
        for m in metrics:
            row[m] = result.scores[i].get(m)
        per_sample.append(row)

    def _avg(metric: str) -> float:
        vals = [r[metric] for r in per_sample if r[metric] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "n_samples": len(samples),
        **{m: _avg(m) for m in metrics},
        "per_sample": per_sample,
    }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
.venv/bin/pytest tests/eval/test_ragas_metrics.py -v 2>&1 | tail -15
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/spar/eval/ragas_metrics.py tests/eval/test_ragas_metrics.py
git commit -m "feat(eval): add ragas_metrics module — faithfulness + answer_relevancy"
```

---

## Task 3: RAGAS Eval CLI

**Files:**
- Create: `src/spar/eval/run_ragas_eval.py`

- [ ] **Step 1: Implement run_ragas_eval.py**

Create `src/spar/eval/run_ragas_eval.py`:

```python
"""
RAGAS 답변 품질 평가 CLI — faithfulness, answer_relevancy

Usage:
    python -m spar.eval.run_ragas_eval \
        --dataset data/eval_results/ragas_dataset.jsonl \
        --metrics faithfulness,answer_relevancy \
        --output data/eval_results/ragas_eval.json \
        [--llm-base-url http://localhost:8000/v1] \
        [--llm-model Qwen2.5-7B-Instruct]

Dataset JSONL format (one JSON object per line):
    {"query_id": "Q001", "question": "...", "answer": "...", "contexts": ["...", "..."]}
    optional: "ground_truth": "..."

LLM judge:
    Default: uses OPENAI_API_KEY env var (set OPENAI_BASE_URL for vllm).
    --llm-base-url / --llm-model override these for convenience.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _build_llm(base_url: str | None, model: str | None):
    if base_url is None and model is None:
        return None  # RAGAS uses OPENAI_API_KEY from env by default
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    kwargs: dict = {}
    if base_url:
        kwargs["base_url"] = base_url
        kwargs["api_key"] = os.environ.get("OPENAI_API_KEY", "token")
    if model:
        kwargs["model"] = model
    return LangchainLLMWrapper(ChatOpenAI(**kwargs))


def _print_summary(metrics: dict) -> None:
    print(f"\n{'─' * 50}")
    print(f"  n_samples : {metrics['n_samples']}")
    for key, val in metrics.items():
        if key in ("n_samples", "per_sample"):
            continue
        print(f"  {key:<22} {val:.4f}")
    print(f"{'─' * 50}\n")


def _save_output(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"  Saved → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS 답변 품질 평가")
    parser.add_argument("--dataset", required=True, type=Path, help="JSONL 데이터셋 경로")
    parser.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy",
        help="평가 지표 (쉼표 구분, default: faithfulness,answer_relevancy)",
    )
    parser.add_argument("--output", type=Path, default=None, help="결과 JSON 저장 경로")
    parser.add_argument("--llm-base-url", default=None, help="LLM judge base URL (vllm 등)")
    parser.add_argument("--llm-model", default=None, help="LLM judge 모델명")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    from spar.eval.ragas_metrics import compute_ragas_metrics, load_ragas_dataset

    metric_names = [m.strip() for m in args.metrics.split(",")]
    valid = {"faithfulness", "answer_relevancy"}
    unknown = set(metric_names) - valid
    if unknown:
        print(f"Error: unknown metrics {unknown}. Valid: {valid}", file=sys.stderr)
        sys.exit(1)

    print(f"데이터셋 로드: {args.dataset}")
    samples = load_ragas_dataset(args.dataset)
    print(f"  → {len(samples)}개 샘플")

    llm = _build_llm(args.llm_base_url, args.llm_model)
    if llm:
        print(f"  LLM judge: {args.llm_model} @ {args.llm_base_url}")
    else:
        print("  LLM judge: OPENAI_API_KEY from env")

    print(f"\n지표 평가 중: {metric_names}")
    result = compute_ragas_metrics(samples, metrics=metric_names, llm=llm)

    _print_summary(result)

    if args.output:
        _save_output(args.output, result)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test CLI help**

```bash
.venv/bin/python3 -m spar.eval.run_ragas_eval --help 2>&1 | head -10
```

Expected: Usage line shown, no import errors.

- [ ] **Step 3: Run full test suite to check no regressions**

```bash
.venv/bin/pytest tests/eval/ -q 2>&1 | tail -5
```

Expected: all pass (≥18 + 5 new = ≥23 passed)

- [ ] **Step 4: Commit**

```bash
git add src/spar/eval/run_ragas_eval.py
git commit -m "feat(eval): add run_ragas_eval CLI for RAGAS faithfulness/answer_relevancy"
```

---

## Task 4: metrics_dashboard.md 산출물

**Files:**
- Create: `src/spar/eval/metrics_dashboard.md`

- [ ] **Step 1: Create dashboard template**

Create `src/spar/eval/metrics_dashboard.md`:

```markdown
# Eval Metrics Dashboard

> 마지막 업데이트: <!-- date -->  
> 골드셋: <!-- path -->

---

## Phase 1 Retrieval Metrics

| Metric | Value |
|--------|-------|
| n_queries | — |
| MRR | — |
| Recall@5 | — |
| Recall@10 | — |
| Recall@50 | — |

### By Query Type

| Type | n | MRR | R@5 | R@10 |
|------|---|-----|-----|------|
| definition | — | — | — | — |
| procedural | — | — | — | — |
| diagnostic | — | — | — | — |
| comparative | — | — | — | — |
| lookup | — | — | — | — |

---

## Phase 1 Answer Quality (RAGAS)

| Metric | Value | Target |
|--------|-------|--------|
| n_samples | — | — |
| faithfulness | — | ≥ 0.9 (Phase 4 목표) |
| answer_relevancy | — | ≥ 0.8 |

---

## 측정 명령

```bash
# Retrieval (Recall@K / MRR)
.venv/bin/python3 -m spar.eval.run_eval \
  --goldset data/goldsets/retrieval_goldset.jsonl \
  --doc-type spec \
  --top-k 50 \
  --output data/eval_results/phase1_eval.json

# 답변 품질 (RAGAS)
.venv/bin/python3 -m spar.eval.run_ragas_eval \
  --dataset data/eval_results/ragas_dataset.jsonl \
  --metrics faithfulness,answer_relevancy \
  --output data/eval_results/ragas_eval.json \
  --llm-base-url http://localhost:8000/v1 \
  --llm-model Qwen2.5-7B-Instruct
```
```

- [ ] **Step 2: Commit**

```bash
git add src/spar/eval/metrics_dashboard.md
git commit -m "docs(eval): add metrics_dashboard.md template (Task 1.7.2 산출물)"
```

---

## Task 5: PRD + AGENTS.md 업데이트

**Files:**
- Modify: `docs/prd.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update docs/prd.md**

In `docs/prd.md`, find the 1.7.2 section and update:

```
- [ ] 답변 품질 평가 (RAGAS faithfulness, answer_relevancy)
```
→
```
- [x] 답변 품질 평가 (RAGAS faithfulness, answer_relevancy)
```

Also update the status line from:
```
> 🔧 **구현 중** (2026-05-01) — Recall@K/MRR CLI 완료. RAGAS·CI/CD 미착수.
```
→
```
> 🔧 **구현 중** (2026-05-01) — Recall@K/MRR CLI 완료. RAGAS 완료. CI/CD 미착수.
```

Also check the `metrics_dashboard.md` deliverable:
```
- [x] **산출물**: `src/spar/eval/run_eval.py` ✅, `src/spar/eval/metrics_dashboard.md` ☐
```
→
```
- [x] **산출물**: `src/spar/eval/run_eval.py` ✅, `src/spar/eval/metrics_dashboard.md` ✅
```

- [ ] **Step 2: Update AGENTS.md**

In `AGENTS.md` section `4. 빌드 / 테스트 / 평가 명령`, find or add RAGAS eval command after the retrieval eval entry:

```markdown
# RAGAS 답변 품질 평가
.venv/bin/python3 -m spar.eval.run_ragas_eval \
  --dataset data/eval_results/ragas_dataset.jsonl \
  --metrics faithfulness,answer_relevancy
```

- [ ] **Step 3: Commit**

```bash
git add docs/prd.md AGENTS.md
git commit -m "docs: mark RAGAS eval done in prd.md, add eval command to AGENTS.md"
```

---

## Self-Review

### Spec Coverage

| PRD requirement | Task |
|----------------|------|
| RAGAS faithfulness | Task 2 (`ragas_metrics.py`) |
| answer_relevancy | Task 2 (`ragas_metrics.py`) |
| CLI 자동화 | Task 3 (`run_ragas_eval.py`) |
| metrics_dashboard.md 산출물 | Task 4 |
| prd.md checkbox | Task 5 |

### Type Consistency

- `RagasSample` defined in Task 2, used in Task 3 — consistent
- `compute_ragas_metrics(samples, metrics, llm, embeddings)` — same signature Tasks 2 and 3
- `ragas_evaluate` alias used in both module and test mock — consistent

### Placeholder Check

- No TBD/TODO left in code steps
- All code blocks complete
- Test file has full implementations

---

## Worktree Info

- Branch: `feat/ragas-metrics`
- Worktree: `.worktrees/feat-ragas-metrics`
- Base: `feat/eval-scripts`
- Test command: `.venv/bin/pytest tests/eval/ -q`
