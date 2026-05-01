# Eval Metrics Dashboard

> 마지막 업데이트: —  
> 골드셋: —

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

## Phase 1 Answer Quality

| Metric | Value | Target |
|--------|-------|--------|
| n_samples | — | — |
| faithfulness | — | ≥ 0.9 (Phase 4 목표) |
| answer_relevancy | — | ≥ 0.8 |

---

## 측정 명령

```bash
# Retrieval (Recall@K / MRR)
python -m spar.eval.run_eval \
  --goldset data/goldsets/retrieval_goldset.jsonl \
  --doc-type spec \
  --top-k 50 \
  --output data/eval_results/phase1_eval.json

# 답변 품질 (faithfulness / answer_relevancy)
python -m spar.eval.run_ragas_eval \
  --dataset data/eval_results/ragas_dataset.jsonl \
  --metrics faithfulness,answer_relevancy \
  --output data/eval_results/ragas_eval.json
```
