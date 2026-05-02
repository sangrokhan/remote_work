# Hybrid Verify Loop 설계

**날짜:** 2026-05-02  
**상태:** 승인됨  
**관련 Phase:** Phase 2 (Retrieval 고도화)

---

## 1. 개요

기존 파이프라인(route → retrieve → rerank → generate)에 `tool_call` 노드와 `verify` 노드를 추가한다. `generate` 결과가 불충분하다고 판단되면 다른 retrieval 전략 + 개선된 쿼리로 재시도한다. 최대 3회 retry. 기존 파이프라인 동작은 `use_verify_loop: False`(기본값)로 보장한다.

---

## 2. 전체 흐름

```
preprocess → rewrite_query → prepare_context → route
  → [retrieve: rag / structured / multi_hop / decomposed]
  → tool_call          ← verify 실패 시 재진입
  → rerank
  → generate
  → verify
      ├─ score ≥ 3 OR retry 소진  →  END
      └─ score < 3 AND retry 가능  →  tool_call
```

---

## 3. 신규 노드

### 3.1 `tool_call` 노드

**역할:** 전략 선택(결정적) + 쿼리 재작성(LLM) + retrieve 실행

**실행 순서:**

1. `tried_strategies`에서 다음 미시도 전략을 결정적으로 선택
2. LLM에 `(original_query, verify_reason)` 전달 → `improved_query` 생성
3. `improved_query`로 해당 전략 직접 실행 (router 우회)
4. 결과를 `raw_chunks`에 머지 (중복 chunk id 제거)
5. `retry_count += 1`, `tried_strategies` 업데이트

**fallback 전략 순서** (1차 routing 결과 제외 후 순서대로):

```
decomposed → multi_hop → structured → rag
```

1차 routing이 `rag`였다면 `decomposed → multi_hop → structured` 순으로 시도.  
이미 시도한 전략은 건너뜀.

**최초 진입 시 (retry_count == 0):**  
verify 실패 전이므로 `verify_reason` 없음. `improved_query = rewritten_query or expanded_query or query`.

### 3.2 `verify` 노드

**역할:** LLM self-eval — 답변 충분성 평가

**프롬프트 (핵심):**
```
Question: {query}
Answer: {answer}
Contexts used: {contexts_summary}

Rate whether the answer sufficiently addresses the question.
Score 1-5 (1=completely insufficient, 5=fully sufficient).
Respond ONLY in JSON: {"score": int, "reason": str}
```

**분기 조건:**

| 조건 | 다음 노드 |
|------|-----------|
| score ≥ 3 | END |
| score < 3 AND retry_count < 3 AND 남은 전략 존재 | tool_call |
| score < 3 AND (retry_count ≥ 3 OR 전략 소진) | END (최선 답변) |

---

## 4. State 추가 필드

```python
# pipeline/state.py 추가
retry_count: int                  # 0부터 시작, tool_call 진입마다 +1
tried_strategies: list[str]       # 실행한 전략명 기록
verify_score: float | None        # verify 노드 출력
verify_reason: str | None         # LLM이 제시한 불충분 이유
improved_query: str | None        # tool_call에서 재작성된 쿼리
```

---

## 5. 그래프 연결 변경 (`graph.py`)

```python
# 기존
[all_retrieve] → rerank → generate → END

# 변경 (use_verify_loop=True 시)
[all_retrieve] → tool_call → rerank → generate → verify
                    ↑                                 |
                    └──────── score < 3 ──────────────┘
                                                      ↓ score ≥ 3 or 소진
                                                     END
```

조건부 엣지:

```python
g.add_conditional_edges(
    "verify",
    _verify_selector,
    {"tool_call": "tool_call", END: END}
)
```

`_verify_selector`:
```python
def _verify_selector(state: SparState) -> str:
    score = state.get("verify_score", 5.0)
    retry_count = state.get("retry_count", 0)
    tried = set(state.get("tried_strategies", []))
    all_strategies = {"rag", "decomposed", "multi_hop", "structured"}
    remaining = all_strategies - tried
    if score < 3 and retry_count < 3 and remaining:
        return "tool_call"
    return END
```

---

## 6. `GraphConfig` 변경

```python
# pipeline/config.py
@dataclass
class GraphConfig:
    ...
    use_verify_loop: bool = False  # 기본 off — 기존 동작 보존
```

---

## 7. 파일 변경 목록

| 파일 | 변경 내용 |
|------|-----------|
| `src/spar/pipeline/state.py` | 5개 필드 추가 |
| `src/spar/pipeline/nodes.py` | `tool_call`, `verify` 노드 메서드 추가 |
| `src/spar/pipeline/graph.py` | `use_verify_loop` 분기, 엣지 추가, `_verify_selector` 추가 |
| `src/spar/pipeline/config.py` | `use_verify_loop` 플래그 추가 |
| `src/spar/prompts/` | `verify.txt`, `tool_call_rewrite.txt` 프롬프트 파일 추가 |
| `tests/pipeline/` | `test_tool_call.py`, `test_verify.py` 추가 |

---

## 8. 미결 사항

- `contexts_summary` 구성 방식: chunk 전체 vs 앞 N자 요약
- `tool_call_rewrite.txt` 프롬프트 퀄리티 — 초기 구현 후 실험 필요
- verify score 임계값(3) — 운영 후 데이터 보고 조정 가능
