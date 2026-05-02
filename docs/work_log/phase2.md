# Phase 2 작업 이력

---

## Task 2.1 — 라우트 정의 (완료, 2026-05-01)

- `src/spar/router/schemas.py` — 6개 라우트 정의
- `src/spar/router/routes_spec.md` — 판단 기준 문서화

---

## Task 2.2 — Hybrid 라우터 (구현 중, 2026-05-01)

### Layer 1 추가 사항

- 3GPP TS spec number 패턴 추가: `TS \d{2}.\d{3}` → `DEFINITION_EXPLAIN` + `spec_number` entity
- `scripts/slice_3gpp_intros.py` — spec intro 1000줄 추출
- `run_ingest.py --intro-only` — spec_number dynamic field 부착

### 추가 산출물

- `router/hybrid_router.py` — 3-layer 통합 오케스트레이터
- `src/spar/prompts/` — LLM 프롬프트 중앙화 패키지 (router_system.txt, abbrev_conflict.txt)

---

## Task 2.3 — 라우터 골드셋 (구현 중)

- `scripts/gen_goldset.py` — QA+router 통합 단일 JSONL 생성기
  - 7개 QA 타입: terminology→definition_explain, technology/behavior→default_rag, diagnostic→diagnostic, procedural→procedural, comparative→comparative, lookup→structured_lookup
  - 6개 라우트 전체 커버
  - Codex→Gemini fallback, MAX_ATTEMPTS=10
- 삭제: `scripts/gen_goldset_qa.py`, `scripts/gen_router_goldset.py` → gen_goldset.py로 통합

---

## Task 2.4 — Query Decomposition (완료, 2026-05-01)

merge commit: `bc9ac0ef`

- `src/spar/retrieval/query_decomposer.py` — `QueryDecomposer`
- `RouteResult.needs_decomposition` — LLMRouter에서 "complex" 판정
- `prompts/decompose_system.txt`
- `pipeline/decomposed_retrieve` 노드 — 병렬 검색, chunk dedup + merge

---

## Task 2.5 — Query Rewriting (완료, 2026-05-01)

브랜치: `feat/query-rewriter-llm`

- `src/spar/retrieval/query_rewriter.py` — `QueryRewriteResult` + `rewrite_query()`
- `src/spar/prompts/query_rewrite_system.txt`
- `src/spar/pipeline/nodes.py` — `rewrite_query` node
- `tests/unit/retrieval/test_query_rewriter.py`
- 부가: `QueryRequest.history`, `SparState.history/history_context`, `pipeline/prepare_context` 노드

---

## Task 2.6 — Hybrid Verify Loop (완료, 2026-05-02)

브랜치: `feat/hybrid-verify-loop` → merge commit: `95604589`

- verify_node: LLM self-evaluation score 1-5, threshold < 3 → fallback
- tool_call_node: 결정론적 fallback 전략 순서 — decomposed → multi_hop → structured → rag
- `src/spar/pipeline/state.py` — `verify_score` field
- `src/spar/pipeline/config.py` — `verify_loop` preset (`use_verify_loop=True`)
- `src/spar/prompts/verify.txt`, `src/spar/prompts/tool_call_rewrite.txt`
- `tests/pipeline/test_verify_node.py` — 59개 테스트

---

## Task 2.8 설계 문서

`docs/superpowers/plans/2026-05-01-hyde-multi-query.md`
