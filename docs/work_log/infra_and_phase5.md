# 인프라 + Phase 5 작업 이력

---

## Task 5.1 — LangGraph 파이프라인 (조기 도입, 2026-05-01)

설계 문서: `docs/superpowers/plans/2026-05-01-langgraph-pipeline.md`

- `src/spar/pipeline/graph.py` — StateGraph scaffold, conditional edges
  - STRUCTURED_LOOKUP → structured_retrieve
  - DIAGNOSTIC → multi_hop_retrieve
  - 나머지 → rag_retrieve
- `src/spar/pipeline/nodes.py` — Milvus hybrid_search 연결 완료 (stub 제거), 실제 LLM generate 노드
- `src/spar/pipeline/state.py` — `SparState`, `node_timings` (노드별 실행 시간 ms)
- `src/spar/pipeline/config.py` — `GraphConfig` + PRESET_CONFIGS (baseline/+reranker/+qexpand/+context/full_retrieval/e2e)

---

## Task INF-1b — Codex 서브에이전트 + Gemini Fallback (완료, 2026-05-01)

브랜치: `feat/gemini-fallback`

- `3e3b3010` — SubagentStart → Codex CLI headless 라우팅
- `3c4ebea7` — codex exec 프롬프트 stdin 파일 주입 (ARG_MAX 회피)
- `41c0a565` — Codex 토큰 소진 시 Gemini CLI fallback

설계 문서: `docs/superpowers/plans/2026-05-01-codex-subagent-workflow.md`
