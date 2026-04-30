# Samsung RAN LLM+RAG 시스템 구축 작업 내역서

> **목적**: Samsung 단일 벤더 환경(LTE+NR)에서 내부 생성 RAN 제어 파라미터, 운영 통계, 알람 코드, 작업/설치 절차서 등에 대한 자연어 질의 응답 시스템 구축
> **운영 환경**: 온프레미스, 영어 응답, 정확성 최우선 (hallucination 최소화)
> **작성일**: 2026-04-30

---

## 0. 전체 로드맵 개요

| Phase | 기간 | 핵심 목표 | 주요 산출물 |
|---|---|---|---|
| Phase 0 | 1-2주 | 현황 진단 및 Baseline 측정 | 골드셋, baseline metrics |
| Phase 1 | 4-6주 | 데이터 파이프라인 + Retrieval 강화 | 문서 유형별 인덱스, hybrid search, reranker |
| Phase 2 | 4-6주 | 쿼리 라우터 + 복잡 질의 처리 | Router, query decomposition, multi-index |
| Phase 3 | 4-6주 | 구조화 데이터 + Knowledge Graph | Parameter/Counter/Alarm DB, KG |
| Phase 4 | 3-4주 | 정확성 강화 (Hallucination 통제) | Citation 강제, self-verification, confidence |
| Phase 5 | 선택 | Agentic 확장 (설정 자동화 대비) | LangGraph 기반 재구성 |

---

## Phase 0: 현황 진단 및 Baseline 측정

> **목적**: 현재 시스템의 성능을 정량적으로 측정하고, 개선 우선순위를 정한다.
> **선행 조건**: 현재 RAG 시스템이 일부 구현되어 동작 중

### Task 0.1 — 사용자 질의 로그 수집 및 분류

- [ ] 실제 사용자 질의 100~200개 샘플링 (다양한 유형 포함)
- [ ] 수동 분류로 질의 유형 도출 (정의/절차/진단/비교/lookup 등)
- [ ] 유형별 빈도 측정 → 어느 라우트에 투자할지 결정
- [ ] **산출물**: `query_log_classified.csv` (query, type, complexity, current_answer_quality)

### Task 0.2 — Retrieval 골드셋 구축

- [ ] SME(현장 전문가)와 협업하여 골드셋 100~200문항 작성
- [ ] 각 질의에 대해 정답 청크 ID 또는 정답 문서/섹션 매핑
- [ ] **복잡 질의에는 여러 relevant chunk 매핑** (분산된 정답 케이스 측정 위해)
- [ ] 질의 유형별로 균형 있게 분포 (definition / procedural / diagnostic / comparative)
- [ ] **산출물**: `retrieval_goldset.jsonl`
  ```jsonl
  {"query_id": "Q001", "query": "...", "type": "diagnostic", "relevant_chunk_ids": ["c_123", "c_456"], "product": "LTE", "release": "v6.0"}
  ```

### Task 0.3 — Baseline 평가 자동화 스크립트

- [ ] Recall@5, Recall@10, Recall@50, MRR 측정 스크립트
- [ ] 답변 품질 평가 (RAGAS faithfulness, answer_relevancy)
- [ ] 질의 유형별 분리 측정 (어디가 약한지 파악)
- [ ] CI/CD 형태로 변경마다 자동 재측정
- [ ] **산출물**: `eval/run_eval.py`, `eval/metrics_dashboard.md`

### Task 0.4 — 실패 케이스 분석 리포트

- [ ] Recall@10 실패 케이스 50개 샘플링
- [ ] 실패 원인 분류 (vocabulary mismatch / decomposition needed / multi-index needed / ambiguous query / chunking issue)
- [ ] 우선순위 매트릭스 작성 (영향도 × 난이도)
- [ ] **산출물**: `failure_analysis_report.md`

---

## Phase 1: 데이터 파이프라인 + Retrieval 강화

> **목적**: 문서 유형별 차별화된 처리와 hybrid retrieval로 단일 질의 retrieval 품질을 끌어올린다.

### Task 1.1 — 문서 유형 분류 및 파서 개발

Samsung RAN 문서는 형태가 매우 다양하므로, 유형별 분리 처리가 필수.

- [ ] 문서 유형 정의 (Parameter Reference / Counter Reference / Alarm Reference / Feature Description / MOP / Installation Guide / Release Notes)
- [ ] 각 유형별 PDF/문서 파서 개발
  - 표 추출: `pdfplumber`, `camelot`, 또는 `unstructured` 라이브러리
  - 섹션 헤더 추출: 목차 기반 또는 폰트 크기 기반
- [ ] OCR이 필요한 스캔 문서는 별도 처리 (Tesseract, PaddleOCR)
- [ ] **산출물**: `parsers/` 디렉토리, 유형별 파서 모듈

### Task 1.2 — 메타데이터 스키마 정의 및 부착

- [ ] 모든 청크에 다음 메타데이터 강제 부착:
  - `doc_type` (parameter_ref / mop / install_guide / ...)
  - `product` (LTE / NR / both)
  - `release` (예: v6.0, v7.1)
  - `deployment_type` (운영 환경/방식 구분)
  - `mo_name` (해당되는 경우)
  - `source_doc`, `section`, `page`
- [ ] 검색 시 메타데이터 필터로 운영 환경별 답변 분리 가능하게
- [ ] **산출물**: `chunk_schema.json`, 청킹 파이프라인

### Task 1.3 — 문서 유형별 청킹 전략 차별화

- [ ] **Reference 문서 (Parameter/Counter/Alarm)**: 항목 단위 청크 (parameter 1개 = chunk 1개), 메타데이터에 name 명시
- [ ] **MOP/Installation**: 절차 단위 청크 (절차 헤더 + 단계들 함께), 절차가 잘리지 않도록
- [ ] **Feature Description**: 섹션 단위 청크, 부모 섹션 컨텍스트 포함
- [ ] **Release Notes**: 변경 항목 단위 청크
- [ ] 청크 크기는 유형별로 다르게 (절대 균일 크기 적용 금지)
- [ ] **산출물**: `chunkers/` 디렉토리, 유형별 청킹 모듈

### Task 1.4 — 인덱스 분리 및 Hybrid Search 구축

- [ ] 문서 유형별 별도 인덱스 구성 (단일 통합 인덱스는 노이즈 발생)
- [ ] **BM25 인덱스**: Elasticsearch 또는 OpenSearch
- [ ] **Dense 인덱스**: 임베딩 모델 선정 후 벡터 DB (**Milvus** 채택, 2026-04-30 결정 / 대안: Qdrant, Weaviate)
- [ ] 임베딩 모델 후보:
  - `BAAI/bge-large-en-v1.5` (검증된 영어 모델)
  - `intfloat/e5-large-v2`
  - `nomic-embed-text-v1.5`
  - 도메인 적응 fine-tuning은 Phase 4 이후 검토
- [ ] Hybrid 결합: RRF (Reciprocal Rank Fusion) 또는 weighted sum
- [ ] **파라미터/카운터/알람 검색은 BM25 가중치 0.5 이상** (정확 매칭 중요)
- [ ] **산출물**: 인덱싱 파이프라인, hybrid search API

### Task 1.5 — Reranker 도입 (Cross-encoder)

> **현재 retrieval 실패의 가장 큰 ROI 개선 항목 중 하나**

- [ ] Bi-encoder로 top-50~100 → Cross-encoder로 top-5~10 패턴
- [ ] Reranker 모델 후보:
  - `BAAI/bge-reranker-v2-m3`
  - `jinaai/jina-reranker-v2-base-multilingual`
- [ ] vLLM 또는 별도 GPU 인스턴스로 서빙
- [ ] Latency 측정 및 batch 최적화
- [ ] **산출물**: reranker 서빙 설정, retrieval 파이프라인 통합

### Task 1.6 — 도메인 약어/동의어 사전 구축

- [ ] Samsung RAN에서 사용하는 약어 사전 작성 (HO, CA, TTT, RACH, BWP 등)
- [ ] 정식 용어 ↔ 약어 ↔ 풀이 매핑
- [ ] 질의 전처리에서 양방향 확장 적용
- [ ] BM25 인덱스에 동의어 확장 적용
- [ ] **산출물**: `dictionary/synonyms.json`, `dictionary/acronyms.json`

### Task 1.7 — Phase 1 평가

- [ ] Phase 0 골드셋으로 Recall@5/10/50, MRR 재측정
- [ ] Baseline 대비 개선 폭 보고
- [ ] 여전히 실패하는 케이스 재분류 → Phase 2 입력
- [ ] **산출물**: `phase1_eval_report.md`

---

## Phase 2: 쿼리 라우터 + 복잡 질의 처리

> **목적**: 복잡한/다측면 질의를 분해하고, 질의 유형별로 다른 검색 전략을 적용한다.

### Task 2.1 — 라우트 정의 (도메인 특화)

- [ ] 다음 라우트로 분류:
  - `structured_lookup` — 파라미터/카운터/알람 정확 조회 → KG/DB 직접
  - `definition_explain` — 서술형 설명 → RAG + structured 보조
  - `procedural` — 절차/설치/설정 방법 → MOP/Install 인덱스
  - `diagnostic` — 트러블슈팅 → multi-hop, KG 활용
  - `comparative` — 버전/기능 비교 → Release Notes 인덱스
  - `default_rag` — 기타 → 전체 hybrid 검색
- [ ] 각 라우트별 검색 전략 명세 문서화
- [ ] **산출물**: `router/routes_spec.md`

### Task 2.2 — Hybrid 라우터 구현 (3-layer)

#### Layer 1: Regex Fast-path

- [ ] 알람 코드 패턴 (`ALM-\d+`, `alarm \d+`)
- [ ] 파라미터명 사전 매칭 (Phase 1.6 사전 활용)
- [ ] MO 이름 매칭
- [ ] Hit 시 → `structured_lookup`으로 직행
- [ ] **산출물**: `router/regex_router.py`

#### Layer 2: Embedding Similarity Router

- [ ] `semantic-router` 라이브러리 활용
- [ ] 라우트별 예시 10~30개씩 작성
- [ ] Threshold 튜닝 (초기 0.7)
- [ ] Confidence 낮으면 Layer 3로
- [ ] **산출물**: `router/embedding_router.py`, `router/route_examples.yaml`

#### Layer 3: LLM Router (Function Calling)

- [ ] 작은 모델 사용 (Qwen 2.5 7B Instruct 또는 Llama 3.1 8B Instruct)
- [ ] vLLM 별도 인스턴스로 서빙
- [ ] **라우팅 + entity extraction을 한 번에**
  - route, entities (parameter/counter/alarm names), product, release
- [ ] Pydantic schema로 structured output 강제
- [ ] **산출물**: `router/llm_router.py`, 라우터 시스템 프롬프트

### Task 2.3 — 라우터 골드셋 및 평가

- [ ] 라우터 전용 골드셋 300~500개 (질의 → 정답 라우트)
- [ ] Confusion matrix 분석 (어느 라우트끼리 헷갈리는지)
- [ ] 라우팅 정확도 측정 (overall + per-route)
- [ ] **산출물**: `router_goldset.jsonl`, `router_eval_report.md`

### Task 2.4 — Query Decomposition (복잡 질의 분해)

> **다측면 질의 retrieval 실패의 주요 해결책**

- [ ] 라우터에서 "complex" 판정 시 decomposition 활성화
- [ ] LLM 프롬프트로 sub-question 리스트 생성
- [ ] 각 sub-question으로 개별 검색
- [ ] 결과 union → reranker로 통합 top-k 선정
- [ ] **산출물**: `retrieval/query_decomposer.py`

### Task 2.5 — Query Rewriting (대화 컨텍스트 반영)

- [ ] 멀티턴 대화에서 질의를 self-contained 형태로 rewrite
- [ ] 약어 자동 확장
- [ ] 모호한 지시어 명시화 ("그 파라미터" → 실제 이름)
- [ ] **산출물**: `retrieval/query_rewriter.py`

### Task 2.6 — HyDE / Multi-Query Expansion (선택적)

- [ ] HyDE: 가상 답변 생성 후 임베딩 검색
- [ ] Multi-Query: 3~4개 alternative phrasing 생성 후 RRF 결합
- [ ] A/B 테스트로 효과 확인 후 적용 여부 결정
- [ ] **산출물**: `retrieval/hyde.py`, `retrieval/multi_query.py`

### Task 2.7 — Multi-Index 병렬 검색

- [ ] 라우트에 따라 여러 인덱스 동시 검색
- [ ] 각 인덱스에서 top-N → 통합 → reranker
- [ ] 진단 질의의 경우 KG hop 결과도 결합
- [ ] **산출물**: `retrieval/multi_index_search.py`

### Task 2.8 — Phase 2 평가

- [ ] 복잡 질의 골드셋 재측정
- [ ] 라우트별 retrieval 품질 분리 측정
- [ ] Latency 측정 (P50, P95, P99)
- [ ] **산출물**: `phase2_eval_report.md`

---

## Phase 3: 구조화 데이터 + Knowledge Graph

> **목적**: 결정론적으로 답할 수 있는 질의는 KG/DB로 직접 처리하여 정확도 100%에 가깝게 한다.

### Task 3.1 — Parameter/Counter/Alarm 구조화 DB 구축

- [ ] 스키마 설계
  ```
  parameters: name, mo, range, default, unit, description, applicable_releases[], applicable_products[]
  counters: name, formula, unit, related_kpis[], related_parameters[]
  alarms: code, severity, description, probable_causes[], recommended_actions[]
  ```
- [ ] Reference 문서에서 자동 추출 (LLM + 규칙 기반)
- [ ] SME 검수 워크플로우 (sample 검증 → 수정 → 재추출)
- [ ] PostgreSQL 또는 SQLite로 저장
- [ ] **산출물**: `db/schema.sql`, 추출 파이프라인, 검증된 DB

### Task 3.2 — Text-to-SQL 인터페이스

- [ ] `structured_lookup` 라우트에서 자연어 → SQL 변환
- [ ] LLM (작은 모델)로 SQL 생성, 화이트리스트 테이블/컬럼 강제
- [ ] SQL 검증 후 실행 (injection 방지)
- [ ] 결과를 LLM이 자연어로 포매팅
- [ ] **산출물**: `db/text2sql.py`

### Task 3.3 — Knowledge Graph 구축

- [ ] Neo4j 또는 유사 그래프 DB 선정
- [ ] 노드 타입: Parameter, Counter, Alarm, Feature, MO, Procedure
- [ ] 엣지 타입: depends_on, affects, triggers, requires, related_to
- [ ] 자동 추출:
  - Microsoft GraphRAG 또는 LightRAG 활용
  - LLM으로 1차 관계 추출
- [ ] SME 검수 (특히 의존성/영향 관계)
- [ ] **산출물**: KG 데이터, 추출 파이프라인, Cypher 쿼리 라이브러리

### Task 3.4 — Text-to-Cypher (KG 질의)

- [ ] `diagnostic` 라우트에서 multi-hop 추론에 활용
  - 예: "p-Max 변경 시 영향 KPI" → KG에서 affects 관계 hop
- [ ] 자연어 → Cypher 변환 LLM 프롬프트
- [ ] 결과를 RAG 컨텍스트와 결합
- [ ] **산출물**: `kg/text2cypher.py`

### Task 3.5 — GraphRAG 통합 패턴

- [ ] Local search: 관련 노드의 community 탐색
- [ ] Global search: 전체 community 요약 활용
- [ ] Hybrid: KG 결과 + vector 검색 결과 결합
- [ ] **산출물**: `kg/graphrag_pipeline.py`

### Task 3.6 — Phase 3 평가

- [ ] structured_lookup 정확도 (목표: 95%+)
- [ ] diagnostic 질의에서 KG 활용 효과 측정
- [ ] **산출물**: `phase3_eval_report.md`

---

## Phase 4: 정확성 강화 (Hallucination 통제)

> **목적**: 답변의 모든 사실을 출처에 grounding하고, hallucination을 거의 없는 수준으로 통제한다.

### Task 4.1 — Citation 강제 시스템

- [ ] 답변의 모든 사실 주장에 출처 청크 ID 매핑 강제
- [ ] LLM 프롬프트에서 citation 형식 명시
- [ ] Post-processing: citation 없는 주장 제거 또는 마킹
- [ ] UI에서 출처 hover/click 가능하게
- [ ] **산출물**: `generation/citation_enforcer.py`, 답변 형식 spec

### Task 4.2 — Self-verification 단계

- [ ] 답변 생성 후 별도 LLM 호출로 검증
- [ ] "이 답변의 각 주장이 컨텍스트에서 지지되는가?" 체크
- [ ] 지지 안 되는 부분은 제거 또는 "uncertain" 마킹
- [ ] **산출물**: `generation/self_verifier.py`

### Task 4.3 — Confidence Scoring

- [ ] 다음 신호를 종합하여 confidence 점수 산출:
  - Retrieval 점수 (BM25, dense, reranker)
  - 정답 청크의 메타데이터 매칭 정확도 (release, product)
  - KG 매칭 여부
  - Self-verification 통과율
- [ ] 임계치 미달 시 "low confidence" 마킹
- [ ] 운영자가 추가 검증할 수 있도록 UI 표시
- [ ] **산출물**: `generation/confidence_scorer.py`

### Task 4.4 — Fallback 정책 구현

- [ ] 정확 매칭 → 인접 매칭 → 유사 매칭 → "정보 없음" 단계적 폴백
- [ ] 인접/유사 매칭 시 차이점 명시 ("R6.0에는 없으나 R7.0에서는...")
- [ ] LLM 프롬프트에 명시적 의사결정 트리 삽입
- [ ] **산출물**: `generation/fallback_policy.py`, 프롬프트 템플릿

### Task 4.5 — 답변 형식 표준화

- [ ] 표준 답변 구조:
  - 직접 답변
  - 출처 (문서명, 섹션, 페이지)
  - Confidence 표시
  - 관련 정보 (related parameters/counters)
  - 주의사항 (release/product 차이 등)
- [ ] **산출물**: 답변 템플릿 라이브러리

### Task 4.6 — Phase 4 평가

- [ ] Hallucination rate 측정 (RAGAS faithfulness)
- [ ] Citation 정확도 (출처가 실제 답변을 지지하는가)
- [ ] 운영자 만족도 조사 (정성 평가)
- [ ] **산출물**: `phase4_eval_report.md`

---

## Phase 5: Agentic 확장 (선택, 향후 설정 자동화 대비)

> **목적**: 현재의 RAG 시스템을 agentic 구조로 재구성하여, 향후 EMS 연동 시 확장 용이하게 한다.

### Task 5.1 — LangGraph 기반 재구성

- [ ] 현재 파이프라인을 LangGraph 상태 머신으로 재작성
- [ ] 노드: Router, Retriever, Decomposer, KG Querier, Generator, Verifier
- [ ] 조건부 엣지로 라우팅 로직 표현
- [ ] **산출물**: `agent/graph.py`

### Task 5.2 — Iterative Retrieval (Sufficiency Check)

- [ ] 검색 결과로 답할 수 있는지 LLM이 판단
- [ ] 부족하면 추가 sub-question 생성하여 재검색
- [ ] 최대 반복 횟수 제한 (예: 3회)
- [ ] **산출물**: `agent/sufficiency_checker.py`

### Task 5.3 — Tool Calling 인터페이스 정의

- [ ] EMS API 호출, 파라미터 dry-run, MOP 실행 등을 tool로 정의
- [ ] 현재는 retrieval만 tool, 추후 확장 가능 구조
- [ ] **산출물**: `agent/tools.py`, tool spec 문서

### Task 5.4 — Human-in-the-loop

- [ ] 설정 변경 같은 critical action 시 사용자 승인 단계
- [ ] Action log 및 audit trail
- [ ] **산출물**: 승인 워크플로우 컴포넌트

---

## 인프라 및 운영 작업 (전 Phase 공통)

### Task INF-1 — LLM 서빙 인프라

- [ ] **메인 LLM 후보**: Llama 3.3 70B Instruct, Qwen 2.5 72B Instruct
- [ ] **라우터/보조 LLM**: Qwen 2.5 7B Instruct 또는 Llama 3.1 8B
- [ ] vLLM 또는 SGLang으로 서빙
- [ ] Quantization (AWQ INT4) 적용 → A100 80GB 1장 또는 L40S 2장
- [ ] **산출물**: 서빙 설정, 부하 테스트 결과

### Task INF-2 — 벡터 DB / 검색 엔진

- [x] **Milvus 채택** (2026-04-30 결정, 온프레 운영성 + 대규모 인덱스 우선) / 대안 후보: Qdrant, Weaviate
- [ ] BM25는 Elasticsearch 또는 OpenSearch
- [ ] **산출물**: 인프라 구성도

### Task INF-3 — 모니터링 및 로깅

- [ ] 모든 질의/응답 로깅 (질의, 라우트, retrieved chunks, 답변, latency)
- [ ] Langfuse 또는 자체 로깅 시스템
- [ ] 운영자 피드백 수집 (thumbs up/down)
- [ ] **산출물**: 모니터링 대시보드

### Task INF-4 — CI/CD 및 평가 자동화

- [ ] 골드셋 기반 회귀 테스트
- [ ] 변경마다 metrics 자동 측정 및 비교
- [ ] **산출물**: CI 파이프라인, 평가 리포트 자동 생성

---

## 즉시 착수 권장 (이번 주~다음 주)

지금 "복잡 질의에서 retrieval 실패"가 핵심 이슈이므로, ROI 순서로:

1. **Task 0.2 + 0.3** — 골드셋 + 평가 자동화 (개선 효과 측정 기반 마련)
2. **Task 1.5** — Reranker 도입 (가장 ROI 높은 단일 개선)
3. **Task 1.6** — 약어/동의어 사전 (저비용 고효과)
4. **Task 2.4** — Query Decomposition (복잡 질의 직접 해결책)
5. **Task 2.2** — Embedding Router (semantic-router로 1~2일)
6. **Task 1.4** — Hybrid Search 강화 (BM25 가중치 조정)

이 6개를 2-3주 내에 적용하면 복잡 질의 retrieval 품질이 큰 폭으로 개선될 가능성이 높습니다.

---

## 부록: 참고 라이브러리 / 도구

| 카테고리 | 추천 도구 |
|---|---|
| LLM 서빙 | vLLM, SGLang |
| 벡터 DB | **Milvus** (채택), Qdrant·Weaviate (대안) |
| BM25 검색 | Elasticsearch, OpenSearch |
| 임베딩 | BGE-large, E5-large, Nomic |
| Reranker | BGE-reranker-v2-m3, Jina-reranker-v2 |
| 라우터 | semantic-router, Pydantic + Instructor |
| KG | Neo4j, Microsoft GraphRAG, LightRAG |
| 오케스트레이션 | LangGraph, LlamaIndex Workflows |
| 평가 | RAGAS, TruLens, 자체 골드셋 |
| 모니터링 | Langfuse, Phoenix |
| 문서 파싱 | unstructured, pdfplumber, camelot, PaddleOCR |

---

## 부록: Claude Code 활용 가이드

이 작업 내역서를 Claude Code에서 활용할 때:

- **Phase별로 별도 작업 세션**: 한 번에 모든 Phase를 다루지 말고, 한 Phase씩 깊이 있게.
- **실제 코드/데이터 동반**: 골드셋, 실패 케이스, 현재 코드를 같이 두고 진행해야 진단 정밀도 ↑
- **반복 사이클**: 가설 → 코드 수정 → 평가 → 분석 → 다음 가설
- **체크박스 활용**: 각 Task의 체크박스를 진척 관리에 그대로 사용

---

*이 문서는 작업 진행에 따라 지속적으로 업데이트됩니다.*
