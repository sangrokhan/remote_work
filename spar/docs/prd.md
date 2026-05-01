# Samsung RAN LLM+RAG 시스템 구축 작업 내역서

> **목적**: Samsung 단일 벤더 환경(LTE+NR)에서 내부 생성 RAN 제어 파라미터, 운영 통계, 알람 코드, 작업/설치 절차서 등에 대한 자연어 질의 응답 시스템 구축
> **운영 환경**: 온프레미스, 영어 응답, 정확성 최우선 (hallucination 최소화)
> **작성일**: 2026-04-30

---

## 0. 전체 로드맵 개요

| Phase | 기간 | 핵심 목표 | 주요 산출물 |
|---|---|---|---|
| Phase 1 | 4-6주 | 데이터 파이프라인 + Retrieval 강화 | 문서 유형별 인덱스, hybrid search, reranker |
| Phase 2 | 4-6주 | 쿼리 라우터 + 복잡 질의 처리 | Router, query decomposition, multi-index |
| Phase 3 | 4-6주 | 구조화 데이터 + Knowledge Graph | Parameter/Counter/Alarm DB, KG |
| Phase 4 | 3-4주 | 정확성 강화 (Hallucination 통제) | Citation 강제, self-verification, confidence |
| Phase 5 | 선택 | Agentic 확장 (설정 자동화 대비) | LangGraph 기반 재구성 |

---

## Phase 1: 데이터 파이프라인 + Retrieval 강화

> **목적**: 문서 유형별 차별화된 처리와 hybrid retrieval로 단일 질의 retrieval 품질을 끌어올린다.

### Task 1.1 — 문서 유형 분류 및 파서 개발

> 🔧 **구현 중** (2026-05-01) — PDF→MD 변환 CLI + 3GPP TSpec fetcher 완료. 유형별 파서 미착수.

Samsung RAN 문서는 형태가 매우 다양하므로, 유형별 분리 처리가 필수.

- [ ] 문서 유형 정의 (Parameter Reference / Counter Reference / Alarm Reference / Feature Description / MOP / Installation Guide / Release Notes / **spec**(3GPP))
- [ ] 각 유형별 PDF/문서 파서 개발
  - 표 추출: `pdfplumber`, `camelot`, 또는 `unstructured` 라이브러리
  - 섹션 헤더 추출: 목차 기반 또는 폰트 크기 기반
- [x] PDF→Markdown 변환 CLI 스켈레톤 (`scripts/convert_pdf_to_md.py`, per-file 에러 격리)
- [x] 3GPP TSpec-LLM fetcher (`scripts/fetch_tspec_llm.py`, smoke test 포함)
- [ ] OCR이 필요한 스캔 문서는 별도 처리 (Tesseract, PaddleOCR)
- [ ] **산출물**: `parsers/` 디렉토리, 유형별 파서 모듈
- 추가 산출물: `scripts/convert_pdf_to_md.py`, `scripts/fetch_tspec_llm.py`

### Task 1.2 — 메타데이터 스키마 정의 및 부착

- [x] 모든 청크에 다음 메타데이터 강제 부착 (`src/spar/retrieval/milvus_client.py`):
  - `doc_type` (parameter_ref / mop / install_guide / ...)
  - `product` (LTE / NR / both)
  - `release` (예: v6.0, v7.1)
  - `deployment_type` (운영 환경/방식 구분)
  - `mo_name` (해당되는 경우)
  - `source_doc`, `section`, `page`
  - 추가(3GPP 전용): `section_num`, `section_title`, `section_depth`, `parent_sections`(ARRAY), `chunk_index`, `chunk_index_in_section`
  - 추가: `keywords`(ARRAY — 청크 내 약어 목록, max 50)
- [ ] 검색 시 메타데이터 필터로 운영 환경별 답변 분리 가능하게
- [ ] **산출물**: `chunk_schema.json`, 청킹 파이프라인

### Task 1.3 — 문서 유형별 청킹 전략 차별화

> 🔧 **구현 중** (2026-05-01) — md-aware/fixed 청커 + doc_type 디스패치 완료. Reference/MOP 전용 청커 미완.

- [ ] **Reference 문서 (Parameter/Counter/Alarm)**: 항목 단위 청크 (parameter 1개 = chunk 1개), 메타데이터에 name 명시
- [ ] **MOP/Installation**: 절차 단위 청크 (절차 헤더 + 단계들 함께), 절차가 잘리지 않도록
- [ ] **Feature Description**: 섹션 단위 청크, 부모 섹션 컨텍스트 포함
- [ ] **Release Notes**: 변경 항목 단위 청크
- [x] **spec (3GPP)**: md-aware 헤더 기반 청크 (`src/spar/ingest/chunkers.py`, code block 보존)
- [x] doc_type 디스패치 + fixed-size fallback
- [x] md 전용 ingest 파이프라인 (`scripts/run_ingest.py`, PDF 입력 거부)
- [ ] 청크 크기는 유형별로 다르게 (절대 균일 크기 적용 금지)
- [ ] **산출물**: `chunkers/` 디렉토리, 유형별 청킹 모듈
- 현 산출물: `src/spar/ingest/chunkers.py` (md-aware + fixed)
- 설계 문서: `docs/superpowers/plans/2026-05-01-md-ingest-pipeline.md`

### Task 1.4 — 인덱스 분리 및 Hybrid Search 구축

> ✅ **완료** (2026-05-01) — BM25 + Dense Hybrid Search 구현 완료. merge: `916ed9f7`

- [ ] 문서 유형별 별도 인덱스 구성 (단일 통합 인덱스는 노이즈 발생)
- [x] **BM25 인덱스**: Milvus 내장 `FunctionType.BM25` + `SPARSE_FLOAT_VECTOR` (Elasticsearch 불필요)
- [x] **Dense 인덱스**: 임베딩 모델 선정 후 벡터 DB (**Milvus** 채택, 2026-04-30 결정 / 대안: Qdrant, Weaviate)
- [x] sentence-transformers embedder wrapper (`src/spar/ingest/embedder.py`)
- [x] encoder 싱글톤 — 완료 (`src/spar/encoder/base.py` + `registry.py`; client.py/factory.py/config.py 제거, `ENCODER_MODEL`/`ENCODER_DEVICE` env vars, 계획 `docs/superpowers/plans/2026-05-01-encoder-singleton.md`)
- [x] 임베딩 모델: **`BAAI/bge-large-en-v1.5`** 채택 (2026-05-01 결정)
  - 대안 후보: `intfloat/e5-large-v2`, `nomic-embed-text-v1.5`
  - 도메인 적응 fine-tuning은 Phase 4 이후 검토
  - 모델 다운로드: `scripts/download_models.py`
- [x] Hybrid 결합: RRF (`RRFRanker`) — `hybrid_search()` in `src/spar/retrieval/milvus_client.py`
- [x] **파라미터/카운터/알람 검색은 BM25 가중치 0.5 이상** (RRF k=60 기본값으로 균등 결합)
- [x] **산출물**: `src/spar/retrieval/milvus_client.py` — `hybrid_search()` API
- [x] **산출물**: `src/spar/retrieval/routing.py` — Route→doc_type 매핑 + Milvus expr 빌더

### Task 1.5 — Reranker 도입 (Cross-encoder)

> 🔧 **구현 중** (2026-05-01) — CrossEncoderClient(remote/local) 완료, LangGraph rerank_node 연결 완료. Latency 측정 및 batch 최적화 미완.

- [x] Bi-encoder로 top-50~100 → Cross-encoder로 top-5~10 패턴 (rerank_node 연결)
- [x] Reranker 모델 선정: **`BAAI/bge-reranker-v2-m3`** 채택
  - `jinaai/jina-reranker-v2-base-multilingual` (대안 후보)
- [x] Remote CrossEncoderClient: vLLM HTTP 서빙 연동 (`src/spar/reranker/client.py`)
- [x] Local CrossEncoderClient: sentence-transformers CrossEncoder 직접 로드 (GPU 없는 환경, `asyncio.to_thread` 래핑)
- [x] RERANKER_BACKEND(local|remote) / RERANKER_DEVICE env var 지원 (`src/spar/reranker/config.py`)
- [x] EncoderFactory local/remote 분기 (`src/spar/reranker/factory.py`)
- [x] LangGraph `rerank_node` 파이프라인 연결 (`src/spar/pipeline/nodes.py`)
- [ ] Latency 측정 및 batch 최적화
- [ ] **산출물**: reranker 서빙 설정, retrieval 파이프라인 통합
- 현 산출물: `src/spar/reranker/client.py`, `config.py`, `factory.py`, `tests/reranker/` (18개 테스트)

### Task 1.6 — 도메인 약어/동의어 사전 구축

> ✅ **완료** (2026-05-01) — feat/abbrev-mapping → main merged (89d8bfe5); feat/excel-term-dict → main merged (e22bbfe5)

- [x] Samsung RAN에서 사용하는 약어 사전 작성 (HO, CA, TTT, RACH, BWP 등)
- [x] 정식 용어 ↔ 약어 ↔ 풀이 매핑
- [x] 질의 전처리에서 양방향 확장 적용
- [x] BM25 인덱스에 동의어 확장 적용 — `nodes.py:preprocess()` → `expand_query()` → `hybrid_search(query_text=expanded_query)` 연동 완료
- [x] **산출물**: `dictionary/acronyms.json`, `src/spar/preprocessing/abbrev_mapper.py`
- [x] ingest 중 문서별 Abbreviations 섹션 자동 추출 → 사전 갱신 (pre-pass)
- [x] 청크별 포함 약어 목록을 Milvus `keywords` ARRAY 필드로 기록
- [x] 노이즈 필터: trailing artifact / stylized 표기 / 밴드 패턴(CA_X 등) 제거
- [x] Excel 파일에서 특정 column 값 추출 → `dictionary/acronyms.json` `keywords` 섹션 병합 (`scripts/ingest_excel.py`)
- [x] `keywords` ARRAY max_length 32→128 (파라미터명 길이 초과 대응)
- [x] 질의 내 `keywords` 섹션 term 탐지 (`extract_terms()`) → `SparState.matched_terms` 추출
- [x] Milvus `array_contains(keywords, term)` expr 필터 (`build_expr()` + `rag_retrieve`/`decomposed_retrieve` 연동)

**구현 내용:**
- `dictionary/acronyms.json`: Rel-18 920개 파일에서 추출, 2503 entries (noise filter 적용); `keywords` 섹션 추가 (Excel ingest 대상)
- `src/spar/preprocessing/abbrev_mapper.py`: 파싱 직후 병기 확장 (`HO→HO(Handover)`), conflict는 LLM closed-set 분류, 역방향 인덱스(`expand_query`); `load_keywords()` + `extract_terms()` 추가
- `scripts/run_ingest.py`: Phase 1 pre-pass(전체 파일 약어 수집) → Phase 2 ingest(map → chunk → keywords 주입) 파이프라인; `_find_chunk_keywords` — global + keywords 섹션 통합 탐색
- `scripts/extract_acronyms.py`: 3GPP md → acronyms.json 자동 추출기 (noise filter: `_clean_expansion`, `_is_stylized_word`)
- `scripts/ingest_excel.py`: Excel column → acronyms.json `keywords` 섹션 병합 CLI (`--file`, `--columns`, `--acronyms`)
- `src/spar/ingest/excel_loader.py`: `load_excel_terms()` (openpyxl) + `merge_into_acronyms()`
- `src/spar/ingest/term_tagger.py`: `tag_chunk()` — 청크 텍스트에서 keyword 탐지 후 ARRAY 주입
- `src/spar/retrieval/milvus_client.py`: `keywords` ARRAY(VARCHAR, max_capacity=50, max_length=128) 필드
- `src/spar/retrieval/routing.py`: `build_expr()` — `matched_terms` 기반 `array_contains` OR 필터 추가
- `src/spar/pipeline/state.py`: `matched_terms: list[str]` 필드 추가
- `src/spar/pipeline/nodes.py`: `_keywords: set[str]` 필드 + `preprocess()` → `matched_terms` 추출 → `rag_retrieve`/`decomposed_retrieve` expr 필터 연동
- 테스트: 19개(기존) + excel_loader(9) + term_tagger(7) + abbrev_mapper(+8) + routing(+5) + nodes(+2) = 50개+
- 설계 문서: `docs/superpowers/specs/2026-04-30-abbrev-mapping-design.md`, `docs/superpowers/specs/2026-05-01-excel-term-dict-design.md`

### Task 1.7 — 골드셋 구축 + Phase 1 평가

> **목적**: 평가 기반을 마련하고 Phase 1 개선 효과를 정량 측정한다.

#### 1.7.1 Retrieval 골드셋 구축

- [ ] SME(현장 전문가)와 협업하여 골드셋 100~200문항 작성
- [ ] 각 질의에 대해 정답 청크 ID 또는 정답 문서/섹션 매핑
- [ ] **복잡 질의에는 여러 relevant chunk 매핑** (분산된 정답 케이스 측정 위해)
- [ ] 질의 관점별로 균형 있게 분포 (terminology / technology / behavior)
  - `terminology`: 용어·약어·개념 정의 (특정 명칭 관점)
  - `technology`: 프로토콜·아키텍처 작동 원리 (기술 관점)
  - `behavior`: 조건·이벤트별 네트워크 요소 동작 (동작 관점)
- [ ] 질문에 spec 번호 포함 ("3GPP TS {spec_number}에서 ..."), 섹션 번호 제외
- [ ] `section` 필드: ground truth용 챕터 번호 (eval 전용, 질문에 미포함)
- [ ] 메타데이터 일치 확인 (`source_doc`, `spec_number`, `release` 필드)
- [ ] **산출물**: `data/goldsets/retrieval_goldset.jsonl`
  ```jsonl
  {"query_id": "Q001", "query": "3GPP TS 29.502 Rel-18에서 SMF의 역할은?", "type": "terminology", "section": "5.2.1", "source_doc": "29502-i40.md", "spec_number": "29.502", "release": "Rel-18", "answer": "..."}
  ```

#### 1.7.2 평가 자동화 스크립트

> 🔧 **구현 중** (2026-05-01) — Recall@K/MRR CLI 완료. RAGAS 방식 답변 품질 평가 완료. CI/CD 미착수.

- [x] Recall@5, Recall@10, Recall@50, MRR 측정 (`src/spar/eval/metrics.py`)
- [x] 답변 품질 평가 — faithfulness, answer_relevancy (`src/spar/eval/ragas_metrics.py`)
- [x] 질의 유형별 분리 측정 (by_type breakdown in `compute_metrics`)
- [ ] CI/CD 형태로 변경마다 자동 재측정
- [x] eval_suite.py 멀티-config 비교 러너 (`src/spar/eval/eval_suite.py`) — GraphConfig ablation, Recall@K/MRR/faithfulness 비교표
- [x] graph.ainvoke() 기반 eval — `run_eval.py` 직접 Milvus 호출 제거, `build_graph()` 위임
- [x] **산출물**: `src/spar/eval/run_eval.py` ✅, `src/spar/eval/metrics_dashboard.md` ✅, `src/spar/eval/ragas_metrics.py` ✅, `src/spar/eval/run_ragas_eval.py` ✅, `src/spar/eval/eval_suite.py` ✅

#### 1.7.3 Phase 1 측정 및 실패 케이스 분석

- [ ] 골드셋으로 Recall@5/10/50, MRR 측정
- [ ] 개선 폭 보고 (구성요소별 ablation: hybrid 단독 / +reranker / +사전)
- [ ] Recall@10 실패 케이스 50개 샘플링
- [ ] 실패 원인 분류 (vocabulary mismatch / decomposition needed / multi-index needed / ambiguous query / chunking issue)
- [ ] 우선순위 매트릭스 작성 (영향도 × 난이도) → Phase 2 입력
- [ ] **산출물**: `phase1_eval_report.md`, `failure_analysis_report.md`

---

## Phase 2: 쿼리 라우터 + 복잡 질의 처리

> **목적**: 복잡한/다측면 질의를 분해하고, 질의 유형별로 다른 검색 전략을 적용한다.

### Task 2.1 — 라우트 정의 (도메인 특화)

> ✅ **완료** (2026-05-01) — 6개 라우트 schemas.py 정의, routes_spec.md 작성

- [x] 다음 라우트로 분류 (`src/spar/router/schemas.py`):
  - `structured_lookup` — 파라미터/카운터/알람 정확 조회 → KG/DB 직접
  - `definition_explain` — 서술형 설명 → RAG + structured 보조
  - `procedural` — 절차/설치/설정 방법 → MOP/Install 인덱스
  - `diagnostic` — 트러블슈팅 → multi-hop, KG 활용
  - `comparative` — 버전/기능 비교 → Release Notes 인덱스
  - `default_rag` — 기타 → 전체 hybrid 검색
- [x] 각 라우트별 정의 및 판단 기준 문서화
- [x] **산출물**: `src/spar/router/routes_spec.md`

### Task 2.2 — Hybrid 라우터 구현 (3-layer)

> 🔧 **구현 중** (2026-05-01) — 3개 레이어 파일 + hybrid + schemas 생성됨. 튜닝/골드셋 평가 미완.
> 🔧 **부분 완료** (2026-05-01) — 3GPP TS spec number 패턴 추가 (`TS \d{2}.\d{3}` → `DEFINITION_EXPLAIN` + `spec_number` entity). spec intro ingest 파이프라인(`slice_3gpp_intros.py`, `run_ingest.py --intro-only`) 구현.

#### Layer 1: Regex Fast-path

- [x] 알람 코드 패턴 (`ALM-\d+`, `alarm \d+`) → `STRUCTURED_LOOKUP`
- [x] 파라미터명 패턴 (camelCase + 기술 접미사: `maxTxPower`, `tReordering` 등) → `STRUCTURED_LOOKUP`
- [x] MO 이름 매칭 (`NRCellDU`, `EUtranCellFDD` 등) → `STRUCTURED_LOOKUP`
- [x] 3GPP TS 문서번호 패턴 (`TS 29.502`, `3GPP TS 38.300`, `TS29502`) → `DEFINITION_EXPLAIN` + `spec_number` entity
- [x] Hit 시 → 해당 라우트로 직행 (confidence 0.9)
- [x] **산출물**: `router/regex_router.py`
- [x] **추가 산출물**: `scripts/slice_3gpp_intros.py` (spec intro 1000줄 추출), `run_ingest.py --intro-only` (spec_number dynamic field 부착)

#### Layer 2: Embedding Similarity Router

- [ ] `semantic-router` 라이브러리 활용
- [ ] 라우트별 예시 10~30개씩 작성
- [ ] Threshold 튜닝 (초기 0.7)
- [ ] Confidence 낮으면 Layer 3로
- [x] **산출물**: `router/embedding_router.py`
- [ ] `router/route_examples.yaml` ← 미작성

#### Layer 3: LLM Router (Function Calling)

- [ ] 작은 모델 사용 (Qwen 2.5 7B Instruct 또는 Llama 3.1 8B Instruct)
- [ ] vLLM 별도 인스턴스로 서빙
- [ ] **라우팅 + entity extraction을 한 번에**
  - route, entities (parameter/counter/alarm names), product, release
- [x] Pydantic schema로 structured output 강제 (`router/schemas.py`)
- [x] **산출물**: `router/llm_router.py`
- [x] 라우터 시스템 프롬프트 → `src/spar/prompts/router_system.txt`

**추가 산출물**: `router/hybrid_router.py` (3-layer 통합 오케스트레이터)
**추가 산출물**: `src/spar/prompts/` — LLM 프롬프트 중앙화 패키지 (router_system.txt, abbrev_conflict.txt)

### Task 2.3 — 라우터 골드셋 및 평가

- [x] 라우터 전용 골드셋 (QA goldset type→route 매핑으로 ~300개) — `scripts/gen_router_goldset.py`
- [x] Confusion matrix 분석 — `scripts/run_router_eval.py` 출력에 포함
- [x] 라우팅 정확도 측정 (overall + per-route) — `scripts/run_router_eval.py`
- [ ] **산출물**: `data/goldsets/router_goldset.jsonl` ← gen_router_goldset.py 실행 후 생성
- [ ] **산출물**: `data/eval_results/router_eval_{date}.md` ← run_router_eval.py 실행 후 생성
- [x] **산출물**: `scripts/gen_router_goldset.py`
- [x] **산출물**: `scripts/run_router_eval.py`

### Task 2.4 — Query Decomposition (복잡 질의 분해)

> **다측면 질의 retrieval 실패의 주요 해결책**

- [x] 라우터에서 "complex" 판정 시 decomposition 활성화 (`RouteResult.needs_decomposition`, `LLMRouter` 판정)
- [x] LLM 프롬프트로 sub-question 리스트 생성 (`prompts/decompose_system.txt`, `QueryDecomposer`)
- [x] 각 sub-question으로 개별 검색 (`pipeline/decomposed_retrieve` 노드 — 병렬 검색)
- [x] 결과 union → reranker로 통합 top-k 선정 (chunk dedup + merge, `rerank` 노드로 연결)
- [x] **산출물**: `src/spar/retrieval/query_decomposer.py`
- **완료**: 2026-05-01, merge commit: bc9ac0ef

### Task 2.5 — Query Rewriting (대화 컨텍스트 반영)

- [ ] 멀티턴 대화에서 질의를 self-contained 형태로 rewrite (LLM rewrite 미구현 — history context 전달로 대체)
- [x] 약어 자동 확장 — dictionary 기반 관련 약어 추출 → LLM 프롬프트 context 제공
- [ ] 모호한 지시어 명시화 ("그 파라미터" → 실제 이름)
- [x] **산출물**: `src/spar/retrieval/query_rewriter.py` — `build_context()`: history(최대 5턴) + 약어 context 준비
- [x] **부가 산출물**: `QueryRequest.history`, `SparState.history/history_context`, `pipeline/prepare_context` 노드

### Task 2.6 — HyDE / Multi-Query Expansion (선택적)

- [ ] HyDE: 가상 답변 생성 후 임베딩 검색
- [ ] Multi-Query: 3~4개 alternative phrasing 생성 후 RRF 결합
- [ ] A/B 테스트로 효과 확인 후 적용 여부 결정
- [ ] **산출물**: `retrieval/hyde.py`, `retrieval/multi_query.py`
- 설계 문서: `docs/superpowers/plans/2026-05-01-hyde-multi-query.md`

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

> 🔧 **조기 도입** (2026-05-01) — StateGraph scaffold + conditional edges + reranker 연결 완료. 고급 노드(Decomposer, KG Querier, Verifier) 미착수.

- [x] 현재 파이프라인을 LangGraph StateGraph로 재작성 (`src/spar/pipeline/` — `SparState`, `Nodes`, `build_graph()`)
- [x] 조건부 엣지로 라우팅 로직 표현 (STRUCTURED_LOOKUP → structured_retrieve, DIAGNOSTIC → multi_hop_retrieve, 나머지 → rag_retrieve)
- [x] CrossEncoderClient reranker 파이프라인 첫 연결 (`rerank_node`)
- [ ] 노드 확장: Decomposer (query decomposition), KG Querier (Phase 3 연동), Verifier (Phase 4 연동)
- [x] node_timings SparState 추가 — 노드별 실행 시간(ms) 기록 (`src/spar/pipeline/state.py`)
- [x] 실제 LLM generate 노드 연결 — `LLMClient` 주입 시 실 답변, 없으면 stub (`src/spar/pipeline/nodes.py`)
- [x] GraphConfig + PRESET_CONFIGS — baseline/+reranker/+qexpand/+context/full_retrieval/e2e ablation 프리셋 (`src/spar/pipeline/config.py`)
- [x] **산출물**: `src/spar/pipeline/graph.py`, `src/spar/pipeline/nodes.py`, `src/spar/pipeline/state.py` — Milvus hybrid_search 연결 완료 (stub 제거), `src/spar/pipeline/config.py` ✅
- 설계 문서: `docs/superpowers/plans/2026-05-01-langgraph-pipeline.md`

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
- [x] vLLM 서빙 스크립트 작성 (`scripts/serve_vllm.sh`)
- [x] LLM 팩토리/싱글톤/레지스트리 구현 (`src/spar/llm/` — client, config, factory, registry)
- [x] FastAPI 앱 기반 구축 (`src/spar/api/app.py`, `scripts/test_api.py`)
- [ ] Quantization (AWQ INT4) 적용 → A100 80GB 1장 또는 L40S 2장
- [ ] **산출물**: 서빙 설정, 부하 테스트 결과

### Task INF-1b — Codex 서브에이전트 + Gemini Fallback 훅

> ✅ **완료** (2026-05-01) — feat/gemini-fallback 브랜치

- [x] SubagentStart → Codex CLI headless 라우팅 (`3e3b3010`)
- [x] codex exec 프롬프트 stdin 파일 주입 (ARG_MAX 회피, `3c4ebea7`)
- [x] Codex 토큰 소진 시 Gemini CLI fallback (`41c0a565`)
- [x] **산출물**: Claude Code hooks 설정, 계획 문서 `docs/superpowers/plans/2026-05-01-codex-subagent-workflow.md`

### Task INF-2 — 벡터 DB / 검색 엔진

- [x] **Milvus 채택** (2026-04-30 결정, 온프레 운영성 + 대규모 인덱스 우선) / 대안 후보: Qdrant, Weaviate
- [x] Milvus 클라이언트 구현 (`src/spar/retrieval/milvus_client.py`)
- [x] Milvus 설정 파일 작성 (`configs/milvus/`)
- [x] Milvus 초기화 스크립트 (`scripts/init_milvus.py`)
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

1. **Task 1.5** — Reranker 도입 (가장 ROI 높은 단일 개선)
2. **Task 1.6** — 약어/동의어 사전 (저비용 고효과)
3. **Task 2.4** — Query Decomposition (복잡 질의 직접 해결책)
4. **Task 2.2** — Embedding Router (semantic-router로 1~2일)
5. **Task 1.4** — Hybrid Search 강화 (BM25 가중치 조정)

위 5개를 2-3주 내에 적용하면 복잡 질의 retrieval 품질이 큰 폭으로 개선될 가능성이 높습니다.

---

## 부록: 참고 라이브러리 / 도구

| 카테고리 | 추천 도구 |
|---|---|
| LLM 서빙 | vLLM, SGLang |
| 벡터 DB | **Milvus** (채택), Qdrant·Weaviate (대안) |
| BM25 검색 | Milvus 내장 FunctionType.BM25 (sparse vector) |
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
