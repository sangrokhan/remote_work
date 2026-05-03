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

> 🔧 **구현 중**

Samsung RAN 문서는 형태가 매우 다양하므로, 유형별 분리 처리가 필수.

- [ ] 문서 유형 정의 (Parameter Reference / Counter Reference / Alarm Reference / Feature Description / MOP / Installation Guide / Release Notes / spec(3GPP))
- [x] 각 유형별 파서 개발
  - [x] Word .docx — `DocxParser` (heading 스타일 기반 섹션, 표→CSV, 이미지 추출)
  - [x] PDF — 폰트 메트릭 기반 헤딩 검출, 크로스페이지 테이블, 임베디드 이미지/드로잉
  - [x] Counter Reference Excel — 병합 셀 전파, value_range 분리, 한/영 헤더 별칭
  - [x] Alarm Reference Excel — alarm_id/severity/category/module/pdf_ref
- [x] PDF→Markdown 변환 CLI
- [x] 3GPP TSpec-LLM fetcher
- [ ] OCR 스캔 문서 처리 (Tesseract, PaddleOCR)

**산출물**:
- [x] Word .docx 파서
- [x] PDF 구조 추출 패키지 (CLI 포함)
- [x] Counter Reference Excel 파서 + ingest CLI + 샘플 데이터
- [x] Alarm Reference Excel 파서 + Alarm Index (싱글톤, lookup API) + 샘플 데이터
- [x] PDF→Markdown 변환 CLI
- [x] 3GPP TSpec fetcher
- [x] Regex 라우터 (counter/alarm/MO 패턴)
- [x] 파서 테스트 스위트

**의존 라이브러리**: `pdfplumber`, `pypdf`, `Pillow`

### Task 1.2 — 메타데이터 스키마 정의 및 부착

- [x] 모든 청크에 메타데이터 강제 부착:
  - `doc_type` (parameter_ref / mop / install_guide / ...)
  - `product` (LTE / NR / both)
  - `release` (예: v6.0, v7.1)
  - `deployment_type`
  - `mo_name`
  - `source_doc`, `section`, `page`
  - 3GPP 전용: `section_num`, `section_title`, `section_depth`, `parent_sections`(ARRAY), `chunk_index`, `chunk_index_in_section`
  - `keywords` (ARRAY — 청크 내 약어 목록, max 50)
- [ ] 검색 시 메타데이터 필터로 운영 환경별 답변 분리

**산출물**:
- [ ] 청크 스키마 JSON
- [ ] 청킹 파이프라인

### Task 1.3 — 문서 유형별 청킹 전략 차별화

> 🔧 **구현 중**

- [x] **Reference 문서 (Parameter/Counter/Alarm)**: 항목 단위 청크 (parameter 1개 = chunk 1개)
- [ ] **MOP/Installation**: 절차 단위 청크 (절차가 잘리지 않도록)
- [ ] **Feature Description**: 섹션 단위 청크, 부모 섹션 컨텍스트 포함
- [ ] **Release Notes**: 변경 항목 단위 청크
- [x] **spec (3GPP)**: md-aware 헤더 기반 청크 (code block 보존)
- [x] doc_type 디스패치 + fixed-size fallback
- [x] md 전용 ingest 파이프라인
- [ ] 청크 크기 유형별 차별화 (균일 크기 적용 금지)

**산출물**:
- [x] md-aware + fixed 청커
- [x] ingest 스크립트
- [x] 유형별 청커 모듈 (`reference_chunker.py` — Parameter/Counter/Alarm)
- [x] Excel ingest 경로 (`run_ingest.py --input-file *.xlsx`)

### Task 1.4 — 인덱스 분리 및 Hybrid Search 구축

> ✅ **완료**

- [ ] 문서 유형별 별도 인덱스 구성
- [x] BM25 인덱스: Milvus 내장 `FunctionType.BM25` + `SPARSE_FLOAT_VECTOR`
- [x] Dense 인덱스: Milvus + `BAAI/bge-large-en-v1.5`
- [x] sentence-transformers embedder wrapper
- [x] encoder 싱글톤 (`ENCODER_MODEL`/`ENCODER_DEVICE` env vars)
- [x] Hybrid 결합: RRF (`RRFRanker`)
- [x] 파라미터/카운터/알람 검색 BM25 가중치 0.5 이상

**산출물**:
- [x] Milvus 클라이언트 (`hybrid_search()` API)
- [x] 라우팅 모듈 (Route→doc_type 매핑 + Milvus expr 빌더)
- [x] Embedder wrapper + Encoder 싱글톤
- [x] 모델 다운로드 스크립트

### Task 1.5 — Reranker 도입 (Cross-encoder)

> 🔧 **구현 중**

- [x] Bi-encoder top-50~100 → Cross-encoder top-5~10 (rerank_node 연결)
- [x] Reranker 모델: `BAAI/bge-reranker-v2-m3`
- [x] Remote CrossEncoderClient: vLLM HTTP 서빙 연동
- [x] Local CrossEncoderClient: sentence-transformers 직접 로드 (`asyncio.to_thread`)
- [x] `RERANKER_BACKEND`(local|remote) / `RERANKER_DEVICE` env var 지원
- [x] LangGraph `rerank_node` 파이프라인 연결
- [ ] Latency 측정 및 batch 최적화

**산출물**:
- [x] CrossEncoder 클라이언트 (remote/local 백엔드)
- [x] Reranker 설정 + 팩토리
- [x] Reranker 테스트 스위트
- [ ] Reranker 서빙 설정

### Task 1.6 — 도메인 약어/동의어 사전 구축

> ✅ **완료**

- [x] Samsung RAN 약어 사전 작성 (HO, CA, TTT, RACH, BWP 등)
- [x] 정식 용어 ↔ 약어 ↔ 풀이 매핑
- [x] 질의 전처리에서 양방향 확장 적용
- [x] BM25 인덱스에 동의어 확장 적용
- [x] ingest 중 문서별 Abbreviations 섹션 자동 추출 → 사전 갱신
- [x] 청크별 포함 약어 목록을 Milvus `keywords` ARRAY 필드로 기록
- [x] Excel 파일 특정 column 값 추출 → `keywords` 섹션 병합
- [x] 질의 내 `keywords` term 탐지 → `SparState.matched_terms`
- [x] Milvus `array_contains(keywords, term)` expr 필터

**산출물**:
- [x] 약어 사전 JSON (2503 entries)
- [x] 약어 매퍼 (양방향 확장, LLM 충돌 해소)
- [x] Excel 로더 + 텀 태거
- [x] 약어 추출 스크립트 (3GPP md 대상)
- [x] Excel ingest 스크립트
- [x] Samsung 도메인 엔티티 사전 (`dictionary/samsung_entities.json` — 파라미터/카운터/알람)
- [x] `build_entity_glossary.py` — Pass A pre-scan 스크립트 (Excel → entity glossary)
- [x] `get_all_keywords()` — 3GPP 약어 + Samsung 엔티티 통합 keyword set (tag_keywords 실동작)

### Task 1.7 — 골드셋 구축 + Phase 1 평가

#### 1.7.1 Retrieval 골드셋 구축

- [x] `gen_goldset.py`로 골드셋 생성 (11,107 문항, 7개 QA 타입, 6개 라우트 커버)
- [x] 각 질의에 정답 문서/섹션 매핑 (`source_doc`, `section` 필드)
- [x] 질의 유형별 분포 (`type` 필드)
- [x] 라우터 레이블 포함 (`expected_route`, `needs_decomposition`)
- [ ] SME 검토 및 품질 검증

**산출물**:
- [x] Retrieval 골드셋 JSONL (`data/goldsets/goldset.jsonl` — 11,107 문항)

#### 1.7.2 평가 자동화 스크립트

> 🔧 **구현 중**

- [x] Recall@5/10/50, MRR 측정
- [x] 답변 품질 평가 — faithfulness, answer_relevancy
- [x] 질의 유형별 분리 측정
- [x] 멀티-config 비교 러너 (GraphConfig ablation)
- [x] graph.ainvoke() 기반 eval
- [x] E2E 평가 — goldset → pipeline(LLM generate) → RAGAS 지표 한 번에 (run_e2e_eval.py)
- [ ] CI/CD 형태로 변경마다 자동 재측정

**산출물**:
- [x] Retrieval 메트릭 모듈 (Recall@K, MRR)
- [x] RAGAS 메트릭 모듈
- [x] 평가 러너 + 멀티-config 비교 러너
- [x] E2E 평가 러너 (`eval/run_e2e_eval.py`)
- [x] 메트릭 대시보드

#### 1.7.3 Phase 1 측정 및 실패 케이스 분석

- [ ] 골드셋으로 Recall@5/10/50, MRR 측정
- [ ] 구성요소별 ablation (hybrid 단독 / +reranker / +사전)
- [ ] Recall@10 실패 케이스 50개 샘플링
- [ ] 실패 원인 분류 (vocabulary mismatch / decomposition needed / multi-index needed / ambiguous query / chunking issue)
- [ ] 우선순위 매트릭스 작성 → Phase 2 입력

**산출물**:
- [ ] Phase 1 평가 리포트
- [ ] 실패 케이스 분석 리포트

---

## Phase 2: 쿼리 라우터 + 복잡 질의 처리

> **목적**: 복잡한/다측면 질의를 분해하고, 질의 유형별로 다른 검색 전략을 적용한다.

### Task 2.1 — 라우트 정의 (도메인 특화)

> ✅ **완료**

- [x] 라우트 정의:
  - `structured_lookup` — 파라미터/카운터/알람 정확 조회
  - `definition_explain` — 서술형 설명
  - `procedural` — 절차/설치/설정 방법
  - `diagnostic` — 트러블슈팅
  - `comparative` — 버전/기능 비교
  - `default_rag` — 기타
- [x] 각 라우트별 판단 기준 문서화

**산출물**:
- [x] 라우트 스키마 (6개 라우트 Pydantic 모델)
- [x] 라우트 판단 기준 문서

### Task 2.2 — Hybrid 라우터 구현 (3-layer)

> 🔧 **구현 중**

#### Layer 1: Regex Fast-path

- [x] 알람 코드 패턴 → `STRUCTURED_LOOKUP`
- [x] 파라미터명 패턴 (camelCase + 기술 접미사) → `STRUCTURED_LOOKUP`
- [x] MO 이름 매칭 → `STRUCTURED_LOOKUP`
- [x] 3GPP TS 문서번호 패턴 → `DEFINITION_EXPLAIN` + `spec_number` entity
- [x] Hit 시 해당 라우트로 직행 (confidence 0.9)

**산출물**:
- [x] Regex 라우터
- [x] 3GPP spec intro 추출 스크립트

#### Layer 2: Embedding Similarity Router

- [ ] `semantic-router` 라이브러리 활용
- [ ] 라우트별 예시 10~30개씩 작성
- [ ] Threshold 튜닝 (초기 0.7)
- [ ] Confidence 낮으면 Layer 3로

**산출물**:
- [x] Embedding 라우터 (스켈레톤)
- [ ] 라우트 예시 YAML

#### Layer 3: LLM Router (Function Calling)

- [ ] 소형 모델 사용 (Qwen 2.5 7B Instruct 또는 Llama 3.1 8B)
- [ ] 라우팅 + entity extraction 한 번에 (route, entities, product, release)
- [x] Pydantic schema로 structured output 강제

**산출물**:
- [x] LLM 라우터
- [x] Hybrid 라우터 (3-layer 통합 오케스트레이터)
- [x] 라우터 시스템 프롬프트

### Task 2.3 — 라우터 골드셋 및 평가

- [x] 라우터 전용 골드셋 (7개 QA 타입 → 6개 route 매핑)
- [x] Confusion matrix 분석
- [x] 라우팅 정확도 측정 (overall + per-route)

**산출물**:
- [ ] 라우터 골드셋 JSONL
- [ ] 라우터 평가 결과 리포트
- [x] 골드셋 생성 스크립트 (Codex→Gemini fallback)
- [x] 라우터 평가 스크립트

### Task 2.4 — Query Decomposition (복잡 질의 분해)

> ✅ **완료**

- [x] 라우터에서 "complex" 판정 시 decomposition 활성화
- [x] LLM 프롬프트로 sub-question 리스트 생성
- [x] 각 sub-question으로 개별 검색 (병렬)
- [x] 결과 union → reranker로 통합 top-k 선정

**산출물**:
- [x] Query Decomposer
- [x] Decompose 시스템 프롬프트

### Task 2.5 — Query Rewriting (대화 컨텍스트 반영)

> ✅ **완료**

- [x] 멀티턴 대화에서 질의를 self-contained 형태로 rewrite
- [x] 약어 자동 확장 (dictionary 기반)
- [x] 모호한 지시어 명시화

**산출물**:
- [x] Query Rewriter + 시스템 프롬프트
- [x] Query Rewriter 테스트

### Task 2.6 — Hybrid Verify Loop

> ✅ **완료**

- [x] Verify node: LLM self-evaluation score (1-5)
- [x] Tool_call node: 결정론적 fallback 전략 (decomposed → multi_hop → structured → rag)
- [x] LangGraph conditional routing 연결
- [x] `use_verify_loop=True` GraphConfig preset 활성화
- [x] 최대 3회 retry + backoff
- [x] LLM score threshold: < 3 → fallback 전략 선택

**산출물**:
- [x] Verify 노드 + Tool_call 노드
- [x] Verify/Tool_call 프롬프트
- [x] Verify 노드 테스트 스위트 (59개)

### Task 2.8 — HyDE / Multi-Query Expansion (선택적)

- [ ] HyDE: 가상 답변 생성 후 임베딩 검색
- [ ] Multi-Query: 3~4개 alternative phrasing 생성 후 RRF 결합
- [ ] A/B 테스트로 효과 확인 후 적용 여부 결정

**산출물**:
- [ ] HyDE 모듈
- [ ] Multi-Query 모듈

### Task 2.9 — Multi-Index 병렬 검색

- [ ] 라우트에 따라 여러 인덱스 동시 검색
- [ ] 각 인덱스에서 top-N → 통합 → reranker
- [ ] 진단 질의의 경우 KG hop 결과도 결합

**산출물**:
- [ ] Multi-Index 검색 모듈

### Task 2.10 — Phase 2 평가

- [ ] 복잡 질의 골드셋 재측정
- [ ] 라우트별 retrieval 품질 분리 측정
- [ ] Latency 측정 (P50, P95, P99)

**산출물**:
- [ ] Phase 2 평가 리포트

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
- [ ] SME 검수 워크플로우
- [ ] PostgreSQL 또는 SQLite로 저장

**산출물**:
- [ ] DB 스키마
- [ ] 추출 파이프라인
- [ ] 검증된 DB

### Task 3.2 — Text-to-SQL 인터페이스

- [ ] `structured_lookup` 라우트에서 자연어 → SQL 변환
- [ ] 화이트리스트 테이블/컬럼 강제
- [ ] SQL 검증 후 실행 (injection 방지)
- [ ] 결과를 LLM이 자연어로 포매팅

**산출물**:
- [ ] Text-to-SQL 모듈

### Task 3.3 — Knowledge Graph 구축

- [ ] Neo4j 또는 유사 그래프 DB 선정
- [ ] 노드 타입: Parameter, Counter, Alarm, Feature, MO, Procedure
- [ ] 엣지 타입: depends_on, affects, triggers, requires, related_to
- [ ] LLM 기반 관계 자동 추출 (Microsoft GraphRAG 또는 LightRAG)
- [ ] SME 검수

**산출물**:
- [ ] KG 데이터
- [ ] 추출 파이프라인
- [ ] Cypher 쿼리 라이브러리

### Task 3.4 — Text-to-Cypher (KG 질의)

- [ ] `diagnostic` 라우트에서 multi-hop 추론 활용
- [ ] 자연어 → Cypher 변환 LLM 프롬프트
- [ ] 결과를 RAG 컨텍스트와 결합

**산출물**:
- [ ] Text-to-Cypher 모듈

### Task 3.5 — GraphRAG 통합 패턴

- [ ] Local search: 관련 노드 community 탐색
- [ ] Global search: 전체 community 요약 활용
- [ ] Hybrid: KG 결과 + vector 검색 결과 결합

**산출물**:
- [ ] GraphRAG 파이프라인 모듈

### Task 3.6 — Phase 3 평가

- [ ] structured_lookup 정확도 (목표: 95%+)
- [ ] diagnostic 질의에서 KG 활용 효과 측정

**산출물**:
- [ ] Phase 3 평가 리포트

---

## Phase 4: 정확성 강화 (Hallucination 통제)

> **목적**: 답변의 모든 사실을 출처에 grounding하고, hallucination을 거의 없는 수준으로 통제한다.

### Task 4.1 — Citation 강제 시스템

- [ ] 답변의 모든 사실 주장에 출처 청크 ID 매핑 강제
- [ ] LLM 프롬프트에서 citation 형식 명시
- [ ] Post-processing: citation 없는 주장 제거 또는 마킹
- [ ] UI에서 출처 hover/click 가능하게

**산출물**:
- [ ] Citation 강제 모듈
- [ ] 답변 형식 spec

### Task 4.2 — Self-verification 단계

- [ ] 답변 생성 후 별도 LLM 호출로 검증
- [ ] 지지 안 되는 부분은 제거 또는 "uncertain" 마킹

**산출물**:
- [ ] Self-verifier 모듈

### Task 4.3 — Confidence Scoring

- [ ] Retrieval 점수, 메타데이터 매칭, KG 매칭, self-verification 통과율 종합
- [ ] 임계치 미달 시 "low confidence" 마킹

**산출물**:
- [ ] Confidence scorer 모듈

### Task 4.4 — Fallback 정책 구현

- [ ] 정확 매칭 → 인접 매칭 → 유사 매칭 → "정보 없음" 단계적 폴백
- [ ] 인접/유사 매칭 시 차이점 명시

**산출물**:
- [ ] Fallback 정책 모듈
- [ ] 프롬프트 템플릿

### Task 4.5 — 답변 형식 표준화

- [ ] 표준 답변 구조: 직접 답변 / 출처 / Confidence / 관련 정보 / 주의사항

**산출물**:
- [ ] 답변 템플릿 라이브러리

### Task 4.6 — Phase 4 평가

- [ ] Hallucination rate 측정 (RAGAS faithfulness)
- [ ] Citation 정확도

**산출물**:
- [ ] Phase 4 평가 리포트

---

## Phase 5: Agentic 확장 (선택, 향후 설정 자동화 대비)

> **목적**: 현재의 RAG 시스템을 agentic 구조로 재구성하여, 향후 EMS 연동 시 확장 용이하게 한다.

### Task 5.1 — LangGraph 기반 재구성

> 🔧 **조기 도입**

- [x] 파이프라인을 LangGraph StateGraph로 재작성
- [x] 조건부 엣지로 라우팅 로직 표현
- [x] CrossEncoderClient reranker 파이프라인 연결
- [x] GraphConfig + PRESET_CONFIGS (ablation 프리셋)
- [x] node_timings SparState 추가
- [x] 실제 LLM generate 노드 연결
- [ ] 노드 확장: Decomposer, KG Querier, Verifier

**산출물**:
- [x] LangGraph StateGraph 파이프라인 (graph, nodes, state)
- [x] GraphConfig + 프리셋

### Task 5.2 — Iterative Retrieval (Sufficiency Check)

- [ ] 검색 결과로 답할 수 있는지 LLM 판단
- [ ] 부족 시 추가 sub-question 생성 후 재검색
- [ ] 최대 반복 횟수 제한 (예: 3회)

**산출물**:
- [ ] Sufficiency checker 모듈

### Task 5.3 — Tool Calling 인터페이스 정의

- [ ] EMS API 호출, 파라미터 dry-run, MOP 실행 등을 tool로 정의
- [ ] retrieval만 tool, 추후 확장 가능 구조

**산출물**:
- [ ] Tool 인터페이스 모듈
- [ ] Tool spec 문서

### Task 5.4 — Human-in-the-loop

- [ ] Critical action 시 사용자 승인 단계
- [ ] Action log 및 audit trail

**산출물**:
- [ ] 승인 워크플로우 컴포넌트

---

## 인프라 및 운영 작업 (전 Phase 공통)

### Task INF-1 — LLM 서빙 인프라

- [ ] **메인 LLM**: Llama 3.3 70B Instruct 또는 Qwen 2.5 72B Instruct
- [ ] **라우터/보조 LLM**: Qwen 2.5 7B Instruct 또는 Llama 3.1 8B
- [x] vLLM 서빙 스크립트
- [x] LLM 팩토리/싱글톤/레지스트리
- [x] FastAPI 앱 기반 구축
- [ ] Quantization (AWQ INT4) 적용

**산출물**:
- [x] vLLM 서빙 스크립트
- [x] LLM 클라이언트 모듈 (팩토리/싱글톤/레지스트리)
- [x] FastAPI 앱
- [ ] 서빙 설정, 부하 테스트 결과

### Task INF-1b — Codex 서브에이전트 + Gemini Fallback 훅

> ✅ **완료**

- [x] SubagentStart → Codex CLI headless 라우팅
- [x] codex exec 프롬프트 stdin 파일 주입 (ARG_MAX 회피)
- [x] Codex 토큰 소진 시 Gemini CLI fallback

**산출물**:
- [x] Claude Code hooks 설정

### Task INF-2 — 벡터 DB / 검색 엔진

- [x] **Milvus 채택** (온프레미스 운영성 + 대규모 인덱스 우선)
- [x] Milvus 클라이언트
- [x] Milvus 설정 파일
- [x] Milvus 초기화 스크립트

**산출물**:
- [x] Milvus 클라이언트 + 설정 + 초기화 스크립트
- [ ] 인프라 구성도

### Task INF-3 — 모니터링 및 로깅

- [ ] 모든 질의/응답 로깅 (질의, 라우트, retrieved chunks, 답변, latency)
- [ ] Langfuse 또는 자체 로깅 시스템
- [ ] 운영자 피드백 수집 (thumbs up/down)

**산출물**:
- [ ] 모니터링 대시보드

### Task INF-4 — CI/CD 및 평가 자동화

- [ ] 골드셋 기반 회귀 테스트
- [ ] 변경마다 metrics 자동 측정 및 비교

**산출물**:
- [ ] CI 파이프라인
- [ ] 평가 리포트 자동 생성

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

*이 문서는 requirements + 산출물 체크리스트만 관리. 구현 이력은 `docs/work_log/` 참조.*
