# SPAR

> **S**amsung RAN **P**arameter **A**ssistant for **R**AG
> Samsung 단일 벤더(LTE+NR) 환경의 내부 RAN 문서에 대한 자연어 질의응답 시스템

---

## 개요

SPAR는 Samsung RAN 운영 환경에서 다음 문서군에 대한 자연어 질의를 정확하게 처리하기 위한 LLM+RAG 시스템입니다.

- Parameter / Counter / Alarm Reference
- Feature Description
- MOP (Method of Procedure) / Installation Guide
- Release Notes

**핵심 원칙**: 온프레미스 운영, 영어 응답, 정확성 최우선 (hallucination 최소화)

---

## 로드맵

| Phase | 기간 | 목표 |
|---|---|---|
| Phase 0 | 1-2주 | 현황 진단 및 Baseline 측정 |
| Phase 1 | 4-6주 | 데이터 파이프라인 + Retrieval 강화 |
| Phase 2 | 4-6주 | 쿼리 라우터 + 복잡 질의 처리 |
| Phase 3 | 4-6주 | 구조화 데이터 + Knowledge Graph |
| Phase 4 | 3-4주 | 정확성 강화 (Hallucination 통제) |
| Phase 5 | 선택 | Agentic 확장 |

상세: [`docs/prd.md`](docs/prd.md)

---

## 빠른 시작

요구 사항: Python 3.12

```bash
# 1. 가상환경 + 개발 의존성 설치
make install-dev

# 1.1 vLLM 임베딩/서빙을 사용할 예정이면 별도 설치
# macOS Apple Silicon은 vllm-metal 경로로 자동 분기
make install-vllm

# 2. 환경 변수 설정
cp .env.example .env
# .env 편집

# 3. 검증
make lint
make test
```

### 모델 다운로드 / 임베딩 서비스 연동

```bash
# 1) 임베딩(BAAI/bge-large-en-v1.5) 및 reranker 모델 다운로드
HF_TOKEN=xxxxx make download-models MODEL_DOWNLOAD_TARGET=all

# 2) 별도 서버에서 OpenAI-compatible /v1/embeddings 제공
# 예: macOS 다른 프로세스에서 sentence-transformers 서버 실행
#     http://127.0.0.1:9000/v1/embeddings

# 3) SPAR가 원격 임베딩 서버를 사용하도록 설정
cat >> .env <<'EOF'
EMBEDDING_URL=http://127.0.0.1:9000/v1
ENCODER_URL=http://127.0.0.1:9000/v1
ENCODER_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_API_KEY=dummy
EOF

# 4) 연결 검증
make test-embedding-server

# 5) 문서 임베딩 + Milvus 적재
make ingest ARGS="--input-file data/skt-md/parameter_ref/foo.md --doc-type parameter_ref"
```

외부 macOS 임베딩 서버 예제는 [examples/embedding_server/README.md](examples/embedding_server/README.md)에서 바로 실행할 수 있습니다.

외부 임베딩 서버 요구사항:

- `POST /v1/embeddings`
- 요청 본문: `{"model":"...","input":["..."]}`
- 응답 형식: OpenAI style `data[].embedding` 또는 단순 `embeddings`

환경 변수 동작:

- `EMBEDDING_URL`: ingest 경로에서 사용하는 원격 임베딩 base URL
- `ENCODER_URL`: app/router 경로에서 우선 사용하는 원격 임베딩 base URL
- `ENCODER_URL`이 비어 있으면 `EMBEDDING_URL`을 재사용
- `ENCODER_MODEL`: 원격 서버에 전달할 모델 이름

> `make install-vllm`은 Linux에서는 `requirements-vllm.txt`를 사용하고, macOS Apple Silicon에서는 `vllm-metal` 설치 절차로 자동 분기합니다. 외부 임베딩 서버를 사용할 경우 이 단계는 필수가 아닙니다.

> `make download-models`는 기본값으로 `models/` 폴더에 내려받으며, `models/`는 `.gitignore`에 이미 등록되어 커밋되지 않습니다.

> **현 상태**: Phase 1 진행 중. LLM 모듈(factory/registry), 3-layer 라우터(Task 2.2), Milvus 클라이언트, 약어 사전(Task 1.6 ✅), FastAPI 앱, md ingest 파이프라인(Task 1.1/1.3 부분), embedder wrapper(Task 1.4 부분 — `EMBEDDING_TIMEOUT`/`EMBEDDING_BATCH_SIZE` env 지원, verbose 배치 진행 추적), encoder 싱글톤(Task 1.4 부분 ✅ — `ENCODER_MODEL`/`ENCODER_DEVICE` env vars), Codex+Gemini fallback 훅(INF-1b ✅), **LangGraph StateGraph 파이프라인** (`pipeline/` — Phase 5 조기 도입, reranker 첫 연결), **3GPP TS spec number 라우터(Task 2.2 부분 ✅ — RegexRouter TS 패턴 + spec intro ingest)**, **ingest acronym pre-pass + Milvus `keywords` ARRAY 필드 ✅** (Rel-18 2503 entries, noise filter 적용), **Milvus retrieval wiring ✅** (`retrieval/routing.py` + pipeline stub 제거 — rag_retrieve/structured_retrieve가 실제 `hybrid_search()` 호출), **멀티턴 history + 약어 컨텍스트 준비 (Task 2.5 부분 ✅** — `retrieval/query_rewriter.py`, `pipeline/prepare_context` 노드, API `history` 필드), **Task 1.7.2 평가 자동화 ✅** (Recall@K/MRR + faithfulness/answer_relevancy, `eval/ragas_metrics.py`), **라우터 골드셋 + 평가 스크립트 ✅** (`scripts/gen_router_goldset.py`, `scripts/run_router_eval.py` — Task 2.3), **Query Decomposition ✅ (Task 2.4** — `retrieval/query_decomposer.py`, `decompose`/`decomposed_retrieve` 노드, `RouteResult.needs_decomposition` 플래그) 구현됨, **LLM Query Rewriting ✅ (Task 2.5** — `retrieval/query_rewriter.py` `rewrite_query()`/`QueryRewriteResult`, `prompts/query_rewrite_system.txt`, `pipeline/rewrite_query` 노드, `tests/unit/retrieval/test_query_rewriter.py`) 구현됨.

---

## 디렉토리 구조

`src/` 레이아웃 사용. 코드는 모두 `src/spar/` 하위.

```
spar/
├── docs/prd.md       # 작업 내역서 (정답 출처)
├── AGENTS.md         # 에이전트/기여자 표준 가이드
├── pyproject.toml    # 프로젝트 메타 + 도구 설정
├── Makefile          # 단축 명령 (install/lint/test ...)
├── .env.example      # 환경 변수 템플릿
├── src/spar/
│   ├── api/              # FastAPI 앱 (app.py)
│   ├── llm/              # LLM 팩토리/싱글톤/레지스트리 (client, config, factory, registry)
│   ├── encoder/          # 임베딩 encoder — base.py (EncoderClient ABC), registry.py (SentenceTransformerEncoder + get_encoder() 싱글톤) (Task 1.4 — 부분 ✅)
│   ├── preprocessing/    # 질의 전처리 — 약어 매퍼 (Task 1.6 ✅)
│   ├── prompts/          # LLM 프롬프트 파일 저장소 — load_prompt() 헬퍼 + *.txt 템플릿
│   ├── router/           # 3-layer 라우터 (regex / embedding / llm / hybrid + schemas)
│   ├── ingest/           # md-aware/fixed 청커 + sentence-transformers embedder (Task 1.1/1.3/1.4 — 부분)
│   ├── pipeline/         # LangGraph StateGraph 오케스트레이션 — SparState, Nodes, build_graph() (Phase 5 조기 도입)
│   ├── reranker/         # CrossEncoderClient + 싱글톤 레지스트리 (Task 1.5)
│   ├── retrieval/        # Milvus 클라이언트, hybrid search, Route→doc_type 매핑, query_rewriter, query_decomposer, hyde, multi_query (Task 1.4~1.5, 2.4~2.6)
│   ├── parsers/          # 문서 유형별 파서 (Task 1.1 — scaffold)
│   ├── chunkers/         # 유형별 청킹 전략 (Task 1.3 — scaffold)
│   ├── db/               # Parameter/Counter/Alarm 구조화 DB (Task 3.1~3.2 — scaffold)
│   ├── kg/               # Knowledge Graph (Task 3.3~3.5 — scaffold)
│   ├── generation/       # citation, self-verify, confidence, fallback (Task 4.x — scaffold)
│   ├── agent/            # LangGraph agentic 확장 예비 (Phase 5 — scaffold)
│   ├── eval/             # 골드셋 평가 — metrics.py, run_eval.py (Retrieval), ragas_metrics.py, run_ragas_eval.py (답변품질) (Task 1.7.2 ✅)
│   └── dictionary/       # 약어/동의어 사전 (scaffold)
├── configs/
│   └── milvus/           # Milvus 연결/컬렉션 설정
├── scripts/              # ETL/배치/유틸리티 (init_milvus, serve_vllm, test_api,
│                          #   convert_pdf_to_md, fetch_tspec_llm, extract_acronyms,
│                          #   run_ingest, slice_3gpp_intros, gen_goldset_qa,
│                          #   gen_router_goldset, run_router_eval)
├── tests/                # pytest
└── data/                 # 골드셋, 샘플, 산출물
```

---

## 골드셋 QA 생성

3GPP `.md` 문서에서 Codex CLI로 질문-답변 세트를 자동 생성한다.  
상세 가이드: [`docs/goldset-qa-generation-guide.md`](docs/goldset-qa-generation-guide.md)

```bash
# 사전 조건: Codex 로그인
codex whoami  # 미로그인 시: codex login

# 파일 지정
python scripts/gen_goldset_qa.py \
  --input-file data/tspec-llm/3GPP-clean/Rel-18/29_series/29502-i40.md \
  --output data/goldsets/retrieval_goldset.jsonl

# 폴더 지정 (재귀 처리)
python scripts/gen_goldset_qa.py \
  --input-dir data/tspec-llm/3GPP-clean/Rel-18/29_series \
  --output data/goldsets/retrieval_goldset.jsonl \
  --append
```

출력 형식 (`data/goldsets/retrieval_goldset.jsonl`):
```jsonl
{"query_id": "Q0001", "query": "SMF가 PDU 세션 수립 시 호출하는 서비스는?", "answer": "Nsmf_PDUSession_CreateSMContext", "type": "lookup", "section": "5.2.2.1", "source_doc": "29502-i40.md", "spec_number": "29.502", "release": "Rel-18"}
```

---

## 기여 가이드

- 작업 단위는 PRD의 Phase / Task
- 코드 주석 및 문서: 한국어
- 로그 메시지: 영어
- 커밋: Conventional Commits (`feat:`, `fix:`, `docs:` 등)
- 상세 규약: [`AGENTS.md`](AGENTS.md)

---

## 라이선스

내부 프로젝트. 외부 배포 시점에 라이선스 결정.
