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

# 2. 환경 변수 설정
cp .env.example .env
# .env 편집

# 3. 검증
make lint
make test
```

> **현 상태**: Phase 1 진행 중. LLM 모듈(factory/registry), 3-layer 라우터(Task 2.2), Milvus 클라이언트, 약어 사전(Task 1.6 ✅), FastAPI 앱, md ingest 파이프라인(Task 1.1/1.3 부분), embedder wrapper(Task 1.4 부분), Codex+Gemini fallback 훅(INF-1b ✅) 구현됨.

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
│   ├── encoder/          # 임베딩 encoder 싱글톤/팩토리 (Task 1.4 — 진행 중, untracked)
│   ├── preprocessing/    # 질의 전처리 — 약어 매퍼 (Task 1.6 ✅)
│   ├── router/           # 3-layer 라우터 (regex / embedding / llm / hybrid + schemas)
│   ├── ingest/           # md-aware/fixed 청커 + sentence-transformers embedder (Task 1.1/1.3/1.4 — 부분)
│   ├── retrieval/        # Milvus 클라이언트, hybrid search, reranker (Task 1.4~1.5)
│   ├── parsers/          # 문서 유형별 파서 (Task 1.1 — scaffold)
│   ├── chunkers/         # 유형별 청킹 전략 (Task 1.3 — scaffold)
│   ├── db/               # Parameter/Counter/Alarm 구조화 DB (Task 3.1~3.2 — scaffold)
│   ├── kg/               # Knowledge Graph (Task 3.3~3.5 — scaffold)
│   ├── generation/       # citation, self-verify, confidence, fallback (Task 4.x — scaffold)
│   ├── agent/            # LangGraph 기반 agentic 파이프라인 (Phase 5 — scaffold)
│   ├── eval/             # 골드셋 평가 스크립트 (scaffold)
│   └── dictionary/       # 약어/동의어 사전 (scaffold)
├── configs/
│   └── milvus/           # Milvus 연결/컬렉션 설정
├── scripts/              # ETL/배치/유틸리티 (init_milvus, serve_vllm, test_api,
│                          #   convert_pdf_to_md, fetch_tspec_llm, extract_acronyms, run_ingest)
├── tests/                # pytest
└── data/                 # 골드셋, 샘플, 산출물
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
