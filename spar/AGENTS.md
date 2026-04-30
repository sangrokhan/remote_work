# AGENTS.md

> 본 문서는 이 저장소에서 작업하는 AI 에이전트(Claude Code, Codex, 기타) 및 신규 합류자를 위한 표준 가이드입니다.
> 정답 출처는 `docs/prd.md`이며, 본 문서는 운영 규약(컨벤션, 빌드/테스트, 디렉토리 맵)을 정의합니다.

---

## 1. 프로젝트 개요

- **이름**: SPAR (Samsung RAN LLM+RAG)
- **목적**: Samsung 단일 벤더(LTE+NR) 환경의 내부 문서(파라미터/카운터/알람/MOP/Feature/Release Notes 등)에 대한 자연어 질의응답 시스템
- **운영 환경**: 온프레미스, 영어 응답, 정확성 최우선 (hallucination 최소화)
- **상위 로드맵**: `docs/prd.md`의 Phase 0 ~ Phase 5 + INF 작업 참조
- **현 단계**: Phase 0 진입 직전 (greenfield, 디렉토리 스캐폴드 단계)

---

## 2. 기술 스택

| 영역 | 선택 |
|---|---|
| 언어 | Python 3.12 |
| 의존성 관리 | `pip` + `venv` + `pyproject.toml` + `requirements*.txt` |
| LLM 서빙 | vLLM (메인), SGLang (대안) |
| 벡터 DB | **Milvus** (1순위), Qdrant/Weaviate (대안) |
| BM25 | Elasticsearch 또는 OpenSearch |
| 그래프 DB | Neo4j (Phase 3) |
| 평가 | RAGAS, 자체 골드셋 |
| 테스트 | `pytest` |
| 린팅/포매팅 | `ruff` (lint + format) |
| 타입 체크 | `mypy` 또는 `pyright` (Phase 1 이후 도입) |

> 라이브러리 후보 전체 표는 `docs/prd.md` 부록 참조.

---

## 3. 디렉토리 맵

`src/` 레이아웃 사용 (import 충돌 방지, 설치 시 패키지 일관성).

```
spar/
├── AGENTS.md                # 본 문서 (루트 표준)
├── README.md
├── pyproject.toml           # 프로젝트 메타 + 도구 설정 (ruff, pytest, mypy, coverage)
├── requirements.txt         # 런타임 의존성
├── requirements-dev.txt     # 개발 의존성
├── Makefile                 # 빌드/테스트 단축 명령
├── .env.example             # 환경 변수 템플릿 (실제 .env는 git 추적 제외)
├── .python-version          # 3.12
├── docs/
│   └── prd.md               # 정답 출처 (Phase 로드맵)
├── src/
│   └── spar/                # 단일 최상위 패키지
│       ├── __init__.py
│       ├── parsers/         # 문서 유형별 PDF/텍스트 파서 (Task 1.1)
│       ├── chunkers/        # 유형별 청킹 전략 (Task 1.3)
│       ├── retrieval/       # hybrid search, reranker, decomposer, rewriter (Task 1.4~1.5, 2.4~2.7)
│       ├── router/          # 3-layer 라우터 (regex / embedding / LLM) (Task 2.1~2.2)
│       ├── db/              # Parameter/Counter/Alarm 구조화 DB + Text-to-SQL (Task 3.1~3.2)
│       ├── kg/              # Knowledge Graph + Text-to-Cypher + GraphRAG (Task 3.3~3.5)
│       ├── generation/      # citation enforcer, self-verifier, confidence, fallback (Task 4.1~4.5)
│       ├── agent/           # LangGraph 기반 agentic 재구성 (Phase 5)
│       ├── eval/            # 골드셋 평가 스크립트, 메트릭 (Task 0.2~0.3)
│       └── dictionary/      # 약어/동의어 사전 (Task 1.6)
├── configs/                 # YAML/JSON 설정 (모델, 인덱스, 라우트 등)
│   └── secrets/             # *.local.yaml — git 제외
├── scripts/                 # 일회성 ETL/배치/유틸리티
├── tests/                   # pytest 단위/통합 테스트
└── data/                    # 골드셋, 샘플 입력, 추출 산출물 (큰 원본은 git LFS 또는 외부 저장)
    └── samples/             # 공개 가능한 샘플만 추적
```

- **import 규칙**: `from spar.retrieval import ...` 형태. 모듈 직접 import 금지.
- **하위 AGENTS.md**: `src/spar/<module>/AGENTS.md`로 둠. 모듈 인터페이스/스키마 명세.

- **하위 AGENTS.md 정책**: 각 모듈 디렉토리는 자체 `AGENTS.md`를 둘 수 있으며, 모듈 고유의 규약(스키마, 인터페이스, 호출 예시)을 기술. 루트 본 문서와 충돌 시 모듈 문서가 우선.
- **빈 디렉토리**: 현재 `.gitkeep`으로 추적. 첫 모듈 코드 추가 시 `__init__.py`로 교체.

---

## 4. 빌드 / 테스트 / 평가 명령

아직 코드 미작성. `Makefile` 단축 명령 제공.

```bash
# 가상환경 생성 + 개발 의존성 설치 (1회)
make install-dev

# 환경 변수 준비
cp .env.example .env
# .env 편집

# 린트 / 포맷 / 타입 / 테스트
make lint
make format
make typecheck
make test
make test-cov

# 골드셋 평가 (Task 0.3 산출물 — 예정)
.venv/bin/python -m spar.eval.run_eval --goldset data/goldsets/retrieval_goldset.jsonl
```

수동 사용 시:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

---

## 5. 코딩 컨벤션

### 5.1 언어 규칙
- **코드 주석**: 한국어
- **docstring**: 한국어 (외부 라이브러리 인터페이스만 영어 허용)
- **로그 메시지**: 영어 (운영 환경/모니터링 도구 호환성 고려)
- **변수/함수/클래스명**: 영어, snake_case (PEP 8)
- **문서(`docs/`, `AGENTS.md`)**: 한국어

### 5.2 Python 스타일
- PEP 8 + `ruff` 기본 룰 준수
- 라인 길이 100자
- 타입 힌트 필수 (공개 API 한정 strict, 내부 헬퍼는 권장)
- `from __future__ import annotations` 기본 사용
- f-string 우선, `.format()` 지양
- `pathlib.Path` 우선, `os.path` 지양

### 5.3 모듈 설계
- 각 모듈은 단일 책임 원칙
- 외부 의존(LLM/DB/벡터DB)은 `configs/`로 주입, 코드에 하드코딩 금지
- 비결정적 호출(LLM 등)은 mock 가능한 인터페이스로 분리

### 5.4 데이터 스키마
- 청크 메타데이터 필수 필드는 `docs/prd.md` Task 1.2 참조
- 스키마 변경 시 `chunk_schema.json` 업데이트 + 마이그레이션 스크립트 동반

---

## 6. 평가 / 품질 게이트

- **모든 Phase 종료 시 골드셋 재측정** (`docs/prd.md` 각 Phase 평가 Task)
- **회귀 방지**: PR마다 Recall@10, MRR, faithfulness 비교 (CI 도입 후)
- **Hallucination**: Phase 4 완료 시점 RAGAS faithfulness ≥ 0.9 목표
- **Citation**: 모든 사실 주장에 출처 청크 ID 매핑 (Phase 4 강제)

---

## 7. 작업 진행 규칙

- **Phase 단위 진행**: 한 세션에 한 Phase. PRD 체크박스를 진척 관리에 그대로 사용.
- **작업 사이클**: 가설 → 코드 수정 → 평가 → 분석 → 다음 가설
- **PRD가 정답**: 본 문서 또는 코드가 PRD와 어긋나면 PRD 우선. 단, PRD가 현실과 맞지 않을 때는 PRD를 갱신하고 변경 사유 기록.
- **TODO/FIXME**: 코드 내 한국어 허용. 단, 만료 조건 또는 담당 이슈 번호 명시.

---

## 8. 보안 / 운영

- **온프레미스 전제**: 외부 API 호출 금지 (모든 LLM은 사내 vLLM/SGLang 통해 호출)
- **시크릿 관리**:
  - 로컬 개발: `.env` (`.env.example` 복사 후 작성, gitignore 처리됨)
  - 운영: `configs/secrets/*.local.yaml` (gitignore) 또는 사내 시크릿 매니저
  - 코드/커밋에 절대 포함 금지. `.env.example`은 키 이름만 포함, 값 비움.
- **로그**: PII/시크릿 마스킹. 사용자 질의 로그는 익명화 후 `data/logs/`.

---

## 9. 커밋 / PR 컨벤션

- **커밋 메시지**: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
- **본문 언어**: 한국어 또는 영어 (팀 합의 시점 통일)
- **PR 본문**: Phase/Task 번호 명시 (예: `[Phase 1 / Task 1.5] reranker 통합`)
- **체크리스트**: PRD 체크박스를 PR 본문에 인용하여 완료 표시

---

## 10. 참고

- 정답 출처: [`docs/prd.md`](docs/prd.md)
- 즉시 착수 권장 작업: PRD §"즉시 착수 권장" 참조 (Task 0.2, 0.3, 1.5, 1.6, 2.4, 2.2, 1.4)

---

*본 문서는 디렉토리/스택/규약 변경 시 즉시 갱신합니다.*
