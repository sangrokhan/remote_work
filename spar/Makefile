# SPAR 개발 환경 단축 명령
# Python 3.12 + pip + venv 기준

PYTHON ?= python3.12
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

MILVUS_COMPOSE := configs/milvus/docker-compose.yml

.PHONY: help venv install install-dev lint format typecheck test test-cov clean \
        milvus-up milvus-down milvus-ui milvus-init milvus-reset milvus-list milvus-logs \
        vllm-serve vllm-serve-router

help:
	@echo "사용 가능한 명령:"
	@echo "  make venv          - 가상환경 생성 ($(VENV))"
	@echo "  make install       - 런타임 의존성 설치"
	@echo "  make install-dev   - 개발 의존성 포함 설치 (권장)"
	@echo "  make lint          - ruff 린트"
	@echo "  make format        - ruff 포매터 적용"
	@echo "  make typecheck     - mypy 타입 체크"
	@echo "  make test          - pytest 실행"
	@echo "  make test-cov      - 커버리지 포함 테스트"
	@echo "  make clean         - 캐시/빌드 산출물 정리"
	@echo ""
	@echo "  [Milvus]"
	@echo "  make milvus-up     - Milvus standalone 시작 (etcd + MinIO + Milvus)"
	@echo "  make milvus-down   - Milvus 중지 (볼륨 보존)"
	@echo "  make milvus-ui     - Attu Web UI 포함 시작 (port 8080)"
	@echo "  make milvus-init   - 컬렉션 초기화 (없으면 생성)"
	@echo "  make milvus-reset  - 컬렉션 전체 삭제 후 재생성"
	@echo "  make milvus-list   - 현재 컬렉션 목록 출력"
	@echo "  make milvus-logs   - Milvus 컨테이너 로그"
	@echo ""
	@echo "  [vLLM]"
	@echo "  make vllm-serve        - 메인 LLM 서빙 (gemma-4-E4B-it, port 8000)"
	@echo "  make vllm-serve-router - 라우터 LLM 서빙 (Qwen2.5-7B, port 8001)"
	@echo "  MODEL=<id> PORT=<n> make vllm-serve  - 커스텀 모델/포트"

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

install: venv
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: venv
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e ".[dev]"

lint:
	$(VENV)/bin/ruff check .

format:
	$(VENV)/bin/ruff format .
	$(VENV)/bin/ruff check --fix .

typecheck:
	$(VENV)/bin/mypy .

test:
	$(VENV)/bin/pytest

test-cov:
	$(VENV)/bin/pytest --cov --cov-report=term-missing --cov-report=html

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[cod]" -delete

# ---------------------------------------------------------------------------
# Milvus
# ---------------------------------------------------------------------------

milvus-up:
	docker compose -f $(MILVUS_COMPOSE) up -d
	@echo "Milvus gRPC: localhost:$${MILVUS_PORT:-19530}"

milvus-ui:
	docker compose -f $(MILVUS_COMPOSE) --profile ui up -d
	@echo "Attu UI: http://localhost:8080"

milvus-down:
	docker compose -f $(MILVUS_COMPOSE) --profile ui down

milvus-logs:
	docker compose -f $(MILVUS_COMPOSE) logs -f milvus

milvus-init:
	$(PY) scripts/init_milvus.py

milvus-reset:
	$(PY) scripts/init_milvus.py --reset

milvus-list:
	$(PY) scripts/init_milvus.py --list

# ---------------------------------------------------------------------------
# 전처리 (Ingest)
# ---------------------------------------------------------------------------
# 사용 예: make ingest ARGS="--input-file data/raw/foo.txt --doc-type parameter_ref"
#          make ingest ARGS="--input-dir data/raw/ --doc-type mop --force"
#          make ingest-dry FILE=data/raw/foo.txt DOC_TYPE=parameter_ref

ingest:
	$(PY) scripts/run_ingest.py $(ARGS)

ingest-dry:
	$(PY) scripts/run_ingest.py --input-file $(FILE) --doc-type $(DOC_TYPE) --dry-run

# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------

vllm-serve:
	MODEL=$${MODEL:-google/gemma-4-E4B-it} PORT=$${PORT:-8000} bash scripts/serve_vllm.sh

vllm-serve-router:
	MODEL=$${MODEL:-Qwen/Qwen2.5-7B-Instruct} PORT=$${PORT:-8001} bash scripts/serve_vllm.sh
