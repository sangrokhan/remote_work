# SPAR 개발 환경 단축 명령
# Python 3.12 + pip + venv 기준

PYTHON ?= python3.12
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: help venv install install-dev lint format typecheck test test-cov clean

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
