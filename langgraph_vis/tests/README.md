# 테스트 실행 가이드

## 테스트 구조
- `python_backend/tests/*`: 파이썬 백엔드 PoC 테스트
- `tests/week-04/*.test.js`: 4주차 UI/표시 파이프라인 PoC 테스트

## 실행 명령
- 가상환경 생성: `python3 -m venv .venv`
- 의존성 설치: `.venv/bin/pip install -r python_backend/requirements.txt && .venv/bin/pip install -r python_backend/requirements-dev.txt`
- 테스트 실행: `.venv/bin/pytest -q python_backend/tests`
- Node 테스트 실행: `node --test tests/week-04/*.test.js`
- 통합 실행: `npm run test`

## 참고
- JS backend 연계 테스트는 Python 전용으로 정리되었지만, Week 4 PoC 용도로 `node:test` 기반 view 모델/host-shell 테스트를 별도 운영합니다.
