# 테스트 실행 가이드

## 테스트 구조
- `python_backend/tests/*`: 파이썬 백엔드 PoC 테스트
- `frontend/frontend_smoke.test.js`: 프론트엔드 HTML 정합성(런 버튼/워크플로우/스트림 이벤트 식별자) 확인

## 실행 명령
- 가상환경 생성: `python3 -m venv .venv`
- 의존성 설치: `.venv/bin/pip install -r python_backend/requirements.txt && .venv/bin/pip install -r python_backend/requirements-dev.txt`
- 백엔드 테스트 실행: `.venv/bin/pytest -q python_backend/tests`
- 프론트엔드 테스트 실행: `node --test frontend/frontend_smoke.test.js`
- 통합 실행: `npm run test`

## 참고
- 현재 JS 테스트는 브라우저 동작까지 실행하는 E2E가 아니라, `frontend/index.html`에 핵심 구현 단서(요소/엔드포인트/이벤트 핸들러)가 있는지 검증하는 스모크 테스트입니다.
