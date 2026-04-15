# LangGraph Vis (Week 04)

이 프로젝트는 `langgraph_vis` 경로에서 백엔드(Python/FastAPI)와 간단한 웹 페이지를 함께 실행할 수 있습니다.

## 실행 방법

### 1) 환경 준비

```bash
cd langgraph_vis
python -m venv .venv
source .venv/bin/activate
python -m pip install -r python_backend/requirements.txt
```

### 2) 백엔드 실행

```bash
# Linux/macOS
cd /path/to/remote_work/langgraph_vis
PYTHONPATH=$(pwd) .venv/bin/python -m uvicorn python_backend.app.main:app --host 0.0.0.0 --port 8000
```

- API 서버: `http://127.0.0.1:8000`
- API 문서: `http://127.0.0.1:8000/docs`
- 테스트 페이지: `http://127.0.0.1:8000/ui/`

참고:
- 서버 루트(`/`)는 `http://127.0.0.1:8000/ui/`로 이동합니다.
- run/event를 즉시 넣고 확인하려면:
  - `POST /api/runs`
  - `POST /api/runs/{runId}/events`
- UI에도 `Create run`/`Emit event` 버튼이 추가되어 실시간으로 데이터 입력 후 SSE를 확인할 수 있습니다.

### 3) 테스트 실행

```bash
cd langgraph_vis
npm test
```

## 페이지 사용

- `http://127.0.0.1:8000/ui/` 에서 실행 상태/이벤트 조회를 할 수 있습니다.
- `runId` 와 `threadId` 를 입력하고:
  - `Load state`: `GET /api/runs/{runId}/state` 조회
  - `Load events`: `GET /api/runs/{runId}/events` 조회
  - `Open stream`: `GET /api/runs/{runId}/events/stream` SSE 구독

## 참고

- 실행 시 이벤트 데이터가 없다면 API 응답이 비어 있을 수 있습니다.
- 실제 런 데이터 준비용 UI(등록/전송 기능)는 이번 버전에서 별도 미구현입니다.
