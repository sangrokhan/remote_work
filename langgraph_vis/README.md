# LangGraph Vis (Week 04)

PoC용 최소 웹페이지 + Python(FastAPI) + LangGraph 연동 예시입니다.

- 페이지는 `Run` 버튼 하나만 표시됩니다.
- `Run` 클릭 시 백엔드에서 LangGraph 노드 스트림을 받아 텍스트로 출력합니다.

## 실행 방법

### 1) 환경 준비

```bash
cd /path/to/remote_work/langgraph_vis
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2) 앱 실행

```bash
PYTHONPATH=. .venv/bin/python -m uvicorn python_backend.app.main:app --host 0.0.0.0 --port 8000
```

- 서버 URL: `http://127.0.0.1:8000`
- 루트 페이지: `http://127.0.0.1:8000`  
  (단일 페이지에서 `Run` 버튼을 통해 스트리밍 로그 확인 가능)
- SSE 스트림 API: `GET /run`  

### 3) 테스트

```bash
npm test
```

## 코드 구조

- `[python_backend/app/main.py]`(/home/han/.openclaw/workspace/remote_work/langgraph_vis/python_backend/app/main.py): FastAPI 앱, `/run` 스트리밍 엔드포인트, 정적 파일 서빙
- `[python_backend/stategraph_workflow.py]`(/home/han/.openclaw/workspace/remote_work/langgraph_vis/python_backend/stategraph_workflow.py): 4개 노드(Planner, Executor, Refiner, Synthesizer) LangGraph 상태 전이
- `[frontend/index.html]`(/home/han/.openclaw/workspace/remote_work/langgraph_vis/frontend/index.html), `[frontend/styles.css]`(/home/han/.openclaw/workspace/remote_work/langgraph_vis/frontend/styles.css), `[frontend/app.js]`(/home/han/.openclaw/workspace/remote_work/langgraph_vis/frontend/app.js): 최소 UI(흰색 배경, Run 버튼, 텍스트 로그)
