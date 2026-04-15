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
- PoC 데모에서는 `POST /api/runs`로 내부 런을 생성한 뒤
  `POST /api/runs/{runId}/demo/run`으로 더미 워크플로우를 실행합니다.
- 기본 워크플로우는 `langgraph_demo_workflow_v1`이며 3개 노드가 **백엔드에서 3~5초 랜덤 지연**으로 동작합니다.
  - 3번째 노드에서 일정 확률로 1번 노드로 되돌아가는 루프를 재현합니다.

### 3) 테스트 실행

```bash
cd langgraph_vis
npm test
```

## 페이지 사용

- `http://127.0.0.1:8000/ui/`에서 바로 데모를 실행할 수 있습니다.
- `Run` 버튼 하나만 클릭하면 스키마 조회, 실행 요청, 실시간 노드 하이라이트가 모두 동작합니다.

## 참고

- 실행 시 이벤트 데이터가 없다면 API 응답이 비어 있을 수 있습니다.
- 실제 런 데이터 준비용 UI(등록/전송 기능)는 이번 버전에서 별도 미구현입니다.
