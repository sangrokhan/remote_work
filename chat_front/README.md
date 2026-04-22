# Agentic RAG — Chat Interface + Workflow Visualization

React/Vite 채팅 UI + FastAPI/LangGraph 워크플로우 실행 시각화 시스템.

## 서비스 구성

| 서비스 | 포트 | 설명 |
|---|---|---|
| `chat-front` | 10000 | React + Vite (`vite preview`) |
| `workflow-api` | 10001 | FastAPI + SSE 스트리밍 |

## 실행

```bash
docker compose up --build -d
docker compose ps   # 두 서비스 모두 Up 확인
```

브라우저에서 `http://localhost:10000` 접속.

## 외부 접속 (NAT 포트포워딩)

포트 10000과 10001 모두 포워딩 필요.

`.env` 파일로 백엔드 URL 지정:

```env
VITE_WORKFLOW_GRAPH_URL=http://<호스트>:10001/graph
VITE_WORKFLOW_RUN_URL=http://<호스트>:10001/api/run
```

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/health` | 헬스체크 |
| `GET` | `/graph` | LangGraph 스키마 `{nodes, edges}` |
| `POST` | `/api/run` | SSE 스트림 — 워크플로우 실행 |

### POST /api/run 요청 바디

```json
{
  "run_id": "<uuid>",
  "input": "<사용자 입력>",
  "model": "GaussO4",
  "response_mode": "normal",
  "max_tokens": 1024,
  "agentic_rag": false,
  "api_url": "",
  "api_key": ""
}
```

### SSE 이벤트 순서

```
run_started → workflow_event (×n) → workflow_complete | workflow_error
```

각 이벤트: `{event, node, name, stage, message, payload?}`  
`stage ∈ {start, end, routing, error}`

## 소스 트리

```
chat_front/
├── backend/                        # FastAPI 서비스
│   ├── app/
│   │   ├── main.py                 # GET /graph, POST /api/run (SSE)
│   │   └── models.py               # RunWorkflowRequest
│   ├── graph_schema.py             # StateGraph → {nodes, edges} JSON
│   ├── stategraph_workflow.py      # 데모 그래프 + 비동기 스트리밍
│   └── requirements.txt
│
├── langgraph_flow/                 # 실제 LangGraph 에이전트 구현 (WIP)
│   ├── agents/
│   │   ├── graph.py                # AgenticRAGGraph 클래스
│   │   ├── state.py                # AgentState TypedDict
│   │   ├── nodes/                  # 노드별 파일 (planner, executor, refiner, synthesizer, retriever, var_constructor, var_binder)
│   │   └── edges/
│   │       └── routing_logic.py    # 조건부 엣지 함수
│   ├── core/
│   │   ├── base.py                 # BaseLLM ABC
│   │   ├── factory.py              # get_llm(model, api_url, api_key)
│   │   └── models/                 # 모델별 구현 (GaussO4, GaussO4Think, Gemma4E4BIt)
│   └── prompts/                    # 노드별 프롬프트
│
├── frontend/                       # React/Vite 프론트엔드
│   ├── App.jsx
│   ├── constants.js
│   ├── components/                 # ChatPane, Composer, PaneHeader, SettingsModal, WorkflowPanel
│   ├── hooks/                      # useWorkflowSSE, useWorkflowSocket, useWorkflowGraph, useScrollBehavior
│   └── utils/
│       └── nodeUtils.js
│
├── docker-compose.yml
├── Dockerfile                      # 프론트엔드 컨테이너
└── backend/Dockerfile              # 백엔드 컨테이너
```

## 주요 기능

- **스플릿 패널** — 두 패널 동시 활성화 시 독립적인 모델/RAG 설정으로 각각 POST /api/run 요청
- **워크플로우 시각화** — Cytoscape.js + dagre로 LangGraph 노드 실행 상태 실시간 하이라이트
- **Agentic RAG** — retriever → var_constructor → var_binder → planner 경로 (WIP)
- **모델 선택** — 패널별 독립 모델 선택 (GaussO4, GaussO4-think, Gemma4-E4B-it)

## 개발 (호스트 직접 실행)

```bash
# 프론트엔드
npm install && npm run dev

# 백엔드 (backend/ 디렉토리 내에서)
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
