# Chat Interface + Workflow Visualization

React/Vite 채팅 UI + FastAPI/LangGraph 워크플로우 실행 시각화 시스템.

## 서비스 구성

| 서비스 | 포트 | 설명 |
|---|---|---|
| `chat-front` | 10000 | React + Vite (`vite preview`) |
| `workflow-api` | 10001 | FastAPI + LangGraph WebSocket |

## 실행

```bash
docker compose up --build -d
docker compose ps   # 두 서비스 모두 Up 확인
```

브라우저에서 `http://localhost:10000` 접속.

## 외부 접속 (NAT 포트포워딩)

**포트 10000과 10001 모두 포워딩 필요.**

백엔드 URL은 빌드 타임 고정값 없이 `window.location.hostname`에서 런타임 도출:
- WS: `ws://<접속호스트>:10001/ws/connect`
- REST: `http://<접속호스트>:10001/graph`

## 프론트엔드 구조

```
src/
├── App.jsx
├── constants.js
├── components/      # ChatPane, Composer, WorkflowPanel, SettingsModal 등
├── hooks/           # useWorkflowSocket, useWorkflowGraph, useScrollBehavior
└── utils/           # nodeUtils (노드 ID/색상 변환)
```

## 개발 (호스트 직접 실행)

```bash
# 프론트엔드
npm install && npm run dev

# 백엔드
pip install -r backend/requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`uvicorn[standard]` 필수 — 기본 `uvicorn`은 WebSocket 지원 없음.
