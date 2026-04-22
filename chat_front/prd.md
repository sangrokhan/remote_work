# 요구사항 명세서: 채팅 인터페이스 + 워크플로우 시각화

## 1. 서비스 개요

React/Vite 기반 채팅 UI와 FastAPI/LangGraph 기반 백엔드로 구성된 워크플로우 실행 시각화 시스템.
사용자는 채팅 인터페이스로 입력하고, 우측 사이드패널에서 LangGraph 워크플로우 그래프 실행 상태를 실시간으로 확인한다.

---

## 2. 시스템 구성

### 2.1 서비스 및 포트

| 서비스 | 기술 스택 | 포트 |
|---|---|---|
| `chat-front` | React + Vite → nginx 이미지 | `10000` |
| `workflow-api` | FastAPI + LangGraph | `10001` |

### 2.2 환경변수 (`.env`)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `VITE_WORKFLOW_WS_URL` | `ws://localhost:10001/ws/connect` | WebSocket 접속 주소 |
| `VITE_WORKFLOW_GRAPH_URL` | `http://localhost:10001/graph` | REST 그래프 조회 주소 |

`VITE_WORKFLOW_WS_URL` 미설정 시 `App.jsx`가 `VITE_WORKFLOW_GRAPH_URL`에서 scheme/path를 변환해 자동 도출.

---

## 3. 백엔드 요구사항

### 3.1 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | `{"status": "ok"}` 반환 |
| GET | `/graph` | 워크플로우 그래프 스키마 반환 |
| GET | `/api/graph` | `/graph`와 동일 (alias) |
| WebSocket | `/ws/connect` | 채팅 및 워크플로우 실행 채널 |

### 3.2 WebSocket 프로토콜

연결 즉시 `{"type": "connected", "status": "ready"}` 전송.

**수신 메시지 처리:**

| 수신 | 응답 |
|---|---|
| `"ping"` | `"pong"` |
| `"graph"` / `"get_graph"` / `"refresh"` | `{"type": "graph", "payload": {nodes, edges}}` |
| `{"type": "run_workflow", "run_id", "input", ...}` | 워크플로우 실행 스트리밍 |

**워크플로우 실행 스트리밍 이벤트 순서:**

1. `{"type": "workflow_started", "run_id": ...}`
2. 노드별 `{"type": "workflow_event", "run_id", "event", "node", "name", "stage", "message"}` 반복
   - `stage ∈ {start, end, routing, error}`
3. `{"type": "workflow_complete", "run_id": ...}` 또는 `{"type": "workflow_error", ...}`

### 3.3 워크플로우 그래프 구조

LangGraph `StateGraph` 기반 4-노드 구성:

```
planner → executor → refiner → (planner | synthesizer)
                   ↘ synthesizer
```

노드: `__start__`, `planner`, `executor`, `refiner`, `synthesizer`, `__end__`

그래프 스키마 응답 형식: `{"nodes": [...], "edges": [...]}`

### 3.4 런타임 의존성

- `uvicorn[standard]` 필수 (`websockets` 런타임 포함). 기본 `uvicorn`만 설치 시 WebSocket 핸드셰이크 실패.

---

## 4. 프론트엔드 요구사항

### 4.1 전체 레이아웃

```
┌─────────────────────────────────┬────────────┐
│  헤더 (타이틀 | 설정버튼 토글버튼) │            │
├─────────────────────────────────┤  우측패널   │
│                                 │  (25% 너비) │
│  채팅 메시지 영역                  │            │
│                                 │  워크플로우 │
│  입력 영역 (80% 너비, 하단 중앙)   │  그래프    │
└─────────────────────────────────┴────────────┘
```

- 우측 패널: 기본 **닫힘** 상태. 열리면 채팅 영역 너비가 25% 축소.
- 패널 개폐는 채팅 영역 `width`만 변경하며 정렬/패딩 구조는 유지.

### 4.2 채팅 영역

- 사용자/어시스턴트 말풍선 구분 렌더링.
- 입력창: 기본 3줄, 최대 5줄 자동 확장. 5줄 초과 시 내부 스크롤.
- 입력 영역은 채팅 너비의 80%, 하단 중앙 배치.
- 전송 버튼: 입력창 우하단 원형 버튼, 상향 화살표 아이콘.
- 입력창 하단에 모델 선택 드롭다운 (compact, 현재 선택 모델 표시).
- 지원 모델: `gpt-4.1`, `gpt-4o-mini`, `gpt-4o`.

### 4.3 스크롤 UX

- 새 메시지/업데이트 시 자동 스크롤 (`scrollTop = scrollHeight`).
- 스크롤바: 기본 숨김. 오버플로우 상태에서 사용자가 스크롤 동작 시 1.2초간 표시 후 재숨김 (`messages-scrollbar-visible` 클래스 토글).

### 4.4 설정 모달

- 헤더 우측 설정 버튼으로 열림.
- 위치: 화면 중앙. 라운드 엣지 (≥18px). 반투명 블러 오버레이.
- 닫기: 닫기 버튼 / 배경 클릭 / ESC.
- 설정 항목: 응답 모드, 최대 토큰.
- 모달과 로그 패널 제어는 서로 독립적으로 동작.

### 4.5 우측 워크플로우 패널

- 토글 버튼: 화면 세로 중앙 고정, 우측 경계에 50% 걸침. 패널 열림/닫힘에 따라 위치 이동.
- 패널 열림 시 WebSocket으로 `get_graph` 전송 → 그래프 수신 → Cytoscape 렌더링.
- 패널 닫힘 시 Cytoscape 인스턴스 파괴 및 하이라이트 초기화.
- WebSocket 오류/비정상 종료 시 REST `/graph`로 폴백.

### 4.6 워크플로우 그래프 시각화

**Cytoscape.js** + **dagre** 레이아웃 사용.

레이아웃 설정:
- `rankDir: TB` (상→하)
- `fit: false` (자동 축소 방지)
- `zoom: 1` 고정
- 레이아웃 완료 후 `cy.center()` 호출

노드 스타일:
- 크기: `width: 80`, `height: 40`
- `font-size: 16`, `text-max-width: 80px`, `padding: 6`
- `font-family`: 명시적 시스템 폰트 스택 (`system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`)
  - `inherit` 사용 금지 — canvas 2D `ctx.font`에서 파싱 실패해 기본 소형 폰트로 fallback됨
- `content: (node) => node.data('label')` 사용. `label:` 속성 사용 금지.
- `max-width`, `max-height`, `shadow-*`, `wheelSensitivity` 사용 금지 (Cytoscape 경고 발생).

노드 라벨 변환:
- `__start__` → `START`, `__end__` → `END`

노드 색상 팔레트 (`WORKFLOW_NODE_PALETTE`):

| 노드 | bg | border | text |
|---|---|---|---|
| planner | `#d7ecff` | `#68a8ee` | `#12365f` |
| executor | `#d7f4dd` | `#7dcf90` | `#1a4f2f` |
| refiner | `#fff1c7` | `#e2be5e` | `#5e4b17` |
| synthesizer | `#f5d9fc` | `#c78ce0` | `#5d2e69` |
| start | `#e7e7f8` | `#9ca2df` | `#32366c` |
| end | `#dceeff` | `#5f8bb0` | `#263246` |
| default | `#edf2ff` | `#a9b6d7` | `#2f3c56` |

엣지 라벨: **미표시** (조건값 비노출).
`__start__`, `__end__` 노드: 표시.

### 4.7 워크플로우 노드 실행 하이라이트

- 각 `sendMessage()` 호출 시 `crypto.randomUUID()`로 `runId` 생성.
- `run_id` 기반으로 `workflow_event`를 동일 assistant bubble에 누적 출력.
- `workflowExecutionRef` (`useRef`, 상태 아님) 로 `{runId, isRunning, activeNode}` 추적 — 하이라이트 업데이트 시 re-render 방지.
- `node_started` 또는 `stage === 'start'` 수신 시: 해당 노드에 `wf-active` Cytoscape 클래스 적용.
- `workflow_complete` / `workflow_error` 수신 시: 하이라이트 제거.
- `normalizeWorkflowNodeId`: `START`/`END` → `__start__`/`__end__` 변환 (백엔드 LangGraph 센티넬과 매핑).

### 4.8 WebSocket 수명주기

- 앱 마운트 시 1회 연결 (패널 열림/닫힘과 무관).
- 20초 간격 `ping` 전송으로 연결 유지.
- 패널 `open` 시 기존 소켓으로 `get_graph` 전송.

---

## 5. 컬러 팔레트 (UI 전체)

| 변수 | 값 |
|---|---|
| `base-bg` | `#f3f5f8` |
| `surface` | `#f7f9fc` |
| `surface-soft` | `#eef1f6` |
| `surface-deep` | `#ebedf2` |
| `surface-elev` | `#ffffff` |
| `line` | `#d8deeb` |
| `line-soft` | `rgba(129, 139, 153, 0.32)` |
| `text` | `#1f2630` |
| `text-soft` | `#5f6a78` |
| `accent` | `#7f95a9` |
| `accent-soft` | `#a2b2c5` |
| `focus` | `rgba(127, 149, 169, 0.28)` |
| `ok` | `#4f8f73` |
| `warn` | `#9f8144` |
| `error` | `#ad5f6e` |

규칙: `accent`/`accent-soft`/`focus`는 hover·focus·아이콘 상태에만 사용. 넓은 면적 배경에 파란색 사용 금지.

---

## 6. 접근성

- 패널 토글 버튼: `aria-expanded`, `aria-controls`
- 설정 모달: `role="dialog"`, `aria-modal="true"`
- 패널/모달 열림 시 첫 대화형 요소로 포커스 이동. 닫힘 시 트리거 버튼으로 복귀.
- `prefers-reduced-motion: reduce` 대응.

---

## 7. nginx 설정

- `index.html`: `Cache-Control: no-store` 강제 (재배포 후 구형 해시 번들 참조 차단).
- JS/CSS 정적 파일: immutable 캐시.

---

## 8. 배포

```bash
docker compose up --build -d
docker compose ps   # chat-front, workflow-api 모두 Up 확인
# http://localhost:10000 접속 확인
```
