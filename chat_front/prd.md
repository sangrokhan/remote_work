# 요구사항 명세서: Agentic RAG 채팅 인터페이스 + 워크플로우 시각화

## 1. 서비스 개요

React/Vite 기반 채팅 UI와 FastAPI/LangGraph 기반 백엔드로 구성된 워크플로우 실행 시각화 시스템.  
사용자가 채팅으로 입력하면 LangGraph 워크플로우가 실행되고 우측 패널에서 노드 실행 상태를 실시간 확인한다.

---

## 2. 시스템 구성

### 2.1 서비스 및 포트

| 서비스 | 기술 스택 | 포트 |
|---|---|---|
| `chat-front` | React + Vite → `vite preview` | `10000` |
| `workflow-api` | FastAPI + SSE | `10001` |

### 2.2 백엔드 URL

`.env` 파일로 빌드 타임 주입:

```
VITE_WORKFLOW_GRAPH_URL=http://<호스트>:10001/graph
VITE_WORKFLOW_RUN_URL=http://<호스트>:10001/api/run
```

기본값: `http://localhost:10001/{graph|api/run}`

---

## 3. 백엔드 요구사항

### 3.1 엔드포인트

| 메서드 | 경로 | 설명 | 상태 |
|---|---|---|---|
| `GET` | `/health` | `{"status": "ok"}` | ✅ 구현 |
| `GET` | `/graph` | 워크플로우 그래프 스키마 `{nodes, edges}` | ✅ 구현 |
| `POST` | `/api/run` | SSE 워크플로우 실행 스트림 | ✅ 구현 |

> WebSocket(`/ws/connect`) 제거됨 — REST+SSE로 대체.

### 3.2 RunWorkflowRequest

```python
class RunWorkflowRequest(BaseModel):
    run_id: str
    input: str
    model: str = "GaussO4"
    response_mode: str = "normal"
    max_tokens: int = 1024
    agentic_rag: bool = False
    api_url: str = ""
    api_key: str = ""
```

### 3.3 SSE 이벤트 프로토콜

**스트림 순서:**

1. `run_started` — 모델/RAG 정보 포함
2. `workflow_event` × N — 노드별 실행 이벤트
3. `workflow_complete` 또는 `workflow_error`

**이벤트 구조:**

```json
{
  "event": "workflow_event",
  "node": "planner",
  "name": "planner",
  "stage": "start",
  "message": "planner 실행됨",
  "payload": {}
}
```

`stage ∈ {start, end, routing, error}`

### 3.4 워크플로우 그래프 구조

**데모 그래프** (`stategraph_workflow.py`):

```
START → planner → executor → refiner → (planner | synthesizer)
                            ↘ synthesizer → END
```

**Agentic RAG 그래프** (`langgraph_flow/agents/graph.py`, WIP):

```
START → retriever → var_constructor → var_binder → planner → executor → (refiner | synthesizer) → END
```

---

## 4. LangGraph 에이전트 구조 (WIP)

### 4.1 AgentState

```python
class AgentState(TypedDict):
    input: str
    agentic_rag: bool
    planner_output: str
    executor_output: str
    refiner_output: str
    final_output: str
    retriever_output: str
    var_bindings: str
    hop_count: int
```

### 4.2 노드 목록

| 노드 | 파일 | 상태 |
|---|---|---|
| planner | `nodes/planner_node.py` | 스텁 |
| executor | `nodes/executor_node.py` | 스텁 |
| refiner | `nodes/refiner_node.py` | 스텁 |
| synthesizer | `nodes/synthesizer_node.py` | 스텁 |
| retriever | `nodes/retriever_node.py` | 스텁 |
| var_constructor | `nodes/var_constructor_node.py` | 스텁 |
| var_binder | `nodes/var_binder_node.py` | 스텁 |

### 4.3 LLM 팩토리

```python
# langgraph_flow/core/factory.py
get_llm(model_name: str, api_url: str, api_key: str) -> BaseLLM
```

| 모델 키 | 클래스 | 파일 |
|---|---|---|
| `GaussO4` | `GaussO4` | `models/gauss_o4.py` |
| `GaussO4-think` | `GaussO4Think` | `models/gauss_o4_think.py` |
| `Gemma4-E4B-it` | `Gemma4E4BIt` | `models/gemma4_e4b_it.py` |

노드는 `config["configurable"]["llm"]`으로 LLM 인스턴스를 수신한다.

---

## 5. 프론트엔드 요구사항

### 5.1 레이아웃

```
┌──────────────────────────────────────┬────────────┐
│  헤더 (타이틀 | 설정버튼 | 토글버튼)    │            │
├──────────────────────────────────────┤  우측패널   │
│  [좌 패널]       [우 패널 - 분할모드]  │  워크플로우 │
│  메시지 영역      메시지 영역          │  그래프    │
│  입력 영역        입력 영역            │            │
└──────────────────────────────────────┴────────────┘
```

- 우측 패널: 기본 닫힘. 열리면 채팅 영역 25% 축소.
- 스플릿 모드(`isSplitMode`): 채팅 영역 좌/우 두 패널로 분할.

### 5.2 스플릿 패널 (✅ 구현)

- 분할 시 좌/우 각 패널이 독립적인 모델 + AgenticRAG 설정 보유.
- 전송 시 두 패널 각각 독립적으로 `POST /api/run` SSE 스트림 요청.
- 기본값: AgenticRAG **ON**, 모델 **GaussO4**.
- 각 패널 헤더에 모델 선택 드롭다운 + AgenticRAG 토글.

### 5.3 SSE 플로우 (✅ 구현)

`useWorkflowSSE.streamWorkflow()`:
1. `fetch(VITE_WORKFLOW_RUN_URL, {method: POST, body: req})`
2. `ReadableStream` 라인 파싱
3. `run_started` → 모델/RAG 정보 말풍선 추가
4. `node_started` → `applyWorkflowNodeHighlight` 호출
5. `workflow_complete` / `workflow_error` → 하이라이트 제거, 상태 업데이트

### 5.4 워크플로우 그래프 시각화 (✅ 구현)

- `GET /graph` (REST)로 그래프 스키마 로드 — WebSocket 불필요.
- Cytoscape.js + dagre 레이아웃 (`rankDir: TB`, `zoom: 1`).
- `cyRef`를 `App.jsx`에서 생성해 `useWorkflowSocket`과 `useWorkflowGraph` 공유.
- `content: (node) => node.data('label')` 사용. `label:` 속성 금지.
- `max-width`, `max-height`, `shadow-*`, `wheelSensitivity` 금지.

### 5.5 노드 색상 팔레트

| 노드 | bg | border | text |
|---|---|---|---|
| planner | `#d7ecff` | `#68a8ee` | `#12365f` |
| executor | `#d7f4dd` | `#7dcf90` | `#1a4f2f` |
| refiner | `#fff1c7` | `#e2be5e` | `#5e4b17` |
| synthesizer | `#f5d9fc` | `#c78ce0` | `#5d2e69` |
| retriever | `#fde8d0` | `#e09050` | `#5a2e0a` |
| var_constructor | `#d0f0e8` | `#50b090` | `#0a3a28` |
| var_binder | `#e8d0f0` | `#9050b0` | `#2a0a3a` |
| start | `#e7e7f8` | `#9ca2df` | `#32366c` |
| end | `#dceeff` | `#5f8bb0` | `#263246` |
| default | `#edf2ff` | `#a9b6d7` | `#2f3c56` |

### 5.6 스크롤 UX

- 새 메시지 시 자동 스크롤.
- 스크롤바: 기본 숨김, 스크롤 동작 시 1.2초 표시 후 재숨김.

### 5.7 설정 모달

- 응답 모드, 최대 토큰 설정.
- 닫기: 닫기 버튼 / 배경 클릭 / ESC.

---

## 6. UI 컬러 팔레트

| 변수 | 값 |
|---|---|
| `base-bg` | `#f3f5f8` |
| `surface` | `#f7f9fc` |
| `surface-soft` | `#eef1f6` |
| `surface-deep` | `#ebedf2` |
| `surface-elev` | `#ffffff` |
| `line` | `#d8deeb` |
| `text` | `#1f2630` |
| `text-soft` | `#5f6a78` |
| `accent` | `#7f95a9` |
| `ok` | `#4f8f73` |
| `warn` | `#9f8144` |
| `error` | `#ad5f6e` |

---

## 7. 접근성

- 패널 토글: `aria-expanded`, `aria-controls`
- 설정 모달: `role="dialog"`, `aria-modal="true"`
- `prefers-reduced-motion: reduce` 대응

---

## 8. 배포

```bash
docker compose up --build -d
docker compose ps   # chat-front, workflow-api 모두 Up 확인
# http://localhost:10000
```

---

## 9. 구현 현황

| 항목 | 상태 |
|---|---|
| FastAPI SSE 엔드포인트 (`POST /api/run`) | ✅ |
| `RunWorkflowRequest` 파싱 및 로깅 | ✅ |
| 그래프 스키마 REST 엔드포인트 (`GET /graph`) | ✅ |
| 스플릿 패널 + 독립 SSE 스트림 | ✅ |
| Cytoscape 워크플로우 시각화 + 하이라이트 | ✅ |
| `AgenticRAGGraph` 클래스 및 노드 스텁 | ✅ |
| `BaseLLM` ABC + 모델 스텁 3종 | ✅ |
| `get_llm` 팩토리 | ✅ |
| 실제 LLM 호출 (노드 구현) | ⬜ WIP |
| BGE3 임베딩 기반 retriever | ⬜ WIP |
| `stategraph_workflow.py` → `langgraph_flow` 이전 | ⬜ WIP |
