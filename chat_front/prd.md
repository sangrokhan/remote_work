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

**실 그래프** (`langgraph_flow/agents/graph.py`, `stategraph_workflow.py` 대체 완료):

```
START → planner → executor → (refiner | synthesizer) → END

Agentic RAG:
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
| `stategraph_workflow.py` → `langgraph_flow` 이전 | ✅ 완료 (Task 6, 2026-04-22) |
| 엔드-투-엔드 배포 검증 (Docker, SSE, 오류 처리) | ✅ 완료 (Task 6, 2026-04-22) |
| 실제 LLM 호출 (노드 구현) | ⬜ WIP |
| BGE3 임베딩 기반 retriever | ⬜ WIP |
| Subtask 바인딩 placeholder 통일 (`$task_N` → `$subtask_N`) | ✅ 완료 (2026-04-26) |
| var_binder fallback 다중 feature 보존 | ✅ 완료 (2026-04-26) |
| executor auto-resolve 다중 feature 보존 | ✅ 완료 (2026-04-26) |
| refiner cross-subtask context 주입 | ✅ 완료 (2026-04-26) |
| var_binder LLM 경로 multi-value 프롬프트 보강 | ✅ 완료 (2026-04-26) |

### 9.5 var_binder LLM 경로 multi-value 프롬프트 보강 (2026-04-26)

`_resolve_bindings_with_llm` 경로에서 LLM이 reference_features 다중 항목 중 첫 항목만 JSON으로 반환해 N→1 collapse가 발생하는 잔존 회귀 vector 차단.

- **증상**: var_binder fallback은 dedupe+join 적용되었으나 LLM 경로(운영 기본 경로)는 BINDER_SYSTEM_PROMPT가 단일값 가정이라 collapse 가능
- **수정 v1**: 규칙 + 4개 few-shot으로 "여러 항목 → 공백 join 단일 문자열"을 강제
- **수정 v2**: pair 형태(`FGR-X Name, ...`) 단일 문자열 — substring 충돌 우려로 폐기
- **수정 v3 (현재)**: 출력을 두 고정 필드 `feature_id` / `feature_name`을 갖는 단일 JSON 객체로 정의
  - bindings의 binding_key는 무시, previous_results 전체에서 (id, name) 쌍 dedupe(등장 순서 보존) 후 두 필드에 같은 순서로 `, ` join
  - 두 필드는 항상 같은 항목 수·순서 유지 (인덱스 정합성)
  - feature_name 빈 자리 보존(콤마 유지), feature_id 빈 항목은 스킵
  - 매칭 없음 → 두 필드 모두 빈 문자열
  - few-shot 6 케이스 (다중 / 단일 / dedupe / 다중 subtask 통합 / 빈 자리 / 매칭 없음)
- **호환성 메모**: 본 스키마는 executor substitution path 3 (`\$subtask_\d+\.{key}` 정규식)과 정합. 단 path 1 (`if key in goal`) 의 substring 매칭 가드 부재로 `feature_id` 키가 `$subtask_0.feature_id` placeholder 내부 substring을 blanket replace할 위험 — 별도 코드 fix(path 1을 `$<key>` whole-token 가드로 변경) 필요.

### 9.4 refiner cross-subtask context 주입 (2026-04-26)

`_llm_refine_with_verdict`가 LLM에 `user_query` + 현재 subtask + 현재 검색 결과만 전달하던 격리된 refine 구조를 cross-subtask 정합성을 갖도록 보강.

- **증상**: refiner가 다른 subtask가 이미 추출한 feature/answer를 모름 → 중복 판단·정합성 검증 불가, 여러 subtask가 동일 feature를 중복 보고하거나 누락 위험
- **수정**: `refiner_node.py`에 3개 헬퍼 추가
  - `_format_plan_overview` — 전체 subtasks의 id/status(current/done/exceeded/pending)/goal preview 표
  - `_format_other_subtask_results` — 현재 외 verdict=True 결과의 (id별 최신 attempt) subtask_answer(300자 캡) + reference_features 압축
  - `_format_cumulative_features` — `state.reference_features` 누적 dedup 리스트
- **user_content 확장**: 위 3개 블록을 "현재 Subtask" 앞에 주입, "이전 결과·누적 feature를 고려해 중복은 제외하고 신규 feature는 누락 없이 포함" 지시 추가
- **데드코드 제거**: 미사용 `subtask_context` 지역변수 삭제
- **토큰 캡**: subtask_answer 300자, goal preview 80자 제한으로 토큰 폭증 방지
- **공유 유틸 추출 (2026-04-26 리팩터)**: `agents/_subtask_utils.py` 신설 — `result_payload` / `pick_latest_successful` / `truncate` / `format_features`. refiner·var_binder 양쪽이 envelope 평면화·최신 attempt 선택·feature 포맷·문자열 자르기를 공유. 매직넘버는 `GOAL_PREVIEW_MAX`/`ANSWER_PREVIEW_MAX` 모듈 상수로 승격, dead `lambda x:(x is None,x)` 정렬·`.rstrip(": ")` 핵 제거.

### 9.3 executor auto-resolve 다중 feature 보존 (2026-04-26)

`_execute_retrieve_subtask` fallback이 `$subtask_N.field` placeholder 치환 시 첫 매치만 사용하여 var_binder가 못 푼 placeholder의 다중 feature가 단수로 collapse되는 버그 수정.

- **증상**: var_binder 단계에서 해결 못한 placeholder가 executor로 넘어와 auto-resolve될 때 5개 feature 중 1개만 retriever 쿼리에 임베딩됨
- **수정**: `executor_node.py:343-355`에서 `for feat in ref_features` 첫-매치-break 루프를 dedupe collect-all 후 공백 join으로 교체 (var_binder fallback과 동일 정책)
- **다운스트림 영향 없음**: 기존 `goal.replace(placeholder, str(value))` 와 호환 — 다중 값은 `"FGR-A FGR-B"` 공백 구분 문자열로 substitute

### 9.2 var_binder fallback 다중 feature 보존 (2026-04-26)

기존 fallback이 `reference_features` 리스트에서 첫 매치만 채택해 N개 feature 중 1개만 다음 subtask로 전달되는 버그 수정.

- **증상**: 5개 문서에서 5개 feature_id 추출했음에도 binding resolution 후 단일 ID만 retriever 쿼리에 임베딩됨
- **수정**: `_resolve_bindings_fallback` (`var_binder_node.py`)에서 `field_name` 값들을 모든 `reference_features` 엔트리에서 수집·dedupe(순서 보존) 후 공백 join으로 단일 문자열 반환
- **regex fallback 동기화**: `subtask_answer` + `refined_text`에서 `feature_id` 추출도 `re.search` → `re.findall` + dedupe로 변경
- **다운스트림 영향 없음**: 기존 substitution 코드(`goal.replace(placeholder, str(value))`)와 호환 — 단일 값일 때는 기존 단일 ID, 다중 값일 때는 `"FGR-A FGR-B"` 공백 구분 문자열로 retriever에 전달됨

### 9.1 Subtask placeholder 통일 (2026-04-26)

LLM 바인딩 참조 문법을 `$task_N` / `$subtask_N` 혼용에서 `$subtask_N` 단일 형식으로 통합. 상태 키 `subtasks`/`subtask_results`/`subtask_id`와 의미 정합성 확보.

- **포맷**: `$subtask_{id}.{field}` 또는 `${subtask_{id}.{field}}` (id는 0-base 정수, validator가 강제)
- **파서 수정**: hardcoded `_0` 제거 → 정규식 `\$subtask_(\d+)\.(\w+)` 적용 → 임의 N 매칭
- **프롬프트 동기화**: `prompts/planner.py`, `prompts/var_binder.py` — 예시를 `$subtask_0.field`로 변경
- **부수 정리**: dead key fallback `r.get("task_id")` 제거 (refiner는 `id`만 기록), Python 변수 `task_id` → `subtask_id` 일관화 (executor / var_binder / synthesizer / refiner / routing_logic)
- **planner 정규화**: `subtask.get("subtask_id", ...)` 듀얼 리드 제거 → `subtask.get("id", i)` 단일 키 사용
