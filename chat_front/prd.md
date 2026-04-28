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

### 5.8 샘플 질문 버튼 (✅ 구현)

- Composer 우측에 두 개의 샘플 질문 버튼 노출: **쉬운 질문** / **어려운 질문**.
- 각 버튼 클릭 시 `frontend/config/sample_questions.json`의 해당 난이도 풀(`easy` | `hard`)에서 랜덤 1건을 입력창에 채움.
- 풀 구조: `{ "easy": [...], "hard": [...] }`. `useSampleQuestion().pick(difficulty)`가 난이도를 인자로 받음.
- 색상: easy = 녹색 톤, hard = 주황/적색 톤(시각적 난이도 구분).

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
| executor substitution path 1 whole-token 가드 | ✅ 완료 (2026-04-26) |
| synthesizer plan/feature 컨텍스트 주입 | ✅ 완료 (2026-04-26) |
| refiner retry_reason 구조화 + goal 재작성 retry | ✅ 완료 (2026-04-26) |
| refiner confirmed_features를 next_goal에 mix | ✅ 완료 (2026-04-27) |
| refiner confirmed_features anchor 폐기 → seen_feature_ids id-only 누적 | ✅ 완료 (2026-04-27) |
| refiner 후 retrieved docs 단일라인 압축 + history dedup | ⛔ 폐기 (2026-04-27) |
| refiner cross-subtask context 제거 + marked_documents 도입 | ✅ 완료 (2026-04-27) |

### 9.12 refiner cross-subtask context 제거 + marked_documents 도입 (2026-04-27)

9.11 의 raw 문서 단일라인 압축이 final answer 품질을 떨어뜨림 → 전략 전환. refiner 단계에서는 (a) 다른 subtask 의 history/answer 를 prompt 에 주입하지 않고 현재 subtask 의 검색 결과만 평가, (b) raw 문서를 압축하지 않고 그대로 보존, (c) refiner 가 relevant 로 판단한 문서를 `marked_documents` 에 raw text 그대로 적재해 synthesizer 가 final answer 작성 시 직접 참조.

- **state.py**:
  - 9.11 의 `_op: replace_subtask` reducer 분기 제거 → `merge_retriever_history` 원래 append-with-dedup 로 복귀.
  - 신규 reducer `merge_marked_documents`: feature_id 단위 dedup 으로 누적 (첫 등장 보존).
  - `AgentState.marked_documents: List[Dict]` 필드 추가 — 각 entry 는 `{feature_id, feature_name, text (raw), subtask_id, attempt}`.
  - `create_initial_state` 에 `marked_documents=[]` 초기화.
- **refiner_node.py**:
  - 9.11 압축 로직 (`_compress_retriever_outputs`, `_op: replace_subtask` push) 전부 삭제.
  - `_llm_refine_with_verdict` user_content 에서 cross-subtask 섹션 (`전체 실행 계획`, `이전 Subtask 결과`, `누적 reference_features`) 제거 → `사용자 질문 / 현재 Subtask / 검색 결과` 3 섹션만 남김. helper `_format_plan_overview`, `_format_other_subtask_results`, `_format_cumulative_features` 제거 + 사용처 사라진 import 정리.
  - 신규 helper `_collect_marked_documents(retriever_outputs, relevant_feature_ids, excluded_ids, subtask_id, attempt)`: refiner LLM 의 `reference_features` (relevant) ∩ raw 문서, `excluded_doc_ids` 제외, feature_id 단위 dedup. raw text 그대로 보존.
  - `refine_results` 에서 update_state 직전 marked docs 빌드 후 `update_kwargs["marked_documents"]` push.
- **prompts/refiner.py**:
  - "이전 Subtask 결과", "누적 reference_features" 참조 문구 제거.
  - suggested_next_goal 가이드를 (a) intent (b) 보강 정보 (c) 배제 (d) query_hints inline 로 단순화 — 의존 subtask 섹션 제거.
- **호환**: synthesizer 는 기존 `subtask_results.refined_text` + `retriever_history` (3 순위 fallback) 외에 신규 `marked_documents` 를 우선 인용 가능 (synthesizer 측 인덱싱은 후속 PR).
- **기대 효과**: refiner input — attempt-level cross-subtask 컨텍스트 제거로 약 (plan_overview + other_results + cumulative_features) 분량 절감. 동시에 raw 문서는 history 와 marked_documents 양쪽에 남아 final answer 품질/추적성 보존.

### 9.10 refiner confirmed_features anchor 폐기 → seen_feature_ids id-only 누적 (2026-04-27)

verdict=false 시도에서 검색된 feature 를 다음 시도 anchor 로 끌고 가는 의미론 폐기. 실패 시도의 reference_features 는 의미 보장이 없으므로 prompt anchor 로 부적합. attempt 간 id-only dedup + excluded union 만 유지. refiner LLM input/output 토큰 절감 → latency 단축.

- **동기**: 9.9 에서 도입한 confirmed_features dict 누적이 (a) refiner system prompt 의 가이드 (b) "관련 feature 명시" 섹션을 부풀리고, (b) `retry_reason.confirmed_features` 출력 필드를 강제해 LLM 출력 토큰을 늘리고, (c) 다음 시도 goal 본문에 anchor 줄을 붙여 input 토큰까지 늘렸음. verdict=false 라는 사실 자체가 "이 시도의 매칭은 의미 없다" 는 신호 → anchor 로 사용하면 다음 시도가 같은 무관 영역으로 회귀.
- **prompts/refiner.py**:
  - `retry_reason` JSON 스키마에서 `confirmed_features` 필드 제거
  - 규칙 #5 (confirmed_features 기록 지시) 제거. 규칙 #4 에 "이전 시도 무관 항목은 자동 누적" 한 줄 추가.
  - 가이드 (b) 를 "의존 subtask reference 명시" 로 축소 — 의존 subtask 산출물만 anchor 로, 이번 시도 검색 결과는 anchor 금지 명시
  - 두 few-shot 예시에서 `confirmed_features` 필드 + `suggested_next_goal` 본문의 "이번 시도 confirmed" 줄 제거
- **refiner_node.py**:
  - `_normalize_retry_reason` 에서 `confirmed_features` 기본값/정규화 블록 제거
  - `_merge_confirmed_features` 헬퍼 삭제
  - verdict=false 분기 재작성: `subtask["confirmed_features"]` (dict 리스트) 폐기 → `subtask["seen_feature_ids"]` (str 리스트, dedup, excluded 차감) 로 교체. 이번 시도의 reference_features 와 이전 누적 seen 을 union → excluded_doc_ids union 차감 → 결과 저장. 다음 시도 prompt 에 anchor 주입 없음.
  - excluded_doc_ids attempt 간 union 은 동일 분기 안에서 새 goal 재작성보다 먼저 실행 → seen 계산이 최신 excluded 를 반영
- **호환**: state schema 변화는 subtask-local 키만(추가: `seen_feature_ids`, 폐기: `confirmed_features`). retry cap / exceeded / cross-subtask context / 다른 노드 인터페이스 무변경.
- **기대 효과**: refiner system prompt ~10줄 단축 + 출력 JSON 1 필드 감소 + 다음 시도 goal 본문에서 anchor 줄 제거. 시도 횟수가 늘어도 input/output 이 단조 증가하지 않음.

### 9.9 refiner confirmed_features 누적 + suggested_next_goal mix (2026-04-27)

verdict=false 재시도 사이에 partial-match로 확인된 feature_id/name이 드롭되던 문제 해결. retry_reason에 `confirmed_features` 필드 추가, 시도 간 누적 후 새 goal 문장에 anchor로 부착.

- **증상**: ① `state["reference_features"]`는 verdict=true일 때만 머지됨(`refiner_node.py:212`) → 실패 시도 발견 feature 누적 풀에 미반영. ② `subtask["reference_features"]`는 매 refine 호출마다 덮어씀(`refiner_node.py:144`) → 직전 시도 positive 매칭 손실. ③ `suggested_next_goal`은 한 문장 제약 → LLM이 ID를 직접 다시 명시하지 않으면 다음 시도 컨텍스트에서 사라짐. var_constructor/executor가 새 goal 문자열만 보고 partial-match anchor를 잃음.
- **prompts/refiner.py**:
  - `retry_reason` 스키마에 `confirmed_features: [{feature_id, feature_name}]` 추가 — excluded_doc_ids와 중복 금지
  - 규칙 #5: 부분 관련성 확인된 feature를 누락 없이 기록하라는 지시 추가
  - few-shot 예시 도메인을 cellular network로 교체 (5G NR inter-gNB handover missing_info, VoNR mouth-to-ear latency no_results)
- **refiner_node.py**:
  - `_normalize_retry_reason`에 `confirmed_features` 기본값/형식 보정 추가, excluded_doc_ids에 들어간 ID는 자동 제외
  - `_merge_confirmed_features` 헬퍼 — 이전 시도 subtask.confirmed_features + 이번 시도분 dedup 머지
  - `_mix_confirmed_into_goal` 헬퍼 — `suggested_next_goal` 문장 뒤에 `(이미 확인된 관련 feature: FGR-XXX(name), ...)` 앵커 부착 → downstream은 새 goal 문자열만 봐도 anchor 보존
  - verdict=false 분기에서 누적 confirmed_features를 subtask에 저장하고 새 goal에 mix
- **루프/스키마 호환**: cumulative `state["reference_features"]` 머지 가드(line 212)는 유지 — confirmed pool은 subtask-local. retry cap·exceeded 처리·다른 노드 인터페이스 무변경.

verdict=false 시 refiner가 동일 쿼리로 재시도하던 무의미 retry 루프를 retry_reason 구조화 → 다음 시도 goal 재작성 + excluded_doc_ids 후필터로 교체.

- **증상**: 기존 prompt가 `retry_reason: ""` 한 줄 자유 문자열만 요구 → LLM이 한 문장 수준 사유 반환. var_constructor / var_binder / retriever 누구도 retry_reason을 읽지 않음 → 동일 subtask.goal로 동일 문서 N회 재검색하다 3회차 exceeded. retry 자체가 시간·토큰 낭비.
- **prompts/refiner.py 재작성**: `retry_reason`을 dict 스키마로 강제
  - 필드: `failure_type` (irrelevant_docs / missing_info / wrong_entity / partial_match / no_results), `missing_info`, `irrelevant_aspects`, `query_hints[]`, `excluded_doc_ids[]`, `suggested_next_goal`
  - 규칙·failure_type 분류 가이드·few-shot 2건 추가 (missing_info / no_results)
  - `suggested_next_goal` 비어있으면 안 됨 명시 — 비면 retry 무의미
- **refiner_node.py**:
  - `_normalize_retry_reason(raw)` 헬퍼 — dict / 자유 문자열 / None 모두 표준 dict로 정규화
  - verdict=false 분기: subtask에 `retry_reason` (dict) 저장, `suggested_next_goal` 있으면 `subtask.goal`을 그것으로 교체 (원본은 `original_goal`에 보존), `excluded_doc_ids`는 누적 dedupe 머지
- **executor_node.py**: `_execute_retrieve_subtask`가 retriever 호출 후 `subtask.excluded_doc_ids`에 매칭되는 `FGR-XXNNNN`을 포함한 doc을 결과에서 제거(`_filter_excluded_docs`). retriever 툴 시그니처 변경 없이 후필터로 처리.
- **루프 안전망 유지**: refiner의 3회 재시도 cap·exceeded 처리·`route_after_refiner` safety cap 변경 없음. 다만 retry마다 goal과 excluded set이 달라져 의미 있는 시도가 됨.
- **synthesizer 호환**: exceeded 시 subtask.retry_reason은 기존대로 `"최대 재시도 횟수(3회) 초과"` 문자열로 덮여서 synthesizer 출력 변화 없음. 비-exceeded 실패 attempt의 dict retry_reason은 synthesizer가 읽지 않음.

### 9.7 synthesizer plan/feature 컨텍스트 주입 (2026-04-26)

`_generate_llm_response`가 LLM에 `user_query` + 각 subtask의 `subtask_answer` 텍스트만 전달하던 구조를 plan 의도와 누적 feature를 보존해 종합하도록 보강.

- **증상**: state-level 누적 `reference_features`(reducer가 dedupe로 누적한 모든 (id, name) 쌍)가 LLM 컨텍스트에 미주입되어 5개 doc → 5개 feature 모두 추출되었음에도 최종 답변에 일부만 노출되는 회귀 가능
- **수정**: `synthesizer_node.py`에 4개 헬퍼 추가
  - `_format_plan_overview` — 전체 subtasks의 id/status(done/exceeded/pending)/goal preview 표
  - `_format_per_subtask_features` — `subtask_results`의 id별 최신 attempt `reference_features` 압축
  - `_format_cumulative_features` — `state.reference_features` 누적 dedup 리스트
  - `_format_failed_detail` — exceeded subtask의 `retry_reason` 노출
- **user_query 확장**: 위 4개 블록을 "수집된 정보" 앞에 주입, "누적 reference_features의 모든 (feature_id, feature_name) 항목을 답변에 빠짐없이 인용" 지시 추가
- **state·다운스트림 무변경**: synthesizer는 terminal 노드(END 직행)이므로 변경 영향 단방향

### 9.6 executor substitution path 1 whole-token 가드 (2026-04-26)

BINDER 출력 스키마 v3(고정 키 `feature_id`/`feature_name`)와 호환되도록 executor의 substitution path 1을 whole-token 가드로 변경.

- **증상**: path 1이 `if key in updated_goal:` 라는 substring 매칭을 사용해 bare 단어 키(`feature_id`, `feature_name`)가 `$subtask_0.feature_id` placeholder 내부 substring과 매칭되어 blanket `replace`로 placeholder 부분 치환·붕괴
- **수정**: `executor_node.py` `_execute_retrieve_subtask` / `_execute_think_subtask` 두 substitution 루프 모두 path 1 조건을 `key.startswith("$") and key in updated_goal`로 강화 — `$` 접두사 literal 키만 직접 치환, bare 키는 path 3 정규식(`\$subtask_\d+\.{key}`)이 처리
- **호환성**: 기존 `$subtask_N.field` 형태 placeholder 동작 보존, BINDER v3 스키마와 정합

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

### 9.5 synthesizer pre-snapshot 로그 비잘림 (2026-04-28)

`synthesizer_node._generate_llm_response` pre-snapshot 로그의 `content`/`preview`/`query` 필드가 `[:120]`/`[:80]`로 잘려 refine·retriever 결과 전체 확인 불가.

- **수정**: synthesizer snapshot 로그의 모든 slice (`[:120]`, `[:80]`) 제거 → 전문 출력. 기존 per-line `logger.info` 구조는 유지.

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
