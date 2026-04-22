
# LangGraph Workflow 실행 시각화 설계

## 문서 목적

LangGraph 기반 workflow의 실행 과정을 웹페이지에서 시각화하기 위한 설계 방향을 정의한다. 본 문서는 구현 상세보다 전체 구조와 설계 문서의 작성 범위를 정리하는 목차 초안에 집중한다.

## 1. 배경
- 현재 LangGraph로 구성된 workflow가 존재함
- 사용자 질문 이후 각 노드의 실행 과정을 웹에서 확인하고 싶음
- backend workflow 변경 시 frontend 수정 부담을 최소화하고 싶음

## 2. 목표
- workflow 실행 상태를 실시간으로 시각화
- backend에서 수신한 JSON을 JavaScript로 파싱/정규화한 뒤 HTML로 렌더링
- 각 노드의 시작, 실행 중, 완료, 실패 상태를 표시
- 우측 사이드바 임베드 형태로 표시되어 다른 시스템 화면에 병행 탑재 가능해야 함
- frontend가 backend의 workflow 구조를 동적으로 받아 렌더링
- workflow 변경 시 frontend 하드코딩을 최소화

## 3. 범위
### 3.1 포함 범위
- LangGraph workflow 구조 조회 방식
- 실행 이벤트 스트리밍 구조
- frontend 시각화 방식
- 상태 조회 및 실행 이력 조회 방식
- API 및 이벤트 모델 초안

### 3.2 제외 범위
- 세부 UI 디자인 시안
- 배포 인프라 상세
- 인증/권한 체계 상세 구현
- 운영 모니터링 외부 도구 연동

## 4. 핵심 요구사항
### 4.1 기능 요구사항
- workflow schema 조회 가능
- run 시작 가능
- run 중 실시간 이벤트 수신 가능
- 현재 상태 및 최종 상태 조회 가능
- 필요 시 실행 이력 조회 가능

### 4.2 비기능 요구사항
- frontend와 backend 간 결합도 최소화
- 새로운 노드 추가 시 frontend 코드 수정 최소화
- 긴 실행에서도 UI가 자연스럽게 갱신될 것
- 실패 지점 식별이 쉬울 것
- 외부 시스템 연동 시 독립 마운트 가능한 우측 사이드바 레이아웃 제공

## 5. 상위 아키텍처
### 5.1 구성 요소
- LangGraph workflow 실행 계층
- workflow schema 제공 API
- run 실행 및 이벤트 스트림 API
- state/history 조회 API
- 범용 workflow renderer frontend

### 5.2 데이터 흐름
- frontend가 workflow schema 조회
- 사용자가 질문 입력 후 run 생성
- backend가 graph 실행 및 이벤트 stream 전송
- frontend가 JSON 이벤트를 JavaScript reducer로 파싱/반영해 노드 상태 갱신
- 갱신 결과를 HTML로 렌더링
- 완료 후 최종 state 또는 history 조회

## 6. Workflow 구조 조회 설계
### 6.1 backend를 Source of Truth로 두는 방식
- workflow 정의 및 메타정보는 backend가 관리
- frontend는 schema를 받아 화면만 생성

### 6.2 schema 구성 항목
- workflow id
- node 목록
- edge 목록
- 시작 노드 / 종료 노드
- 각 node의 label, 설명, stateKey, 정렬 정보

### 6.3 schema 제공 방식
- graph introspection 기반 동적 생성
- 또는 별도 manifest 기반 제공
- 두 방식의 장단점 비교 필요

## 7. Run 실행 및 이벤트 스트리밍 설계
### 7.1 실행 방식
- backend에서 graph 단위로 `stream()` 또는 `astream()` 실행
- 각 노드 실행 과정에서 발생한 이벤트를 frontend로 전달

### 7.2 이벤트 종류
- run_started
- node_started
- node_progress
- node_token
- node_completed
- node_failed
- run_completed

### 7.3 custom progress 이벤트
- LangGraph 기본 이벤트만으로 부족한 진행률은 custom event로 보강
- 노드 내부에서 세부 진행률 또는 중간 메시지 emit 가능

## 8. Backend 설계
### 8.1 workflow registry
- 여러 workflow를 식별하고 조회할 수 있는 구조 필요

### 8.2 run lifecycle 관리
- run_id 또는 thread_id 생성
- 실행 시작, 진행 중, 완료, 실패 상태 관리

### 8.3 state 조회
- 현재 실행 상태 조회
- 최종 결과 조회
- 필요 시 state history 조회

### 8.4 에러 처리
- 노드 단위 실패 이벤트 전송
- 전체 run 실패 상태 정리
- 재시도 또는 재실행 정책 검토

## 9. Frontend 설계
### 9.1 generic renderer
- schema를 기반으로 노드와 연결 구조를 렌더링
- workflow별 하드코딩 금지
- schema/event/state는 JSON 수신 후 JS 뷰 모델로 변환해 DOM을 구성

### 9.2 표현 방식 후보
- step 카드 뷰
- DAG 뷰
- 타임라인 뷰
- 초기 버전에서 무엇을 채택할지 결정 필요
- 우측 사이드바 내 임베드 기준 뷰를 기본 후보로 추가

### 9.3 상태 모델
- pending
- running
- completed
- failed
- skipped

### 9.4 출력 표시 방식
- 실행 중 토큰 스트리밍 표시
- 완료 후 node별 결과 고정
- 최종 응답과 중간 산출물 분리 여부 검토
- 모든 출력은 JSON 정규화 결과를 JS가 계산해 HTML로 반영

## 10. API 설계 초안
### 10.1 workflow schema 조회 API
- `GET /api/workflows/{id}/schema`

### 10.2 run 생성 API
- `POST /api/workflows/{id}/runs`

### 10.3 실시간 stream API
- `GET /api/runs/{run_id}/stream`
- SSE 또는 WebSocket 중 선택 필요

### 10.4 상태 조회 API
- `GET /api/runs/{run_id}/state`
- `GET /api/runs/{run_id}/history`

## 11. 데이터 모델 초안
### 11.1 WorkflowSchema
- workflowId
- version
- nodes[]
- edges[]

### 11.2 WorkflowNode
- id
- label
- description
- stateKey
- order
- metadata

### 11.3 RunEvent
- eventType
- runId
- nodeId
- timestamp
- payload

## 12. UI/UX 고려사항
- 실행 중인 노드를 직관적으로 강조할 것
- 실패 노드는 원인 파악이 쉽게 보일 것
- 긴 workflow에서도 화면이 과도하게 복잡해지지 않을 것
- 모바일 대응 필요 여부 검토
- 우측 사이드바 고정/폭 변경/접힘으로 외부 시스템에서 재사용성 확보

## 13. 성능 및 확장성 고려사항
- 이벤트 빈도가 높을 때 프론트 렌더링 부하 관리
- 긴 토큰 스트림 처리 방식 검토
- 다중 사용자/다중 run 동시 처리 고려
- 대형 workflow에서 schema 및 그래프 렌더링 비용 검토

## 14. 구현 단계 제안
### 14.1 1단계
- schema 조회
- run 실행
- 노드 상태 표시

### 14.2 2단계
- 토큰 스트리밍 표시
- 중간 산출물 표시

### 14.3 3단계
- state/history 조회
- 재실행 또는 특정 시점 복기 기능 검토

### 14.4 4단계
- UI 고도화
- DAG/타임라인 전환
- 필터링 및 탐색 기능 추가

## 15. 오픈 이슈
- schema는 introspection과 manifest 중 무엇을 기준으로 할지
- stream transport는 SSE와 WebSocket 중 무엇이 적절한지
- 초기 UI는 step 카드 뷰와 DAG 뷰 중 무엇을 우선할지
- node별 결과를 어느 수준까지 일반화해 표시할지
- state/history를 실시간 UI와 어떻게 연결할지

## 16. 결론
본 설계의 핵심은 backend가 workflow 구조와 실행 이벤트의 단일 진실 공급원이 되고, frontend는 이를 해석하여 범용 시각화 UI를 제공하는 구조를 만드는 것이다. 이를 통해 backend 변경 시 frontend 수정 비용을 줄이고, LangGraph workflow 실행 과정을 유연하게 시각화할 수 있다.
