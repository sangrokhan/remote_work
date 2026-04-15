# Week 1 고위험 항목 해결 계획 (Week 2 영향 고려)

작성일: 2026-04-14
작성 위치: /home/han/.openclaw/workspace/remote_work/langgraph_vis/docs/week-01/issue-resolution-plan.md

## 컨텍스트
현재 Week 1은 contract 테스트가 pass 상태이며 checklist은 완료로 표시되어 있으나, subagent 4개 합의 평가는 다음 항목을 Week 2/3 연계 상 위협으로 판정했습니다.

- `schema-registry`의 식별자 충돌 미검증(워크플로우/노드/상태키/엣지)
- 405 응답의 에러 메시지 비표준
- (보완적) 5xx 구성 시 내부 원인 텍스트 노출 가능성

관련 파일
- [src/schema-registry.mjs](/home/han/.openclaw/workspace/remote_work/langgraph_vis/src/schema-registry.mjs)
- [src/schema-api-server.mjs](/home/han/.openclaw/workspace/remote_work/langgraph_vis/src/schema-api-server.mjs)
- [src/error-contract.mjs](/home/han/.openclaw/workspace/remote_work/langgraph_vis/src/error-contract.mjs)
- [tests/schema/schema-api.spec.mjs](/home/han/.openclaw/workspace/remote_work/langgraph_vis/tests/schema/schema-api.spec.mjs)
- [tests/schema/contract.spec.mjs](/home/han/.openclaw/workspace/remote_work/langgraph_vis/tests/schema/contract.spec.mjs)

## 해결 항목 및 적용 방식

### 1) `schema-registry` 중복 식별자 강제 (우선순위: 높음)
- 목표: Week2 이벤트 라우팅 충돌을 사전 차단
- 범위:
  - `load()`에서 `workflowId` 중복 적재 방지
  - `validateWorkflowPayload()`에서 `node.id`, `stateKey`, `edge.id` 중복 방지
  - `node.order` 음수/비정수 방지 강화 (`>= 0`)
  - `edge.from`, `edge.to` 필수 문자열 및 정합성 검사 강화
- 영향: `WorkflowSchemaRegistry`에서 `WorkflowApiError`로 명확 실패를 반환해 API 레벨로 전파됨
- 검증: `tests/schema/registry.spec.mjs` 신규 추가

### 2) `schema-api` 비-GET 메서드 에러 메시지/헤더 통일 (우선순위: 높음)
- 목표: 운영 자동화/모니터링에서 실패 코드 메시지 분류 일관성 확보
- 범위:
  - `createWorkflowSchemaHandler`의 non-GET 처리 메시지를 `ERROR_MESSAGES.INVALID_WORKFLOW_PAYLOAD`로 고정
  - `Allow` 헤더를 `GET`으로 반환
- 검증: `tests/schema/schema-api.spec.mjs`에 `POST`/`PUT` 비-GET 테스트 추가

### 3) 내부 실패 메시지 고정 + 원인 보존(선택)
- 목표: 클라이언트 응답 과잉 노출 최소화(감사/보안)
- 범위:
  - `schema-registry`의 manifest parse 에러를 공통 에러 메시지로 감싸고(기존 동작 유지), 필요 시 에러 원인은 내부 로깅/디버그 경로로 이동
- 검증: API 5xx 테스트는 고정 메시지 계약 유지

## 반영 후 기대 결과
- Week 2 handoff에서 node/state 기반 이벤트 매핑 충돌 확률 저감
- 에러 관측성(`code/message/requestId`)과 메시지 일관성 강화
- checklist의 구현 충족 근거가 강화되고 회귀 탐지 범위 확대
