# Week 02 Hand-off (from Week 01)

## 전달 항목(필수)

- `workflowId` 규칙
  - 형식: `^[a-z][a-z0-9_-]{1,63}$`
  - 범위: 전역 유일
  - 네임스페이스 규칙: event 키는 `workflowId::nodeId` 권장
- `nodeId` 규칙
  - 형식: `^[a-z][a-z0-9_-]{1,63}$`
  - 범위: 동일 workflow 내 유일
- `stateKey` 규칙
  - 형식: `^[a-z][a-zA-Z0-9_-]{0,127}$`
  - 범위: 권장되는 workflow 내 유일
- `schemaVersion`
  - 고정값: `1.0.0`
  - 호환성: 필드 추가 허용, 필수 필드 삭제/타입 변경 비허용

## Week 02 payload 연동 가이드

- run 시작 payload는 `workflowId`와 `workflowVersion`(선택)을 기준으로 대상 workflow를 식별한다.
- run 이벤트의 노드 식별은 `nodeId`를 기본 키로 사용한다.
- 상태 조회는 `stateKey`를 사용하고, 이벤트가 `stateKey`를 누락할 경우 nodeId 기반 lookup 보완 규칙 적용.
- 모든 실패 응답은 공통 에러 구조 사용:
  - `code`
  - `message`
  - `requestId`

## 오류 코드 동기화

- Week 01 고정 코드
  - `WORKFLOW_NOT_FOUND`
  - `INVALID_WORKFLOW_ID`
  - `INVALID_WORKFLOW_PAYLOAD`
  - `WORKFLOW_REGISTRY_ERROR`
  - `INTERNAL_ERROR`

## 테스트/문서 연동 포인트

- Week 02 API/이벤트 테스트에서 다음 조합은 고정값으로 가정:
  - `workflowId`, `nodeId`, `stateKey`, `schemaVersion`
- Week 01 fixture(`schema-contract.fixture.json`) 1건을 Week 02 기본 통합 스키마로 재사용 가능.
