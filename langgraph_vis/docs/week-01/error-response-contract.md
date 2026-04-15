# Week 01: 실패 응답 통합 계약

## 표준 실패 응답 형식 (필수)

모든 실패 응답은 다음 3개 필드를 공통으로 반환한다.

- `code`: 고정 에러 코드
- `message`: 사용자/운영자 식별 가능한 메시지
- `requestId`: 추적용 고유 식별자

## 허용 에러 코드

- `WORKFLOW_NOT_FOUND`
- `INVALID_WORKFLOW_ID`
- `INVALID_WORKFLOW_PAYLOAD`
- `WORKFLOW_REGISTRY_ERROR`
- `INTERNAL_ERROR`
- `RUN_NOT_FOUND`
- `INVALID_RUN_ID`
- `INVALID_RECONNECT_QUERY`
- `INVALID_RUN_TRANSITION`

## 계약 매핑

- `GET /api/workflows/{id}/schema`
  - 400: `INVALID_WORKFLOW_ID`, `INVALID_WORKFLOW_PAYLOAD`
  - 404: `WORKFLOW_NOT_FOUND`
  - 500: `WORKFLOW_REGISTRY_ERROR`, `INTERNAL_ERROR`
- `GET /api/runs/{runId}/state`, `GET /api/runs/{runId}/events`, `GET /api/runs/{runId}/events/stream`
  - 400: `INVALID_RUN_ID`, `INVALID_RECONNECT_QUERY`
  - 404: `RUN_NOT_FOUND`
  - 409: `INVALID_RUN_TRANSITION` (상태 전이 실패 시)
  - 405: `INVALID_RUN_PAYLOAD` (method 미지원)
  - 500: `INTERNAL_ERROR`

## run API 상세 계약 참조

- [week-02 run API 계약](/home/han/.openclaw/workspace/remote_work/langgraph_vis/docs/week-02/run-api-contract.yaml): run API의 `code`, `message`, `requestId` 계약 및 커서 규격 일원화.

## 정합성 규칙

- 1개 에러 응답에도 항상 `code`, `message`, `requestId` 필수
- `code`는 시스템 내부 예외 메시지에서 직접 노출 대신 계약 코드로 치환
- `message`는 사용자/운영자 모두가 검색 가능한 형태로 관리
- week02에서는 동일 스키마를 `run`/`state` API 에러 응답에도 동일 적용
