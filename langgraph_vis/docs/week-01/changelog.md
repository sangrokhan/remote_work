# Week 01 Changelog

## v1.0.0 (2026-04-14)

- `GET /api/workflows/{id}/schema` 계약의 v1 응답 및 에러 형식을 확정.
- `workflowId`, `nodeId`, `stateKey` 네임스페이스 및 형식 규칙을 정의.
  - workflowId: `^[a-z][a-z0-9_-]{1,63}$`
  - nodeId: `^[a-z][a-z0-9_-]{1,63}$`
  - stateKey: `^[a-z][a-zA-Z0-9_-]{0,127}$`
- 핵심 필수 필드 확정: `schemaVersion`, `workflowId`, `version`, `nodes`, `edges`.
- 노드 필수 필드 확정: `id`, `label`, `description`, `stateKey`, `order`.
- 에러 응답 통일: `code`, `message`, `requestId`.
- metadata 파생 금지 원칙 확정(원본 값만 전달).
- 버전 호환성 규칙 채택
  - 필드 추가: 호환
  - 필수 필드 삭제/타입 변경: 비호환

## 참고

- 하위 호환성 변경이 필요할 경우 `schemaVersion` bump 후 `docs/week-01/context-to-week02.md`와 handoff 항목 동기화 후 반영.
