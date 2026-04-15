# Week 01 Identifier & Required Field 규칙 (최종 정리)

## 1) 식별자 네임스페이스 정책

### workflowId
- 패턴: `^[a-z][a-z0-9_-]{1,63}$`
- 최소 2자 ~ 최대 64자
- 범위: 글로벌 유일
- 충돌 처리: 충돌 시 조회/실행 요청은 오류로 판단

### nodeId
- 패턴: `^[a-z][a-z0-9_-]{1,63}$`
- 범위: 동일 workflowId 안에서 유일
- event key 구성 시 권장: `workflowId::nodeId`

### stateKey
- 패턴: `^[a-z][a-zA-Z0-9_-]{0,127}$`
- 범위: 동일 workflow 내 유일 권장
- 상태/이벤트 매핑 기본 키로 사용

## 2) 필수/선택 규칙(요청/응답 계약)

### WorkflowSchemaResponse (최상위)
- 필수: `schemaVersion`, `workflowId`, `version`, `nodes`, `edges`
- 선택: `workflowName`

### WorkflowNode
- 필수: `id`, `label`, `description`, `stateKey`, `order`
- 선택: `metadata`

### WorkflowEdge
- 필수: `from`, `to`
- 선택: `id`, `label`, `metadata`

## 3) 정합성 규칙
- `metadata`는 파생 금지, backend source-of-truth 원천 값만 전달
- `schemaVersion`은 하위 호환 정책(`필드 추가 허용, 필수 삭제/타입 변경 금지`) 아래 관리
- `nodeId/stateKey` 충돌이 발생하면 week02 이벤트 바인딩/조회에서 일관성 오류 가능성 높음
