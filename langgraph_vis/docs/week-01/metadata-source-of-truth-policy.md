# Week 01: metadata Source-of-Truth 정책 (파생 금지)

## 원칙

- `nodes[].metadata`와 `edges[].metadata`는 **backend source-of-truth 원천 데이터**만 전달한다.
- UI/Frontend는 `metadata`를 표시/활용할 수는 있지만, 계산해 재생성하지 않는다.
- schema API는 이벤트 중계/집계 과정에서 새 필드를 주입하지 않는다.

## 금지 항목

- UI 상태값 기준 자동 보정(예: `metadata.displayLabel = "..."` 생성)
- 클라이언트가 렌더 시점에 node 결과를 기준으로 추가 파생 필드 생성
- stream 처리 중 임시 계산 값을 schema 응답의 `metadata`로 반영

## 허용 항목

- backend가 schema/registry 소스에서 명시적으로 넣은 필드 제공
- backend가 필요 시 `metadata` 키 하위에 추가 필드를 넣는 경우(형식/키는 변경 이력/호환성 규칙 하에서 확정)
- null/비어 있는 metadata가 필요한 경우 `metadata: {}` 또는 아예 생략(타입/버전 정책 준수)

## 정합성 맵

- `schema-contract.yaml`: `metadata` 설명에 파생 금지 기재
- `workflow-domain.schema.json`: `metadata` 정의에 파생 금지 설명 기재
- `workflow-contract.types.ts`: 타입은 그대로 수신값 타입으로 유지(파생 정책은 렌더 규칙이 담당)

## 구현 지침

- week02/03 이벤트 연동은 `nodeId`, `stateKey`, `workflowId`, `run payload`를 사용하고, `metadata` 자체 해석은 최소화한다.
