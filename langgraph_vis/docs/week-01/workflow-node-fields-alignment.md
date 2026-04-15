# Week 01: Workflow Node 핵심 필드 정합성

## node 핵심 필드 (필수)

- `id`: workflow 내부 고유 nodeId
- `label`: UI 노출 라벨
- `description`: 노드 기능 설명
- `stateKey`: state/event 매핑 키
- `order`: UI 정렬/표시 순서

## node 선택 필드

- `metadata`: node 실행/렌더 동작 보조 정보
- 파생 금지: backend source-of-truth 원천 값만 전달

## 정합성 확인(문서 간)

- `schema-contract.yaml`
  - `WorkflowNode.required`: `id`, `label`, `description`, `stateKey`, `order`
  - `metadata`: 정의됨, 선택
- `workflow-domain.schema.json`
  - `WorkflowNode.required`: 동일
  - `metadata`: 정의됨, 선택
- `workflow-contract.types.ts`
  - `WorkflowNode` 인터페이스: `id`, `label`, `description`, `stateKey`, `order` 모두 타입 필수
  - `metadata`: 선택적 필드

## 주석

- 이번 v1 기준으로 `metadata`는 필수화하지 않음.
- week02 전달 값은 `nodeId`, `stateKey` 중심으로 유지.
