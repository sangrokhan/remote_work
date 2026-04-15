# Week 01: Workflow Meta Model v1 (정적 스키마)

## 대상 항목

- `workflowId`: workflow 식별자
- `version`: workflow 정의 버전
- `nodes`: 노드 목록
- `edges`: 엣지 목록

## v1 고정 규칙

- 위 4개 항목은 `GET /api/workflows/{id}/schema` 응답에서 **필수**.
- `schemaVersion`은 별도 호환성 키로 항상 함께 존재.
- `version`은 MAJOR.MINOR.PATCH를 기본 형식으로 권장.
- `nodes[]`는 최소 1개 이상.
- `edges[]`는 0개 이상 허용.

## 필드 정합성 매핑

| 계층 | 필드 | 상태 |
|---|---|---|
| `schema-contract.yaml` | `workflowId`, `version`, `nodes`, `edges` | required |
| `workflow-domain.schema.json` | `workflowId`, `version`, `nodes`, `edges` | required |
| `workflow-contract.types.ts` | `workflowId`, `version`, `nodes`, `edges` | required |
