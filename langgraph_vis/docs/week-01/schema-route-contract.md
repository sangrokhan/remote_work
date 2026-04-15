# Week 01: GET /api/workflows/{id}/schema 계약 요약

## 계약 개요

- Method: `GET`
- Path: `/api/workflows/{id}/schema`
- 목적: workflowId 기준 정적/정규화된 schema 조회

## 경로 파라미터

- `id`
  - 타입: `string`
  - 패턴: `^[a-z][a-z0-9_-]{1,63}$`
  - 비고: `workflowId` 패턴과 동일

## 성공 응답

- `200`: `schema-contract.yaml`의 `WorkflowSchemaResponse` 사용
  - 필수: `schemaVersion`, `workflowId`, `version`, `nodes`, `edges`
  - 선택: `workflowName`

## 실패 응답

- `400`: `ErrorResponse` (`code`: `INVALID_WORKFLOW_ID`, `INVALID_WORKFLOW_PAYLOAD`)
- `404`: `ErrorResponse` (`code`: `WORKFLOW_NOT_FOUND`)
- `500`: `ErrorResponse` (`code`: `WORKFLOW_REGISTRY_ERROR`, `INTERNAL_ERROR`)

## 형식/검증 규칙

- `nodes`/`edges`는 `schemaVersion` 정책(필수 필드 보존) 하에 파싱
- `edge.from`, `edge.to`는 존재하는 node id와 매핑 가능해야 함(레지스트리 단계 검증)
