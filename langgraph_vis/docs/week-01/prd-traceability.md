# PRD 항목 추적표 (Week 1)

## PRD-3.1 포함 범위
- 요구: `workflow 구조 조회 방식` / `workflow schema 조회`
- 반영 문서:
  - `prd.md` 3.1
  - `docs/week-01/plan.md` (범위)
  - `docs/week-01/context.md` (PRD 정합성)
  - `docs/week-01/schema-route-contract.md`
  - `docs/week-01/schema-contract.yaml`

## PRD-4.1 기능 요구사항
- 요구: `workflow schema 조회`
- 반영 문서:
  - `prd.md` 4.1
  - `docs/week-01/context-to-week02.md` (week02 입력값: `workflowId`, `nodeId`, `stateKey`, `schemaVersion`)
  - `docs/week-01/workflow-node-fields-alignment.md`
  - `docs/week-01/schema-contract.fixture.json`

## 비기능/설계 정합성
- 요구: `backend source-of-truth`, `frontend 하드코딩 최소화`
- 반영 문서:
  - `docs/week-01/metadata-source-of-truth-policy.md`
  - `docs/week-01/context.md`
  - `docs/week-01/workflow-domain.schema.json`

## 범위/비범위
- scope:
  - `docs/week-01/plan.md`의 범위 섹션 (`workflow schema 조회`, 라우트 계약, 에러 규약)
  - `docs/week-01/schema-contract.yaml`
  - `docs/week-01/context-to-week02.md`
- out-of-scope:
  - 런 실행, stream, 히스토리/리트라이, UI 렌더 정책 최적화
  - `docs/week-01/plan.md`의 비범위 섹션
- handoff 기준: plan/context에서 week02로의 `handoff` 문맥에서 nodeId/stateKey/schemaVersion를 그대로 전달.

## 후속연결
- week02 run/event payload에서 `run payload`와 `node/state` 식별자를 동일하게 사용
- 후속작업: `docs/week-01/context-to-week02.md`, `docs/week-01/checklist.md`
