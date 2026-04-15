# Week 01: Schema 생성 방식 정책 (manifest + 런타임 검증)

## 1) 1차 원칙

- `workflow` 스키마는 `manifest` 기반으로 조회한다.
- 조회 시점에 스키마를 동적으로 계산(파생)하지 않고, 저장된 정규 manifest를 반환한다.

## 2) 정적 manifest 규칙

- workflow별 고정 객체: `workflowId`, `version`, `nodes`, `edges`
- manifest 항목은 `schemaVersion` 정책하에 버전 관리
- 필수 필드 누락 시 조회 불가로 처리

## 3) 런타임 검증 규칙

스키마 반환 직전, manifest에서 다음을 검증한다.

- `workflowId` 패턴/길이/형식 검사
- `version` 형식(`MAJOR.MINOR.PATCH`) 검사
- `nodes` 최소 1개
- 각 `node.id`의 형식/중복 검사
- 각 `edge.from`, `edge.to`가 존재하는 node id인지 검사
- 필수 필드 존재성(`schemaVersion`, `workflowId`, `version`, `nodes`, `edges`)
- `metadata`가 존재 시 object 타입인지 검사

## 4) 검증 실패 정책

- `workflowId`/형식/필수성 오류: `INVALID_WORKFLOW_ID`, `INVALID_WORKFLOW_PAYLOAD`
- workflow 미존재: `WORKFLOW_NOT_FOUND`
- 내부 manifest 로딩/파싱 실패: `WORKFLOW_REGISTRY_ERROR`

## 5) 출력 정책

- 검증 성공 시 `schema-contract.yaml`의 `WorkflowSchemaResponse`로 렌더링
- 실패 시 `error-response-contract.md` 스펙의 실패 응답으로 반환
