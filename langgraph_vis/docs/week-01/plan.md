# Week 1 Plan

## 범위 / 비범위
- 범위: workflow 식별자 조회, workflow/노드/엣지 메타 모델 고정, `/api/workflows/{id}/schema` 반환 규격 확정, 실패/미존재 조건의 오류 응답 규칙 확정.
- 비범위: 런 실행, 스트림, 히스토리 저장, UI 렌더 정책 최적화.
- 후속연결: week02에서 run lifecycle/이벤트 payload를 설계할 때 schema 계약과 node/state 필드명을 그대로 참조.

## 목표
- backend를 Source of Truth로 하는 workflow schema 계약을 확정한다.
- 최소 기능으로도 안정적으로 스키마를 조회할 수 있고, 후속 주차의 구조 분해 및 타입 확장 기반이 깔린다.

## 구성요소 분해 (작업 순서별)
1. `workflow-domain`: workflowId, node, edge, metadata의 정규화된 도메인 핵심 필드부터 확정.
2. `workflow-contract`: 스키마 API 응답 타입(요청/응답 DTO, validation 스키마) 고정.
3. `workflow-contract`: 에러 응답 스키마(`code`, `message`, `requestId`) 통일.
4. `workflow-registry`: `workflowId`/`nodeId` 네임스페이스 정책으로 in-memory + manifest 적재 규칙 수립.
5. `docs-sync`: PRD 항목 대비 계약 버전, fixture, changelog 산출물 템플릿.
6. `workflow-api`: `GET /api/workflows/{id}/schema` 라우트 매핑과 오류 처리 바인딩.

## Iteration 기반 검증(다중 페르소나)
- Iteration-01: 아키텍트 관점으로 도메인 모델(`workflowId`, `nodeId`, version, 필수/선택 필드) 정합성 검토.
- Iteration-02: SRE/운영 관점으로 에러 스키마(404/400/5xx) 일관성 검토 및 예외 메시지 정책 점검.
- Iteration-03: QA 관점으로 fixture/문서 스펙(노드·엣지 스키마, 계약 규칙) 누락 여부 체크.
- Iteration-04: API 관점으로 `GET /api/workflows/{id}/schema` 실패 케이스와 라우트 바인딩 리스크 점검.

## 의존성
- 선행 의존성: PRD 3.1~4.1의 schema 조회 요구, 공통 에러 규약.
- week 내부 의존성: domain 모델 확정 -> API 계약 고정 -> 조회 처리 -> 에러 계약·문서 반영.
- 후속 의존성: week02 run 시작 payload(노드 메타, stateKey), week03 history/토큰 시각화에 필요한 node 식별자 정합성.

## 일정(의존성 낮은 순)
1. 식별자와 기본 규칙 확정: `workflowId`/`nodeId` 네임스페이스 정책, 필드 required/optional, version 규칙.
2. 도메인 모델과 공개 스키마 v1 정적 정의(타입/필수 필드, 메타데이터 파생 금지 원칙).
3. 공통 에러 계약 고정: `code`, `message`, `requestId` 규칙과 404/400/5xx 매핑.
4. 워크플로우 조회 방식 확정: 정적 manifest + 런타임 검증 체크리스트.
5. `GET /api/workflows/{id}/schema` 라우트 입력/출력 바인딩.
6. 계약 기반 테스트와 문서 산출물(changelog, fixture, handoff) 정리.

## 아키텍처 정합성
- 이벤트/스트림 확장에 대비해 필수키를 고정하고, 노드 식별자 충돌 방지 정책을 문서화한다.
- metadata는 파생 금지를 유지하고, 렌더러/클라이언트는 오직 source-of-truth에서 내려온 값만 소비한다.
- 에러 응답은 동일 구조로 통일해 주차 간 디버깅 비용을 낮춘다.

## 구현 가능성
- 첫 주차는 단일 endpoint + 정적 fixture 중심이라 위험도가 낮고, 동시 수정 포인트를 최소화해 안정적으로 선행작업 완료 가능.
- 주차 완료 후 week02에 필요한 최소 인터페이스를 안정적으로 고정할 수 있다.

## 완료 기준
- `GET /api/workflows/{id}/schema`가 1주차 계약(`schemaVersion`, `nodes[].stateKey`, `nodes[].metadata`, `edges[]`)을 항상 반환한다.
- `workflowId` 미존재/파싱 실패에 대해 동일 에러 구조(`code`, `message`, `requestId`)로 응답한다.
- 스키마 fixture 100% 계약 일치율을 단위 테스트로 보장한다.

## 산출물(계약 증빙)
- `docs/week-01/schema-contract.yaml` (Schema v1 응답/오류 DTO)
- `docs/week-01/schema-contract.fixture.json` (최소 1개 workflow 샘플)
- `docs/week-01/changelog.md` (schema version 변경 내역, 호환성 원칙)
- `docs/week-01/context-to-week02.md` (week02 입력물: workflowId, nodeId, stateKey, schemaVersion, error schema)
