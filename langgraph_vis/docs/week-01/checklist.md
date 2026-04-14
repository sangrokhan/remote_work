# Week 1 Checklist

## 구현 체크리스트
- [ ] `workflowId`/`nodeId` 네임스페이스 정책과 필드 required/선택 규칙 확정.
- [ ] `workflowId`, `version`, `nodes`, `edges`를 포함한 메타 모델 v1 고정.
- [ ] `workflow node` 핵심 필드(`label`, `description`, `stateKey`, `metadata`, `order`) 정합성 점검.
- [ ] `metadata`는 파생 금지/원천 유지 정책으로 고정.
- [ ] 에러 스키마(`code`, `message`, `requestId`)를 모든 실패 응답으로 통일.
- [ ] `GET /api/workflows/{id}/schema` 라우트 계약 고정(필드 필수성, 타입, 버전 규칙).
- [ ] `schema` 생성 방식(정적 manifest + 런타임 검증) 확정 및 구현.
- [ ] `schemaVersion` 하위 호환 원칙 등록(필드 추가 허용, 필수 제거 금지).
- [ ] `docs/week-01/schema-contract.yaml` 산출(메타/응답/오류 DTO 고정).
- [ ] `docs/week-01/schema-contract.fixture.json` 산출(최소 1개 workflow 샘플).
- [ ] `docs/week-01/changelog.md` 산출(버전/호환성 의사결정 기록).
- [ ] `docs/week-01/context-to-week02.md` 산출(week2 입력값 정의 반영).
- [ ] `5xx` 에러 응답(`code`,`message`,`requestId`) 계약 최소 1건 검증.
- [ ] 구성요소 분해 완료: workflow-domain, workflow-contract, workflow-registry, api, error-layer.
- [ ] week02 handoff 산출물(`workflowId`,`nodeId`,`stateKey`,`schemaVersion`) 생성.

## 테스트 커버리지
- [ ] 단위 테스트: `workflowId` 누락/비정상 형식/미존재 응답을 고정 메시지로 검증.
- [ ] 단위 테스트: schema 조회 응답의 필드 타입/필수성 100% 검증.
- [ ] 통합 테스트: 최소 1개 workflow fixture 기준 응답 구조 일치성 검증.
- [ ] 회귀 테스트: `nodeId/stateKey` 매핑이 week02 handoff 입력값과 충돌하지 않는지 점검.
- [ ] 회귀 테스트: 식별자 네임스페이스 규칙과 `metadata` 파생 금지 정책이 plan/context 요구와 충돌하지 않음을 1건으로 검증.
- [ ] 문서 산출물 동기화 체크: `schema-contract`, `changelog`, `context-to-week02`가 plan/context의 입력값과 일치.
- [ ] 계약 테스트: PRD 항목 추적표와 문서 항목(범위/비범위/후속연결) 동기화 확인.
