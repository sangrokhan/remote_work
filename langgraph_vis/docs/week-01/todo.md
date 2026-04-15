# Week 1 → Week 2 인계 TODO (PoC 관점)

## 결론
- 현재 상태는 **Go To Week 2** 가능.
- 치명적 장애를 유발할 정도의 결함은 없음. 다만 Week 2 확장 시 영향을 줄이기 위한 보완이 필요.

## 완료 상태 (요약)
- Week 1 핵심 기능:
  - GET `/api/workflows/{workflowId}/schema` 응답 계약 구현
  - 에러 응답 형식 통일
  - manifest 기반 스키마 검증 및 중복/제약 조건 검사 강화
  - 테스트 분리 및 정리 (`tests/schema/*`)
- 검증:
  - `tests/schema/contract.spec.mjs`: 7/7
  - `tests/schema/schema-api.spec.mjs`: 6/6
  - `tests/schema/registry.spec.mjs`: 5/5

## Week 2 전에 꼭 반영 (우선순위 높음)
1. `src/schema-registry.mjs`
   - `getById()` 반환 시 캐시 객체가 외부에서 변경되지 않도록 방어적 복제 수행(불변성 보장).
2. `src/schema-registry.mjs`
   - `manifestPath` 안전성 검사(예상 위치/확장자/존재 여부) 최소 가드.
3. `tests/schema/schema-api.spec.mjs` 또는 신규 테스트
   - workflowId 경계/문자셋(길이, 대문자/특수문자/포맷) 검증 케이스 추가.
4. `tests/schema/schema-api.spec.mjs`
   - `WORKFLOW_REGISTRY_ERROR` 5xx 라우팅 검증 케이스 추가.

## 다음 주차에서 다듬으면 좋은 항목 (우선순위 낮음)
- 라우팅/에러 처리 분리(현재 단일 수기 라우트) 공통 미들웨어로 확장용 개선.
- 저장소 레이어 추상화(Manifest 로더/저장소 교체 고려) 설계 선행.
- 테스트 계약을 문자열 고정보다 스키마/구조 기반 단언으로 일부 완화.

## 참고
- 문서:
  - [docs/week-01/checklist.md](/home/han/.openclaw/workspace/remote_work/langgraph_vis/docs/week-01/checklist.md)
  - [docs/week-01/issue-resolution-plan.md](/home/han/.openclaw/workspace/remote_work/langgraph_vis/docs/week-01/issue-resolution-plan.md)
  - [docs/week-01/prd-traceability.md](/home/han/.openclaw/workspace/remote_work/langgraph_vis/docs/week-01/prd-traceability.md)

