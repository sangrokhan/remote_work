# Week 2 Checklist

## 구현 체크리스트
- [x] `docs/week-02/run-state-machine.md` 작성 및 transition table와 상태 집합 정합.
- [x] `docs/week-02/sse-envelope.yaml` 작성 및 envelope 필드(plan/context 기준) 고정.
- [x] run 상태머신 transition table 생성 및 invalid transition 예외 정책 적용.
- [x] run 생성 규칙(`runId`, `threadId`, 초기 `eventSeq`, checkpoint) 문서화 및 구현.
- [x] 상태머신 + run 생성 연계: `run_started`/`running`/`terminal`의 초기 전이 바인딩.
- [x] `GET /api/runs/{runId}/state`에서 cursor(eventSeq/checkpoint) 노출.
- [x] SSE envelope 메시지 생성기 구현.
- [x] `docs/week-02/reconnect-guide.md` 작성 및 heartbeat/idle timeout 정책(예: 20s/90s) 문서화.
- [x] `fromSeq`/`lastEventId` 기반 재동기화 API 플로우 구현.
- [x] `GET /api/runs/{runId}/state`와 stream cursor 간 일관성 점검.
- [x] frontend reducer에서 `eventSeq` 기준 정렬·중복 제거·오프라인 복구 처리.
- [x] `event_recovered`는 상태값이 아닌 메타 이벤트임을 reducer/검증 규칙에 반영.
- [x] 구성요소 분해 완료: run-orchestrator, state-machine, emitter, resync-controller, state-query, reducer.
- [x] run API/오류/재연결 계약 통일 문서 산출(`docs/week-02/run-api-contract.yaml`).

## 테스트 커버리지
- [x] 단위 테스트: 상태머신 전이 규칙(정상/예외 분기) 100%.
- [x] 단위 테스트: envelope 공통 필드 유효성(`eventId`, `runId`, `threadId`, `eventSeq`, `eventType`, `issuedAt`) 검증.
- [x] 통합 테스트: `run-state-machine.md` transition 표와 invalid transition 정책 일치 검증.
- [x] 통합 테스트: run 생성 시 초기 `eventSeq`=1, cursor(`eventSeq`,`checkpoint`) 노출 보장 검증.
- [x] 통합 테스트: run_started → node 실행 이벤트 → run_completed 정상 순서 시나리오.
- [x] 통합 테스트: `GET /api/runs/{runId}/state`와 stream cursor 간 일관성 검증.
- [x] 통합 테스트: disconnect 후 `fromSeq` 재요청 시 중복/누락 없이 동기화.
- [x] 통합 테스트: 잘못된 `runId/threadId` 요청 시 일관된 오류 규약 적용.
- [x] 통합 테스트: terminal 이후 추가 이벤트/중복 이벤트를 무시하는지 검증.
- [x] 통합 테스트: run_started→node_started→run_completed 시나리오에서 재연결 1회 복구 성공(최소 PoC 시연 경로).
- [x] 통합 테스트: `event_recovered` 이벤트가 상태머신 `completed/failed/cancelled`로 혼입되지 않음을 검증.
- [x] 통합 테스트: frontend reducer에서 eventSeq 정렬/중복 제거/재연결 복구가 우선순위 규칙을 지키는지 검증.
- [x] 계약 검증: `sse-envelope.yaml`과 `reconnect-guide.md`의 필수/선택 정책이 구현과 일치.

## 테스트 증빙 맵

| 항목 | 근거 문서 | 증빙 테스트 | 상태 |
| --- | --- | --- | --- |
| 상태머신 전이/invalid transition | `docs/week-02/run-state-machine.md` | [run-state-machine.spec.mjs](/home/han/.openclaw/workspace/remote_work/langgraph_vis/tests/unit/run-state-machine.spec.mjs) | ✅ |
| run 생성 초기 cursor 이벤트 시퀀스 | `docs/week-02/context.md` | [run-state-store.spec.mjs](/home/han/.openclaw/workspace/remote_work/langgraph_vis/tests/unit/run-state-store.spec.mjs) | ✅ |
| state 조회 cursor 노출 | `docs/week-02/run-api-contract.yaml` | [run-state-api-server.spec.mjs](/home/han/.openclaw/workspace/remote_work/langgraph_vis/tests/integration/run-state-api-server.spec.mjs) | ✅ |
| SSE envelope 필드/검증 | `docs/week-02/sse-envelope.yaml` | [sse-envelope.spec.mjs](/home/han/.openclaw/workspace/remote_work/langgraph_vis/tests/unit/sse-envelope.spec.mjs) | ✅ |
| reconnect fromSeq/lastEventId 재요청 | `docs/week-02/reconnect-guide.md` | [resync-controller.spec.mjs](/home/han/.openclaw/workspace/remote_work/langgraph_vis/tests/integration/resync-controller.spec.mjs), [run-state-api-server.spec.mjs](/home/han/.openclaw/workspace/remote_work/langgraph_vis/tests/integration/run-state-api-server.spec.mjs) | ✅ |
| 에러 계약 정규화 | `docs/week-01/error-response-contract.md`, `docs/week-02/run-api-contract.yaml` | [run-error-contract.spec.mjs](/home/han/.openclaw/workspace/remote_work/langgraph_vis/tests/unit/run-error-contract.spec.mjs) | ✅ |
| 통합 동작 시나리오(생성→진행→완료) | `docs/week-02/context.md` | [run-orchestrator.spec.mjs](/home/han/.openclaw/workspace/remote_work/langgraph_vis/tests/integration/run-orchestrator.spec.mjs) | ✅ |
| schema/registry 계약 연결 유지 | `docs/week-01/schema-contract.yaml` | [contract.spec.mjs](/home/han/.openclaw/workspace/remote_work/langgraph_vis/tests/schema/contract.spec.mjs), [registry.spec.mjs](/home/han/.openclaw/workspace/remote_work/langgraph_vis/tests/schema/registry.spec.mjs) | ✅ |

## 주차 종료 인계 조건
- [x] `tests/README.md`에서 단일 실행 진입점(`npm test`)이 문서화됨.
- [x] 다음 주차 하위 인계 문서(`docs/week-02/context-to-week03.md`) 존재.
- [x] week02 변경 이력(`docs/week-02/changelog.md`) 및 미해결 리스크가 누락 없음.
