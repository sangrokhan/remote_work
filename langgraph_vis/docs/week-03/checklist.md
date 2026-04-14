# Week 3 Checklist

## 구현 체크리스트
- [ ] `docs/week-03/canonical-event.yaml` 작성 및 이벤트 공통 스키마 확정.
- [ ] canonical event 타입 정의(`run_started`, `node_started`, `node_token`, `node_completed`, `run_completed`, `run_failed`).
- [ ] `stream event -> canonical event` 변환 규칙 문서화 및 구현.
- [ ] `docs/week-03/history-store.md` 작성 및 history 모델 v1 확정(`runId`, `events`, `nodes`, `finalState`, `failureContext`, `cursor`).
- [ ] history store 인덱스(`lastSeq`, `nodeAgg`) 구현 여부 및 비정상 성장 케이스 점검.
- [ ] history API pagination/cursor, node-level 요약 필터 추가.
- [ ] token 이벤트 병합 정책(청크 정렬, 중복 제거) 적용.
- [ ] 실패 직전/실패 이벤트와 history failureContext 매핑.
- [ ] `docs/week-03/parity-test-plan.md` 작성 및 stream-history 검증 포인트 정의.
- [ ] `docs/week-03/history-sample.json` 최소 1건 작성(예시: run 1회 실행, 실패 케이스 포함).
- [ ] 구성요소 분해 완료: canonical 모델, 저장소, API, token pipeline, diagnostics 뷰.

## 테스트 커버리지
- [ ] 단위 테스트: canonical event 스키마 유효성(필수 필드, 타입, enum).
- [ ] 단위 테스트: `canonicalMeta`(`isTerminal`, `source`, `replayable`, `schemaVersion`) 필드 채움 규칙 검증.
- [ ] 단위 테스트: stream payload 누락/중복/역순 입력 시 history 정합성 처리.
- [ ] 통합 테스트: `stream event -> canonical event` 1:1 매핑 및 eventSeq 유지.
- [ ] 통합 테스트: run 완료 직후 history와 stream replay 결과 비교(일치율 100%).
- [ ] 통합 테스트: token burst(20건)에서 순서/누락 없이 병합되는지 확인.
- [ ] 통합 테스트: 실패 이벤트와 history의 `failureContext` 1:1 매핑.
- [ ] 통합 테스트: history API cursor + node-level 요약 필터가 canonical 모델과 일치.
- [ ] 회귀 테스트: week02 상태머신 전환 표(`eventSeq` 연속성)가 깨지지 않는지 감시.
- [ ] 복구 테스트: history write 실패 시 재시도/재생성 경로와 최소 보존 동작 점검.
- [ ] 문서 연동 테스트: `canonical-event.yaml`, `history-store.md`, `parity-test-plan.md`의 테스트 조건이 구현 포인트와 1:1 대응.
