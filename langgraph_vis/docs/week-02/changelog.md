# Week 02 Changelog

## v1.0.0 (2026-04-14)

- run 상태머신/이벤트 체인 구현을 정리했고 `run_started`, `node_started`, `run_completed` 흐름을 `eventSeq` 기반으로 고정.
- SSE event envelope와 `run` API cursor 조회/리플레이 규약을 `run-state-store`, `run-state-api-server`, `resync-controller`에 반영.
- run 에러 정규화 계약(`RUN_NOT_FOUND`, `INVALID_RUN_TRANSITION`, `INTERNAL_ERROR`)을 `run-error-contract`와 `run-state-server` 경로에 적용.
- week02 계약 문서 `run-api-contract.yaml`을 추가해 `/state`, `/events`, `/events/stream` 응답 규격을 통일.
- 테스트 실행 진입점(`package.json`, `tests/README`)을 고정해 week03 인수 기준을 명시.

## 기존 이슈 / carry-over

- 재연결은 현재 쿼리 파라미터 기반이고, `Last-Event-ID` 헤더 전제는 미결정 상태.
- heartbeat/idle 정책(20s/90s)은 구현·테스트 문서 간 정합성 검토가 필요.
- unknown `lastEventId` 처리(현재 0 기반 전체 리플레이)는 정책적으로 엄격화 필요.
- 프로세스 재기동 시 이벤트 지속성은 메모리 기반 한계가 남아 있어 week03에서 영구 저장 경로 연동이 선행되어야 함.
