# Week 02 → Week 03 Hand-off

## 확정값 (Carry-over)

- `run` 상태머신은 `queued`, `running`, `awaiting_input`, `completed`, `failed`, `cancelled`로 정합.
- `eventSeq`는 단조 증가 정합성으로 사용되며, run 시작 이벤트는 `eventSeq: 1`으로 시작.
- SSE 재동기화 커서 정책:
  - 쿼리 우선순위: `fromSeq` 우선, 미지정 시 `lastEventId`.
  - `lastEventId`는 이벤트 `eventId` 기준으로 eventSeq를 찾아 `eventSeq+` 으로 재전송.
- run API 에러 계약:
  - 400: `INVALID_RUN_ID`, `INVALID_RECONNECT_QUERY`
  - 404: `RUN_NOT_FOUND`
  - 405: `INVALID_RUN_PAYLOAD`
  - 409: `INVALID_RUN_TRANSITION`
  - 500: `INTERNAL_ERROR`
- `/api/runs/{runId}/state` 응답에는 `cursor`(최신 `eventSeq`, `state`)가 포함되어야 함.

## week03 시작 시점의 보장 가정

1. `src/run-state-store.mjs` 상태와 `src/run-state-api-server.mjs` 조회 계약이 동일한 커서 의미(`eventId`/`eventSeq`)를 사용.
2. `event_recovered`는 상태 변경을 일으키지 않는 메타 이벤트로 처리.
3. 리플레이/재요청은 `fromSeq`와 `lastEventId` 기반으로 동일 처리.

## 미해결/승인 조건 (week03에서 우선 처리)

- [ ] `Last-Event-ID` 헤더 기반 재연결 지원/비지원 여부를 클라이언트 정책으로 명확화.
- [ ] `unknown lastEventId`를 `fromSeq=0`으로 전체 재전송할지, 별도 실패 코드로 강제 재동기화할지 정책 확정.
- [ ] 재연결 가이드의 heartbeat/idle 정책을 실제 프로덕션 프록시와 맞는지 운영 검증.
- [ ] 재시도 스킵 정책(중복 eventId가 동일 payload인지 여부)에 대한 week03 기준 확정.

## week03로 전달되는 주요 산출물

- [week-02 체크리스트 증빙](/home/han/.openclaw/workspace/remote_work/langgraph_vis/docs/week-02/checklist.md)
- [run API 계약](/home/han/.openclaw/workspace/remote_work/langgraph_vis/docs/week-02/run-api-contract.yaml)
- [reconnect 가이드](/home/han/.openclaw/workspace/remote_work/langgraph_vis/docs/week-02/reconnect-guide.md)
- [state-machine](/home/han/.openclaw/workspace/remote_work/langgraph_vis/docs/week-02/run-state-machine.md)
- [sse envelope](/home/han/.openclaw/workspace/remote_work/langgraph_vis/docs/week-02/sse-envelope.yaml)
