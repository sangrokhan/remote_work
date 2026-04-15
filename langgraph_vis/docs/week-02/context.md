# Week 2 Context

## 왜 이 주차인가
- week01 계약이 고정되면 이제 상태와 이벤트가 같은 타입 집합에서 흘러야 하므로, 실행 흐름을 규격화하지 않으면 week03 이후 분석 데이터가 신뢰되지 않는다.
- 실시간성과 복구성은 동일 주차에서 함께 정리해야 한다. 즉, 이벤트를 보낼 때만 안정적인 게 아니라, 끊김/재접속 복구 전략까지 선결되어야 한다.

## run 상태머신 + SSE envelope + 재동기화 규칙
- 상태머신: `queued → running → completed / failed / cancelled`를 기본 축으로 두고, 필요시 `awaiting_input`를 중간 상태로 둔다.
  - `event_recovered`는 상태값이 아닌 보조 이벤트 메타로만 처리한다.
- SSE envelope: 모든 이벤트를 `{ eventId, runId, threadId, eventSeq, eventType, payload, checkpoint, issuedAt }`로 감싼다.
  - `checksum`은 선택 필드로만 허용한다.
- 재동기화 규칙:
  - 클라이언트는 마지막 처리한 `eventSeq`를 저장한다.
  - 재연결 시 `fromSeq` 또는 `lastEventId`를 전달해 서버가 누락 구간을 재전송한다.
  - 중복 구간은 `eventSeq`로 식별해 idempotent 처리한다.

## PRD와의 정합성
- 7장 Run 실행/스트리밍의 핵심 요구(실시간 갱신, 상태 신뢰성, 일관성)를 직접 구현한다.
- 8.2~8.3 state 조회와 에러 처리 요구를 SSE envelope의 cursor/재시도 규칙과 연결한다.
- 11.2 `RunEvent` 확장 여지를 `payload`의 추가 속성으로 유지해 week03 canonical 이벤트 규격과 충돌을 줄인다.

## 구성요소 분해
- Orchestrator: 생성/취소/최종 종료의 진입점.
- Machine: 상태 전이의 유효성, guard 조건, terminal state 결정.
- Transport: SSE 채널, 재연결 제어, heartbeat/idle 정책.
- Consumer: reducer 및 reconciliation 로직.

## week02 완료 기준(다음 주차 입력값)
- `run-state-machine`, `sse-envelope` 필드는 plan의 동의어 집합과 동일해야 하며 `docs/week-03`의 canonical event 변환기 입력으로 사용한다.
- `threadId`, `eventSeq`, `checkpoint` 누락 없이 run state cursor를 반환한다.
- API 계약 정합성은 `docs/week-02/run-api-contract.yaml`에서 `code/message/requestId` 포함 형태로 고정.

## 구현 가능성 및 리스크
- 리스크: 이벤트 순번 누락, 중복 수신, stream disconnect.
- 대응: `eventSeq` 단조성 검사 + `resync window` + 상태 snapshot API 병행.

## week03 handoff
- week03 진입은 `docs/week-02/context-to-week03.md`를 기준으로 하며, carry-over 리스크는 `docs/week-02/changelog.md`로 추적한다.
