# Week 3 History Store v1

## 기본 모델

- `runId`: 단일 실행 식별자
- `threadId`: 실행 스레드 식별자
- `events`: canonical event 배열
- `nodes`: 노드별 집계 맵 (`nodeId` → `{ nodeId, state, startSeq, endSeq, lastSeq, tokenCount }`)
- `finalState`: 최신 run 상태 스냅샷 (`state`, `eventSeq`, `lastEventId`)
- `failureContext`: 마지막 실패 이벤트의 근거 (`run_failed`) 또는 `null`
- `cursor`: 조회/재전송에 쓰는 커서 (`state`, `eventSeq`, `lastEventId`, `schemaVersion`)

## 인덱스 전략

- `lastSeq`: 마지막 저장 이벤트 시퀀스
- `eventId` 중복맵: 동일 `eventId` 재수신 시 idempotent 처리
- `nodeAgg`: 노드별 집계 상태를 유지해 매 조회마다 전체 이벤트 스캔 없이 집계 제공

## write path

- 입력은 `run_store` 기준 `raw events`를 canonical로 1회 변환한다.
- canonical 변환 규칙:
  - `run_*` / `node_*` 이벤트 타입을 canonical 타입으로 정규화
  - `run_failed`는 `canonicalMeta.isTerminal = true`
  - `event_recovered`는 `canonicalMeta.replayable = false`
- 변환 후 이벤트 정렬은 `(eventSeq, eventId)` 기준으로 보장한다.

## query path

- 기본 조회는 run 단위 전체 history를 snapshot한 뒤 쿼리 파라미터를 적용한다.
- `fromSeq`, `lastEventId`는 배타적으로 사용하며, 전달 값이 없으면 전체를 조회한다.
- `limit`은 페이지 크기(최대 500, 기본 100)로 사용한다.
- `nodeId`는 node-scope 필터.
  - `events`뿐 아니라 `nodes`도 동일 nodeId 집계로 축소한다.
- 응답은 `pagination.hasMore`와 `pagination.nextCursor`를 포함한다.
  - `nextCursor`는 현재 페이지 마지막 이벤트 기준으로 `{ eventSeq, eventId, state }`를 사용한다.
  - `state`는 마지막 이벤트 `checkpoint.state`를 우선 사용하고, 미제공 시 `finalState.state`로 보정한다.

## 실패 진단

- `failureContext`는 가장 최근 `run_failed` 이벤트를 기준으로 구성한다.
- 필드:
  - `eventId`, `eventSeq`, `eventType`, `issuedAt`
  - `nodeId`
  - `reason`, `error`
  - `failureCode`, `failureCategory`, `retryable`, `rootCause`
  - `resolutionHints`, `evidenceRefs`, `retryInfo`

## retention

- PoC v1은 인메모리 저장이므로 영속화/보존 정책은 week2 run event 생명주기에 의존한다.
- retention key는 `runId` 단위로 run 삭제 시 함께 정리한다.

## 위험 요소

- 이벤트 역순 유입 시 정렬이 재계산되어야 하며, 동일 `eventSeq` 다중 이벤트는 `eventId`로 안정 정렬이 필요하다.
- 같은 `eventId` 재수신은 중복 저장 없이 기존 이벤트 반환.
