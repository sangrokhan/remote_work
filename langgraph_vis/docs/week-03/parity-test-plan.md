# Week 3 Stream ↔ History Parity Test Plan

## 목표

- stream 재생 이벤트와 history 재조회 이벤트가 `eventId`, `eventSeq`, `eventType`, `canonicalMeta` 기준으로 일치하는지 검증한다.
- 노드별 요약(`nodes`)과 실패 정보(`failureContext`)가 raw 이벤트에서 정확히 파생되는지 검증한다.

## 테스트 그룹

1. canonical 변환
   - run_started/node_started/node_token/node_completed/run_completed/run_failed → canonical로 1:1 변환
   - `canonicalMeta.schemaVersion`은 `1.0.0` 고정
   - `canonicalMeta.source`는 `stream`
2. 스트림 누락/중복 회복
   - 동일 `eventId` 재수신 시 history append는 무시하고 eventSeq 유지
   - `fromSeq`와 `lastEventId` 복구 경로 모두 올바른 기준점 계산
   - `fromSeq`/`lastEventId` 조합은 API 정책에 맞는 에러 분기 검증
3. 순서 정합성
   - 역순 이벤트 입력 후에도 `events` 정렬 기준이 `eventSeq` 상승순 보장
4. 노드 집계
   - `node_started`/`node_token`/`node_completed` 시퀀스로 node 상태/카운트 집계
   - `nodeId` 필터 시 `events`와 `nodes`가 동일 node 집계로 축소
5. 페이지네이션
   - `limit`과 `fromSeq` 조합으로 동일 run 에 대한 연속 페이지 보장
   - 페이지 마지막 이벤트를 기준으로 `nextCursor.eventSeq` 연속성 보장
6. 실패 컨텍스트
   - `run_failed`를 마지막 실패 근거로 `failureContext` 구성
   - `failureCode`, `failureCategory`, `retryable`, `retryInfo` 존재 여부 검증
7. 오류 처리
   - 잘못된 runId / invalid query 시 기존 run API와 동일한 오류 코드 사용
   - `limit`은 `INVALID_RUN_PAYLOAD`, `fromSeq`/`lastEventId`는 `INVALID_RECONNECT_QUERY`로 분기 점검
