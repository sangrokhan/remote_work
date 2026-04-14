# Week 3 Context

## 왜 이 주차인가
- week02에서 실시간 실행이 가능해졌더라도, 실행 근거를 증적할 수 없으면 실패 분석/재현이 불가능하다.
- 따라서 event와 history를 canonical 모델로 통합해 스트리밍 가시성과 조회 일치성을 동시에 확보해야 다음 주의 진단/성능 단계로 넘어갈 수 있다.

## week02 선결조건
- run 상태머신은 `eventSeq` 단조 증가와 terminal state 보장을 충족해야 함.
- SSE envelope는 `runId`, `threadId`, `eventSeq`, `eventType`, `issuedAt`, `checkpoint`를 항상 포함해야 함.
- state 조회 API는 cursor(최신 seq) 및 마지막 영속화 시점을 제공해야 함.

## canonical event 모델
- 공통 필드: `eventId`, `runId`, `threadId`, `eventSeq`, `eventType`, `nodeId`, `issuedAt`.
- 공통 메타: `isTerminal`, `source`, `replayable`, `schemaVersion`.
- 토큰 이벤트, 노드 시작/완료, 실패/중단, 사용자 메시지 이벤트를 동일한 규약으로 수집해 `history.events`에 축적.

## PRD와의 정합성
- 7.3 custom progress 이벤트, 8.4 에러 처리, 11.2 RunEvent 확장성 요구를 반영한다.
- 14.3 단계 요구(분석성, 재현성)와 week02의 스트림 완결성을 연동.

## 구성요소 분해 및 검증
- 변환기: raw stream → canonical event.
- 저장기: canonical event append + 정렬/중복제거.
- 조회기: filter/pagination + node별 timeline 집계.
- 검증기: stream-history parity 테스트 및 checksum 비교.

## 주차 출력물 연결
- week03 종료 시 `docs/week-04`가 소비할 입력:
  - `history.events[].canonicalMeta.schemaVersion`
  - `history.failureContext`
  - `history.store cursor`
- PoC 증빙: `docs/week-03/history-sample.json`(single run 재생성 예시) 를 병행 제공.

## 구현 가능성
- 타입 시스템으로 이벤트 스키마를 고정한 뒤 코드 공유를 적용하면 backend/frontend 간 변환 비용이 줄어든다.
- 기존 week02 테스트 자산을 확장해 재사용할 수 있어 일정 리스크가 낮다.
