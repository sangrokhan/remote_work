# Week 2 Plan

## 목표
- run 생명주기와 이벤트 발행의 단일 계약을 확정하고, SSE를 통해 frontend 상태 동기화를 완성한다.
- week01 계약을 소비하는 run 상태 머신을 정의해 week03 이후 모델 확장으로 자연스럽게 이어지게 한다.

## 구성요소 분해 (작업 순서별)
1. sse-envelope 필드 공통 규격: `eventId`, `runId`, `threadId`, `eventSeq`, `eventType`, `payload`, `checkpoint`, `issuedAt` 최소 집합.
2. run-state-machine: `queued`, `running`, `awaiting_input`, `completed`, `failed`, `cancelled` 상태 및 전환 규칙 표.
3. run-orchestrator: run 생성, 상태 초기화, 타임라인 전이 트리거.
4. state-query API: state 조회 + event cursor 노출.
5. sse-envelope-emitter: state 머신 이벤트를 envelope로 직렬화하고 eventSeq/checkpoint를 관리.
6. resync-controller: 중단/재접속 시 `lastEventId`/`fromSeq` 기반 재동기화 정책.
7. client-sync reducer(JavaScript): 이벤트 정규화 및 DOM 반영.

## Iteration 기반 검증(다중 페르소나)
- Iteration-01: 상태머신 엔지니어 관점으로 transition table 및 invalid transition 정책 정합성 점검.
- Iteration-02: 메시지 설계 관점으로 SSE envelope 필수/선택 필드의 하위호환성 점검.
- Iteration-03: 운영/회복력 관점으로 `fromSeq`/`lastEventId` 재동기화 전략과 heartbeat/idle 정책 검증.
- Iteration-04: 클라이언트 QA 관점으로 reducer 정렬·중복제거 정책의 실패 시나리오 점검.

## 의존성
- 선행: week01 `workflowId/node/stateKey` 계약 고정.
- 내부: run 생성 → 상태 머신 → envelope 발행 → state 조회 → 프런트 동기화.
- 후행: week03 token/history 모델은 상태머신의 `eventType`/`eventSeq`를 전제 조건으로 가정.

## 일정(의존성 낮은 순)
1. run 상태머신과 transition table 먼저 고정.
2. SSE envelope 스키마 확정(필수/선택 필드 분리).
3. run 생성 규칙(초기 `eventSeq`, run/thread ID, checkpoint 규칙) 정합.
4. state 조회 API에 cursor 노출.
5. sse emitter가 상태머신 전이에 1:1 대응되도록 바인딩.
6. 재동기화 규칙 구현(구간 재요청, 중복 제거, 누락 탐지).
7. frontend reducer 연동 및 스트림 실패 fallback 동기화 확인.

## 아키텍처 정합성
- 상태머신은 week01의 노드 식별자/타입 모델을 그대로 사용해 contract drift를 차단한다.
- SSE envelope는 이벤트 정렬/중복제거/부분복구를 위해 단조 증가 `eventSeq`를 의무화한다.
- 재동기화 규칙에서 `lastEventId`/`fromSeq`는 state 캐시 무결성의 단일 기준점이다.
- `event_recovered`는 상태머신 상태가 아닌 보조 이벤트 표식이며 `run-state-machine` 상태 집합은 위 6개 항목만 사용한다.

## 구현 가능성
- state-machine + emitter + reducer는 기존 라우트 계층과 분리되므로 단계적 배포가 가능하다.
- 재동기화 실패 지점(연속 이벤트 손실, 타임아웃)을 분리해 fallback 경로를 명시하면 운영 위험을 낮출 수 있다.
- 외부 의존도를 낮춘 단일 노드 내 이벤트 큐/체크포인트 전략으로 구현 난이도 제어 가능.

## 완료 기준
- run 시작/진행/완료/실패 전이가 `run-state-machine` transition table을 벗어나지 않는다.
- SSE envelope가 모든 이벤트에 대해 `eventSeq` 연속성과 `checkpoint` 규칙을 보장하고, heartbeat(예: 20초)·idle timeout(예: 90초) 정책이 문서화된다.
- disconnect 후 재접속 시 `lastEventId` 기반으로 누락 이벤트 없이 동기화된다.
- `event_recovered` 메타(선택)는 상태머신 종료 판정을 변경하지 않는다.

## 산출물(PoC 증빙)
- `docs/week-02/run-state-machine.md` (상태 전이 표 + terminal 규칙)
- `docs/week-02/sse-envelope.yaml` (event 필수/선택 필드)
- `docs/week-02/reconnect-guide.md` (heartbeat/idle/retry/resync 전략)
