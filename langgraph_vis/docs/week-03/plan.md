# Week 3 Plan

## 목표
- week02 상태머신/이벤트 흐름의 정합성을 유지하면서 canonical event/history 모델을 정립한다.
- token/중간 산출물을 history 재조회와 동일한 근거 기반으로 보이게 한다.

## 구성요소 분해 (작업 순서별)
1. canonical-event 타입 정의(기본 필드와 enum, `canonicalMeta` 최소 규칙).
2. stream payload → canonical 이벤트 변환 규칙 1:1 매핑.
3. history 저장 모델(v1): `events`, `nodes`, `finalState`, `failureContext`, `cursor`.
4. history-store 인덱스 전략(`lastSeq`, `nodeAgg`) 및 중복/누락 처리 규칙.
5. history API v1 조회 파이프라인(필터/range/cursor).
6. token-pipeline 병합 규칙(중복 제거, 순서 정렬).
7. diagnostics-view 집계 포맷 및 실패 컨텍스트 연계.
8. history-sample 생성으로 stream-history parity 검증.

## Iteration 기반 검증(다중 페르소나)
- Iteration-01: 데이터 모델 관점으로 canonical event 스키마와 `canonicalMeta` 규칙 정합성 점검.
- Iteration-02: 저장소 엔지니어 관점으로 history-store 인덱스(`lastSeq`, `nodeAgg`) 성장/정합성 위험 검토.
- Iteration-03: 스트림/컨버터 관점으로 변환 규칙 누락·역순 입력·중복 입력 처리 검증.
- Iteration-04: 운영/운영분석 관점으로 failureContext와 진단 뷰 매핑 누락 포인트 점검.

## 의존성
- 선행 필수: week02 run 상태머신(`eventSeq` 단조 증가), SSE 재동기화(`lastEventId`), run 상태 조회 API.
- 내부 의존성: canonical 이벤트 스키마 → history store → history API → frontend 재표시.
- 후행: week04 실패 진단/성능 개선은 history의 `timeline`과 `failureContext`를 소비.

## 일정(의존성 낮은 순)
1. canonical 이벤트 기본 스키마 및 타입 확장 규칙 고정.
2. canonical 변환기 규칙 문서화(스트림 event → canonical event).
3. history 모델(v1) 확정.
4. 저장 레이어(write path)와 정렬 규칙 확정(`eventSeq` 기반).
5. history API 조회 경로(fitler/range/cursor) 구현.
6. token/payload 병합 규칙과 중복 정리.
7. failureContext/diagnostics 연결.
8. history 샘플 기반 parity 시나리오 작성 및 검증.

## 아키텍처 정합성
- canonical 이벤트는 week02 envelope의 `payload`를 그대로 감싸는 구조로 설계해 두 번 해석되는 변환을 제거한다.
- history 조회는 runId/threadId 단위의 단일 진실원천을 유지하며, stream와 조회 결과가 동일한 정렬 규칙을 사용.
- week02 `eventSeq`를 primary key로 사용해 감사 가능성과 재현성을 확보한다.

## 구현 가능성
- week02에서 만든 전이/전송 체인을 그대로 재사용하므로 신규 구현 부담은 저장모델/조회 API에 집중된다.
- 대량 token 처리 대비로 노드별 buffer/window 계산을 분리해 성능 리스크 완화.

## 완료 기준
- canonical event와 SSE event 매핑이 역변환 없이 일치하며, `canonicalMeta`가 `{ isTerminal, source, replayable, schemaVersion }` 기준으로 채워진다.
- stream 이벤트 1건당 history 1건이 대응되며 누락/중복 없음이 테스트로 검증된다.
- history API 결과가 run 완료 직후 즉시 재조회해도 stream 최종 결과와 동일하다.

## 산출물(PoC 증빙)
- `docs/week-03/canonical-event.yaml` (공통 필드 + canonicalMeta 스키마)
- `docs/week-03/history-store.md` (인덱스 전략: `lastSeq`, `nodeAgg`, retention)
- `docs/week-03/parity-test-plan.md` (stream ↔ history 일치 기준)
- `docs/week-03/history-sample.json` (run 1건 기준 canonical event + history 샘플)
