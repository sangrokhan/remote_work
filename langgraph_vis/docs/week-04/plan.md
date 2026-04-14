# Week 4 Plan

## 범위 / 비범위
- 범위: 렌더링 모드 전환, 고밀도 이벤트 성능 튜닝, 실패 진단 스키마 정착, 다음 주차 전달 항목 정리.
- 비범위: 런 엔진 내부 재작성, 새 스트리밍 프로토콜 도입(대체 전송기 반영).

## 목표
- week01~03 산출물을 운영 가능 수준으로 정리하고, 장애 대응/성능 실패의 원인 분석을 자동화한다.

## 구성요소 분해 (작업 순서별)
- failure-diagnostic-schema: 실패 코드, 루트원인, 재시도 가능성, 권장 조치 포함 스키마.
- JSON-to-DOM 렌더 파이프라인: 이벤트 JSON → view model 정규화 → DOM 출력 규칙.
- ui-host-shell: 외부 시스템 임베드용 우측 사이드바 레이아웃(패널 크기/접힘/독립 마운트).
- view-mode-controller: step-card/DAG/로그 모드 전환 및 상태 라우팅.
- perf-observer: 이벤트 처리량, 렌더 지연, 메모리 사용량 계측.
- release-gate: Phase 조건 충족 여부를 판단하는 게이트.
- issue-handoff pack: 다음 스프린트 이관사항 정규화.

## Iteration 기반 검증(다중 페르소나)
- Iteration-01: SRE/운영 관점으로 failure schema의 실패 코드 체계와 대응 정책 검토.
- Iteration-02: UI/성능 관점으로 렌더 파이프라인 병목 지점 및 토글 깜빡임 리스크 점검.
- Iteration-03: FE 통합 관점으로 host shell과 mode controller 상태 보존 시나리오 점검.
- Iteration-04: PM/운영 관점으로 KPI 산출·release gate 판정의 문서/근거 완결성 검토.

## 의존성
- 선행: week01 schema 계약, week02 상태머신/재동기화, week03 canonical 이벤트/history.
- 내부: 진단 스키마는 `failureContext`를, 렌더 최적화는 history payload 크기와 연동.
- 후행: 다음 분기에서 UX 고도화 및 분산 추적 연동 시 failure schema를 그대로 상속.

## 일정(의존성 낮은 순)
1. failure진단 스키마 v1 확정 및 backend 노출.
2. failure schema 매핑을 통해 history `failureContext`와 1:1 대응 점검.
3. JSON-to-DOM 렌더 파이프라인(view model + DOM 출력) 확정.
4. ui-host-shell 기본 골격 구축(패널 크기/접힘/독립 마운트).
5. 렌더 모드 전환 구조(상태 저장/쿼리 키/깜빡임 억제) 구축.
6. 성능 게이트 지표 수집 및 임계치 적용.
7. Phase 기준으로 오픈 이슈 패키징 및 다음 단계 작업 분해.

## 산출물(PoC 증빙)
- `docs/week-04/release-gate-report.md` (Phase별 통과/차단 근거)
- `docs/week-04/hand-off-template.md` (다음 스프린트 이관 항목: 미해결 이슈, 우선순위, 선행조건)
- `docs/week-04/failure-diagnostic-schema.md` (실패진단 스키마 필수 필드와 매핑)

## 아키텍처 정합성
- 진단 스키마는 week03 history의 `failureContext`와 1:1 대응해 이벤트-이력-UI 진단이 분리 없이 이어지도록 구성.
- Phase 게이트는 기존 계약 위반 탐지와 무관하지 않게 기존 schema 버전/체크섬 기반으로 동작.
- 고밀도 이벤트에서는 UI 파이프라인에서 직접 렌더 대신 요약/버퍼링 레이어를 넣어 아키텍처 정합성 유지.

## 구현 가능성
- 기능 추가보다 정책화(게이트/스키마/측정) 위주이므로 주차 내 완료 가능성이 높다.
- 성능 지표는 계측기 + 임계치 기반으로 결정해 배포 리스크를 낮춘다.

## 정량 완료 기준 (KPI)
- p95 렌더 지연: 300ms 이하(노드 200개, 초당 200 event 입력 시 기준).
- state-hydrate 성공률: 재동기화 시도 대비 99.5% 이상.
- 실패 진단 커버리지: run 실패 케이스의 100%가 코드/범주/권장조치에 매핑.

## Phase
- Phase 0(안정): 핵심 기능 동작, 스키마/이벤트 정합성, 기본 error 매핑 완료.
- Phase 1(운영): 성능 임계치 충족, 재동기화 실패 자동 복구 정책 적용.
- Phase 2(진단): failure schema 도입, 사용자 안내 메시지 자동 생성, 다음 스프린트 handoff 완료.
