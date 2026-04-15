# Week 4 Checklist

## 구현 체크리스트
- [x] `docs/week-04/failure-diagnostic-schema.md` 작성/리뷰 및 backend 노출(표시/참조 경로) 확인.
- [x] `docs/week-04/failure-diagnostic-schema.md` 값 매핑 확인(`failureCode`/`failureCategory`/`rootCause`/`resolutionHints`).
- [x] `JSON 워크플로우/이벤트 payload`를 JS view model로 정규화 후 HTML 노드 렌더링 기본 파이프라인 구축.
- [x] `우측 사이드바 임베드 shell(컨테이너 고정/너비/접힘/close/open)` 구현해 최소 PoC 화면에 반영.
- [x] `view-mode-controller`(step-card / DAG / 로그) 스위치 안정화 및 상태 유지.
- [x] `queryKey`/`mode` 조합으로 깜빡임 없는 토글 및 state 잔존성 보장.
- [x] performance KPI 지표 수집기 연결(렌더 소요시간, 이벤트 처리량, 메모리 피크).
- [x] view-mode PoC 1개 시나리오: step-card 또는 DAG/로그 중 최소 2개 mode에서 렌더 토글 1회 동작 시연.
- [ ] Phase gate를 CI/PR 체크포인트로 반영(phase0/1/2별 PASS/REJECT 분기 명시).
- [x] `docs/week-04/release-gate-report.md` 작성 및 pass/reject 근거를 텍스트로 고정.
- [x] `docs/week-04/hand-off-template.md` 작성 및 다음 스프린트 입력값(미해결 항목/우선순위/선행조건) owner, 완료기준 포함 고정.
- [ ] 재시도/재동기화 실패 안내 메시지 1개와 회귀 이슈 템플릿 1개를 hand-off 문구로 정리.
- [x] 구성요소 분해 완료: failure-diagnostic schema, 렌더 파이프라인, host-shell/mode controller, perf observer, release gate, hand-off.

## 테스트 커버리지
- [ ] 단위 테스트: 실패진단 스키마 필드 유효성 및 값 사상 매핑.
- [ ] 문서 검증: `failure-diagnostic-schema.md`의 필드와 UI 진단 표시 키가 1:1 매핑되는지 확인.
- [ ] 문서 검증: `failure-diagnostic-schema.md`의 `evidenceRefs`가 진단 패널 또는 진단 로그에 표시되는지 확인.
- [x] 통합 테스트: JSON 샘플 fixture 1건을 JS로 파싱해 view model 변환 후 HTML 렌더 결과가 기대 구조(노드/에지/상태)와 일치함을 확인.
- [x] 통합 테스트: 우측 사이드바 shell에서 최소 1개 run 시나리오 렌더링 및 닫기/열기 동작 시 데이터 유지 여부 확인.
- [x] 통합 테스트: queryKey+mode 동작에서 상태 보존 + 불필요 리렌더 0건을 PoC 시나리오 1회 기록으로 확인.
- [x] 통합 테스트: mode 전환 10회 반복 시 state 잔존성/메모리 유실 없음.
- [ ] 통합 테스트: 실패 run 1건에서 진단 패널 표시(E2E smoke).
- [ ] 통합 테스트: plan KPI(`p95<=300ms`, 노드 200개/초당 200 event 기준) 수동 측정.
- [ ] 통합 테스트: `performance metrics` 수집 파이프라인이 `release-gate-report`의 KPI 섹션에 연결되는지 점검.
- [ ] 통합 테스트: phase0/1/2 게이트 조건 위반 시 release 차단 동작 검증.
- [ ] 통합 테스트: CI/PR gate에서 phase0/1/2 상태가 release-gate-report의 pass/reject와 일치하는지 확인.
- [ ] 운영 체크: `state-hydrate` 재동기화 성공률(목표 99.5%)을 최소 PoC 지표로 산출물에 기록.
- [ ] 운영 체크: 운영 체크(`release-gate-report`)에서 측정한 KPI 결과와 phase gate 결정을 근거화.
- [ ] 운영 체크: 실패 케이스 100%가 진단 패널에 rootCause/resolutionHints 출력되는지 회귀 테스트.
- [ ] E2E 체크리스트: run 실패 → 진단 패널 → 사용자 가이드 노출 전체 흐름.
- [ ] E2E 체크: view-mode PoC(2 mode 1회 토글) 완료 후 결과를 `release-gate-report`에 상태 및 합격판정으로 기록.
- [ ] 문서 검증: `hand-off-template`의 다음 스프린트 미해결 항목/owner/우선순위/완료기준이 채워졌는지 확인.
