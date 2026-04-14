# Week 4 Context

## 왜 이 주차인가
- week01~03이 기능을 완성했으면, 마지막 주는 운영 품질(성능·가시성·진단) 단계로 이동한다.
- 특히 실패 원인 추적 체계를 고정하지 않으면 주간 결과물이 운영 단계에서 재해석되지 못해, 다음 단계 인수테스트에서 지연이 발생한다.

## 핵심 반영: 정량 기준 / Phase / 실패진단 스키마
- 정량 기준(KPI):
  - 상태 재동기화 성공률, 렌더 지연, 실패 진단 매핑 커버리지.
- Phase는 요구 충족 순서를 gate로 관리해 partial 배포를 허용(phase0→phase1→phase2).
- 실패진단 스키마:
  - `failureCode`, `failureCategory`, `nodeId`, `errorAt`, `retryable`, `rootCause`, `resolutionHints`, `evidenceRefs`.

## 산출물
- `release-gate-report.md`에 phase별 판정(PASS/REJECT), KPI 수치, 미해결 risk 기록.
- `hand-off-template.md`에 다음 스프린트 이관 범위만 남기고, 새로운 기능추가는 보류한다.
- `failure-diagnostic-schema.md`에 실패 코드/범주/권장조치 필드와 우선순위 매핑을 고정한다.
- `ui-host-shell`을 통해 우측 사이드바 임베드 동작(패널 너비/접힘/재진입) 기준을 정리한다.

## PRD와의 정합성
- 9장 표현 방식, 13장 성능/확장성, 15항 운영 오픈 이슈 정리에 대응.
- 8.4 에러 처리와 11.2 이벤트 모델을 결합해 오류 원인에서 UI 진단까지 경로를 닫는다.

## 구성요소 분해
- 진단 서비스: 실패 이벤트 수신 시 코드화·카테고리화.
- 표출 서비스: 모드 전환 시 렌더 비용이 큰 뷰 최소화.
- 관측성 서비스: KPI 수집기, 이벤트 드롭/지연 모니터.
- 운영 전이기: Phase 달성 체크, fail-fast/rollback 조건 정의.
- 임베드 호스트 서비스: 외부 시스템에서 사이드바로 호출 가능한 렌더 컨테이너 규격화.

## 구현 가능성
- 기존 UI/이벤트 흐름을 유지한 채 계측기와 표시 정책만 추가해 리스크 낮음.
- 실패진단 스키마는 API 응답에 optional field로 시작해 점진적으로 확장 가능.
