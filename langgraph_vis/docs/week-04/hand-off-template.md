# Week 4 Hand-off Template

## 한계점/미완료 항목

1. 렌더 성능 KPI 자동 계측 미구현
   - Owner: platform
   - Priority: High
   - 기준: p95 렌더 지연 임계치(300ms) 자동 수집 및 임계치 경보
   - 선행조건: 이벤트 렌더 뷰 계측 포인트 정의

2. UI host-shell 및 view-mode-controller PoC 미구현
   - Owner: frontend
   - Priority: Medium
   - 기준: 2개 모드 토글 동작, 패널 open/close 상태 보존
   - 선행조건: 최소 렌더 파이프라인 정합성 확정

3. release gate 자동 연결
   - Owner: qa
   - Priority: Medium
   - 기준: CI 단계에서 phase gate 판정 산출물 자동 생성
   - 선행조건: KPI 측정 스크립트/리포트 파서 준비

## 다음 스프린트 입력값

- 우선순위 상위: Week4 잔여 항목(Host shell, 렌더 모드, 성능 observer)
- 입력값: `failureContext` 스키마(v1), history API 응답, 기존 run state store 이벤트 로그
- 완료기준:
  - 성능 KPI 자동화가 CI/PR에서 재현 가능한 형태로 구동
  - 렌더 mode 토글 및 host shell에서 데이터 소실 없이 최소 2회 반복 동작
  - 진단 스키마를 화면/로그/이벤트에 일관 노출
