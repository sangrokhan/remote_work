# LangGraph Workflow 실행 시각화 구현 계획(4주 단위)

## 목표
PRD(`prd.md`) 기준으로, 의존성이 적고 단위가 작은 작업부터 구현해 backend-first 단계를 반복하면서 frontend로 확장한다.

## 우선순위 전략
- 계약 우선: API/이벤트 스키마를 먼저 고정해 backend/frontend 동시 의존성 충돌을 줄인다.
- 점진적 스트리밍: state-only run 조회 → event stream → token/history 확장.
- 제어 반전: backend를 Source of Truth로 두고 frontend는 schema 기반 렌더러로 고정.
- 프론트는 backend JSON을 받아 JavaScript로 정규화·상태 적용 후 HTML DOM 렌더한다.
- 외부 시스템 탑재 전제의 우측 사이드바 형태 렌더 레이아웃을 4주차 말기 목표에 포함.
- 오픈 이슈는 초기 주차에서 결정하고 이후 주차는 구현 고정값으로 진행.

## 주차 맵
- 1주차: 기반 설계/계약 확정 + schema 조회 API
- 2주차: run 실행 + 이벤트 스트림 기초
- 3주차: state/history + 토큰/중간 산출물 표시
- 4주차: 렌더링 고도화 + 안정성/성능/재실행 검토
