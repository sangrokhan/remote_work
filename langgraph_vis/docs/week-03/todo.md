# Week 3 TODO (잔여 이슈)

## P0 / 꼭 해소해야 하는 문제
- [ ] Week 2 기반에서 추가되는 `datetime.utcnow()` 경고(`python_backend/app/run_state_store.py`) 정리
  - 현재 기능에는 영향이 없지만, Python 3.13+ 호환성과 정적 분석에서 경고 정리를 위해 `timezone-aware` 타임스탬프(`datetime.now(datetime.UTC)`)로 교체 필요.

## Week 3 종료 기준 미충족 영역
- [ ] `docs/week-03/checklist.md`의 구현/테스트 항목이 [x]로 갱신되지 않음
  - 문서상 진행 상태 추적이 불명확해 다음 주차 인수인계 시 품질 위험 발생 가능.

## Week 3에 보완할 성능/안정성 항목
- [ ] 실제 `canonical` 스트림 입력(역순/누락/중복 케이스)에 대한 파이프라인 스트레스/복구 케이스 테스트가 아직 문서-구현 1:1로 증빙되지 않음
  - 현재 테스트는 핵심 기능 정합성은 통과했으나, 스트림 이상 시나리오를 대규모로 검증할 수 있는 시나리오가 추가 필요.
- [ ] `history`와 `stream` 간 패리티 비교 규격의 자동검증(파서/매핑 룰과 결과물 checksum 비교 자동화) 미구현
  - `docs/week-03/parity-test-plan.md` 계획은 있으나 실행 가능한 자동 테스트 파이프라인은 미구축.

## 다음 주차 인수인계용 메모
- [ ] JS 테스트 스위트는 문서 및 backend 중심 PoC 성격으로 현재 정적 상태 (`Frontend JS tests are not maintained here`)이므로, week4 초반에 js 테스트 정리/이관 범위 재확정 필요.
