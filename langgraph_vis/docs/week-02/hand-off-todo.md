# week03 인계 TODO (week02 발견 이슈)

## 진행 조건: 다음 주차 진입 전 반드시 처리

- [ ] 1) `docs/week-02/run-api-contract.yaml`과 Python 구현의 에러 계약 정합성 통일
  - `INVALID_RUN_PAYLOAD`/`405`, `InvalidEvent` 매핑, 메시지 포맷, 상태 코드의 명세-코드 일치 여부 정리
  - 문서: `docs/week-02/run-api-contract.yaml`, `python_backend/app/run_error_contract.py`, `python_backend/app/run_state_api.py`

- [ ] 2) 재연결 정책( `fromSeq` / `lastEventId` ) 명세 단일화
  - `lastEventId` 미일치(Unknown) 처리 정책을 현재의 자동 폴백(`0`)이 아니라 실패 or 명시적 경고로 통일할지 결정
  - header/query 우선순위 및 결과 cursor 의미를 1개 문단에 고정
  - 문서: `docs/week-02/reconnect-guide.md`, `docs/week-02/run-api-contract.yaml`, `python_backend/app/resync_controller.py`, `python_backend/app/run_state_api.py`

- [ ] 3) SSE 이벤트 실시간성 요건 확정
  - 현재 스트림은 연결 시점 스냅샷 리플레이 + heartbeat 중심이므로, 실시간 이벤트 push가 필요한지 여부를 결정
  - 필요 시 stream producer/subscribe 동작 및 테스트 범위 추가
  - 문서: `docs/week-02/sse-envelope.yaml`, `python_backend/app/run_state_api.py`, `python_backend/tests/test_run_state_api.py`

- [ ] 4) timestamp/`issuedAt` 포맷 정합
  - contract(`date-time`)와 일치하도록 timezone-aware 값 통일(UTC suffix `Z` 또는 `+00:00` 규칙)
  - 문서: `docs/week-02/sse-envelope.yaml`, `python_backend/app/run_state_store.py`, `python_backend/app/sse_envelope.py`

- [ ] 5) SSE envelope 필드 안정화
  - extra 필드 허용 범위를 허용할지 금지할지 계약에 고정
  - 고정한다면 구현에서 임의 메타 필터링 정책 추가
  - 문서: `docs/week-02/sse-envelope.yaml`, `python_backend/app/sse_envelope.py`, `python_backend/app/run_state_store.py`

- [ ] 6) state cursor와 stream cursor 정합 문서화
  - `/state` 응답 cursor와 stream replay cursor(`fromSeq` 계산)에 대한 일치 규칙(필드/의미/빈값 동작) 정리
  - 문서: `docs/week-02/run-api-contract.yaml`, `python_backend/app/run_state_api.py`, `python_backend/app/run_state_store.py`

- [ ] 7) 테스트 재현성 개선
  - 루트에서 바로 재현 가능한 실행 명령으로 정리 (`PYTHONPATH=. pytest -q python_backend/tests`)
  - 문서: `tests/README.md`, `python_backend/requirements-dev.txt`
  - 남은 테스트 갭: `run_reducer.py`, `schema_api.py`, `main.py`, `error_contract.py`, `run_error_contract.py` 등 핵심 경로 직접 테스트 추가

- [ ] 8) 데드타임/오류 대응 리스크 정리
  - 메모리 스토어 영속성/재기동 손실 리스크 문서화
  - next-step에서 최소한의 영속 전략(디스크 스냅샷 또는 단일 영속 레이어) 계획 추가
  - 문서: `docs/week-03/plan.md`, `python_backend/app/run_state_store.py`
