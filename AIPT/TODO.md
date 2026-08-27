# AIPT — 남은 작업 (TODO)

이관/병합 핵심 작업은 완료됐다(`MIGRATION.md` Phase 1~4.9 전부 [x],
`pytest tests/ -q -m "not live"` → 433 passed / 1 skipped / 12 deselected).
아래는 그 이후 남은 실제 미해결 항목이다. **이 파일이 남은 작업의 단일
소스(SSoT)다** — 새 대화/세션에서 "남은 작업"을 물으면 이 파일을 먼저 확인할 것.
완료 시 `[x]`로 갱신하고 근거(검증 커맨드/결과)를 한 줄 남긴다.

- [x] 1. **Gateway L3 포워딩 실제 컨테이너 검증** — 완료 (2026-08-27).
  `docker compose build && docker compose up -d`로 web/gateway/mock-server
  3개 컨테이너 실기동. `web`(net-client)과 `mock-server`(net-backend)는
  서로 다른 브리지 네트워크에 격리돼 있어 `gateway`의 IP 포워딩 외에는
  물리적 통신 경로가 없음 — 통신 자체가 경유 증거. 정량 검증: `POST
  /gateway/profile`로 clean→3g(delay 150±40ms, loss 1%, reorder 0.5%) 전환 시
  web→mock-server 왕복 지연이 clean 1.0~1.4ms → 3g 874~1000ms로 실측 증가
  (양쪽 인터페이스(eth0/eth1) 모두 delay 적용되므로 대략 2x150ms 왕복과
  부합). `GET /health`로 `ip_forward_available:true`, `netem_available:true`
  확인. 이후 clean으로 리셋 후 `docker compose down`으로 정리, 잔여
  컨테이너/네트워크 없음 확인.

- [ ] 2. **local_llm 서비스 compose 미통합** — `scripts/run_local_llm_engine.sh`
  (llama.cpp 부트스트랩)는 준비됐지만 `docker-compose.yml`엔 `local-llm`
  서비스 자체가 없음. 현재는 `LOCAL_LLM_ENGINE_URL`로 외부 엔진을 가리키는
  방식만 지원. 필요 시 4번째 서비스로 편입할지 결정 필요.

- [ ] 3. **`aipt/web/store.py` run 이력 영속화 미구현** — 현재 프로세스 메모리
  (`MAX_RUNS=50`)에만 저장, 재시작하면 소실. 모듈 docstring에
  `TODO(persistence)`로 명시.

- [ ] 4. **`/api/run/stream` SSE 엔드포인트 미구현** — 현재 진행상황은 폴링만
  지원. `aipt/web/app.py` docstring에 명시된 TODO.

- [ ] 5. **`routes_run.py`의 pcap 응답 필드 미배선** — `"pcap": None,  # TODO:
  wire aipt.core.capture once a route asks for it`. 실행 결과 응답에 pcap
  경로가 채워지지 않음.

- [ ] 6. **원본 디렉터리 정리 확인** — README에는 `token_traffic/`,
  `tcp_congestion/`이 "병합 완료 후 저장소에서 제거됨"이라 적혀 있으나,
  실제로는 `remote_work/tcp_congestion/`가 아직 디스크에 남아있는 것으로
  확인됨(2026-08-27 기준). 삭제 또는 `_archive/`로 이동 방침 재확인 필요.
