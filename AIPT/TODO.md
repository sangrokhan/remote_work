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

- [x] 2. **local_llm 서비스 compose 미통합** — 완료 (2026-08-27). `docker-compose.yml`에
  `local-llm` 서비스 신규 추가 (`docker/Dockerfile.local_llm`이 상용
  `ghcr.io/ggml-org/llama.cpp:server` 이미지를 감싸는 방식, 추론 재구현 없음).
  mock-server와 동일한 net-backend 격리 + gateway 경유 라우팅 패턴 적용,
  4-서비스(web/gateway/mock-server/local-llm) 토폴로지로 확장. **포트를
  40000번대(40080)로 통일**: `engine_adapter.DEFAULT_ENGINE_URL`,
  `scripts/run_local_llm_engine.sh`(호스트에서 스크립트로 직접 구동하는
  대안 경로), docker-compose의 `local-llm` 서비스가 전부 8080(AIPT
  자체 `gateway` 포트와 충돌하던 llama-server 기본값)에서 40080으로
  이동해 서로 충돌하지 않음. 포트 이동 중 실제 버그 1건 발견/수정:
  `LocalLLMBackend.ready()`가 인스턴스의 `self._engine_url`이 아니라
  매번 새로 env를 읽는 모듈 레벨 `ready()`를 호출하고 있었음(기존엔
  기본값과 흔한 override가 우연히 같은 8080이라 안 들켰음).
  **실컨테이너 end-to-end 검증**: 4개 이미지 빌드 → 4개 컨테이너 기동 →
  local-llm이 HF Hub에서 `bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M`
  실제 다운로드 후 서빙 확인 → `POST /api/run`(웹 UI가 실제 쓰는 API)으로
  `backend=local_llm` 실행 → gateway 경유 실제 chat completion 응답("OK")
  + wire/cwnd/TCP_INFO 계측값까지 정상 수집 확인. 단, `web`은 net-client
  전용이라 Docker 내장 DNS로 `local-llm` net-backend 호스트명을 해석하지
  못함(mock-server와 동일한 제약) — `LOCAL_LLM_ENGINE_URL` 기본값을
  호스트명 대신 고정 IP(172.28.2.4)로 설정해 우회. `pytest tests/ -q -m
  "not live"` → 433 passed, 1 skipped, 12 deselected (기존과 동일, 회귀 없음).

- [x] 3. **`aipt/web/store.py` run 이력 영속화 미구현** — 완료 (2026-08-27).
  `save_run()`이 메모리(`OrderedDict`, `MAX_RUNS=50`) + 디스크(`RUN_STORE_DIR`,
  기본 `data/runs/<exec_id>.json`) 이중 저장으로 변경. 프로세스 재시작 시
  첫 호출에서 디스크로부터 자동 rehydrate. `MAX_RUNS` 초과로 evict될 때
  디스크 파일도 함께 삭제(token_traffic의 기존 pruning 정책과 동일 취지).
  `get_run()`은 메모리에 없으면 디스크 직접 읽기로 폴백. 디스크 I/O 실패는
  로그만 남기고 삼켜서 run 자체는 실패시키지 않음(honesty-over-crash).
  `docker-compose.yml`에 `RUN_STORE_DIR` env + `./data/runs` 볼륨 마운트
  추가. 신규 유닛테스트 7개(재시작 시뮬레이션, evict 시 파일 삭제, 디스크
  폴백, 손상 파일 스킵, 쓰기불가 디렉토리 방어 등) + 기존 web 테스트
  fixture들 `RUN_STORE_DIR` 격리 처리. `pytest -q -m "not live"` → 446
  passed(작업 중 동시에 진행되던 다른 세션의 mock/public_ai 리팩터링과
  함께 그린 확인). 실제 재시작 시나리오 plain-python으로 직접 검증
  (save → 메모리 초기화 → list_runs()에서 여전히 조회됨).

- [ ] 4. **`/api/run/stream` SSE 엔드포인트 미구현** — 현재 진행상황은 폴링만
  지원. `aipt/web/app.py` docstring에 명시된 TODO. **주의(2026-08-27)**:
  이 저장소에서 동시에 다른 세션이 관련 작업(`routes_run.py`에
  `/api/run/stream` 추가하는 것으로 보이는 변경)을 진행 중인 흔적 발견 —
  착수 전 `git status`/`git log`로 최신 상태 확인 필요, 중복 작업 주의.

- [ ] 5. **`routes_run.py`의 pcap 응답 필드 미배선** — `"pcap": None,  # TODO:
  wire aipt.core.capture once a route asks for it`. 실행 결과 응답에 pcap
  경로가 채워지지 않음.

- [ ] 6. **원본 디렉터리 정리 확인** — README에는 `token_traffic/`,
  `tcp_congestion/`이 "병합 완료 후 저장소에서 제거됨"이라 적혀 있으나,
  실제로는 `remote_work/tcp_congestion/`가 아직 디스크에 남아있는 것으로
  확인됨(2026-08-27 기준). 삭제 또는 `_archive/`로 이동 방침 재확인 필요.
