# AIPT — 남은 작업 (TODO)

이관/병합 핵심 작업은 완료됐다(`MIGRATION.md` Phase 1~4.9 전부 [x],
`pytest tests/ -q -m "not live"` → 448 passed / 1 skipped / 12 deselected,
2026-08-27 재검증).
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

- [x] 4. **`/api/run/stream` SSE 엔드포인트 미구현** — 완료 (2026-08-27, 커밋
  `98e4314f`). `routes_run.py`에 `POST /api/run/stream` 라우트 실존 확인
  (`_run_conversation_stream()` 제너레이터 + `_drive_stream_to_queue`로
  threadpool↔이벤트루프 브리지, 턴마다 SSE 이벤트 + `<exec_id>.stream.jsonl`
  서버측 로깅). `MIGRATION.md`에도 완료 기록 있음(라인 586, 631). 재검증:
  `pytest tests/ -q -m "not live"` → 448 passed, 1 skipped, 12 deselected
  (문서와 실측 재확인 결과 일치, 그대로 유지).

- [x] 5. **`routes_run.py`의 pcap 응답 필드 미배선** — 완료 (2026-08-27, 사용자 지시).
  `RunRequest.capture: bool = True`(체크박스도 기본 체크로 변경) 신규 추가.
  `_run_conversation_stream()`이 `connect()` 직후 `backend.api_host()`를
  `_split_api_host()`(신규, 3-backend의 서로 다른 host 표현 — mock의
  `host:port`, public_ai의 순수 hostname, local_llm의 `scheme://host:port`
  URL — 을 `(host, port)`로 통일)로 정규화해 `aipt.core.capture.Capture`를
  실제로 열고, 결과의 `"pcap": None`을 `cap.result()`로 교체. `capture=True`라도
  `aipt.core.capture.available()`(tcpdump 미설치/NET_RAW 부재)이면 조용히
  `pcap=None`로 폴백(기존 계약 유지, 하드 실패 없음).

  **오프로딩/스냅렌 요구사항**(사용자 지시): 캡처 창 동안 TSO/GSO/GRO를
  끄고, snaplen을 MTU/MSS 경계 확인에 충분한 200바이트로 줄임.
  `aipt.core.capture.PCAP_SNAPLEN` 기본값을 100→200으로 변경(이미 있던
  `aipt.core.offload.Window`가 `Capture.__enter__`에서 자동으로 호출되므로
  코드 변경 불필요, 다만 **`docker/Dockerfile.web`에 `ethtool` 패키지가
  누락돼 있어 오프로딩이 실제로는 꺼지지 않고 있었음**(offload.Window가
  "ethtool not installed"로 조용히 폴백) — 추가 설치로 해결.
  `docker-compose.yml`의 `web` 서비스에 `NIC_OFFLOAD_DISABLE=1` 추가(기존
  `NET_ADMIN` capability 그대로 사용, 신규 권한 불필요).

  **실컨테이너 검증**: `docker compose build web gateway mock-server` →
  `docker compose up -d`(3개 컨테이너, local-llm 제외) → `POST /api/run`
  (mock backend, capture=true) 실행 → 응답의 `pcap.offload.during_capture`
  = `{tso:false, gso:false, gro:false}`(이전엔 ethtool 없어서
  `{}`+`"ethtool not installed"`였던 것 확인 후 수정) `.disabled` =
  `["gro","gso","tso"]`, `.snaplen`=200 확인. 실제 pcap을 `docker cp`로
  꺼내 `tcpdump -r`로 열어 진짜 TCP 세그먼트(SYN/데이터/ACK)가 찍혀 있음을
  눈으로 확인. **MTU/MSS 경계 검증**: `web` 컨테이너 안에서 8000바이트
  페이로드를 실제 소켓으로 `mock-server`(gateway 경유, 172.28.1.3→
  172.28.2.3, MTU 1500)에 전송하며 `eth0`(오프로딩 끈 상태)를 캡처 →
  세그먼트 길이가 1448바이트(=1500 MTU - 40 IP/TCP 헤더, MSS 1460과 일치)
  단위로 정확히 쪼개져 있음을 확인(8000바이트 super-packet 없음, 오프로딩이
  실제로 커널 세그먼테이션을 억제했다는 직접 증거). 검증 후 컨테이너 정리
  (`docker compose down`), 테스트용 pcap 삭제.

  **신규 테스트**: `tests/web/test_app.py`에 3개 추가 —
  `test_api_run_capture_defaults_true_and_produces_a_pcap`(capture 기본값
  검증 + 실제 pcap 파일 생성 확인, tcpdump 없는 환경에서는 skip),
  `test_api_run_capture_false_leaves_pcap_none`,
  `test_split_api_host_handles_all_three_backend_shapes`(mock/public_ai/
  local_llm 3가지 host 표현 모두 파싱). `pytest tests/ -q -m "not live"` →
  **451 passed**(기존 448 + 신규 3), 1 skipped, 12 deselected.

- [x] 6. **원본 디렉터리 정리 확인** — 완료 (2026-08-27, 사용자 결정: 완전 삭제).
  `tcp_congestion/`는 git에는 이미 없었음(`3d393be2` 병합 커밋에서 이미
  히스토리 반영, 로컬 디스크에만 미삭제 워킹 디렉토리로 남아있던 것 —
  git 작업 아니라 순수 파일시스템 정리). 에이전트 권한으로 `.venv/`,
  `.pytest_cache/`, `tests/`, 코드 디렉토리를 먼저 삭제(64M→588K), 남은
  root 소유 pcap 14개(`data/pcaps/`, 구버전 tcp_congestion이 tcpdump를
  root로 띄워 생성 — AIPT의 `aipt/core/capture.py`는 `_keep_uid()`로 이미
  해결된 문제)는 사용자가 `sudo rm -rf`로 직접 삭제. 최종 확인:
  `ls ~/repo/remote_work/tcp_congestion` → No such file or directory.
  `token_traffic/`은 이미 이전부터 없었음 — README의 "병합 완료 후 저장소에서
  제거됨" 서술이 이제 실제 디스크 상태와 완전히 일치.
