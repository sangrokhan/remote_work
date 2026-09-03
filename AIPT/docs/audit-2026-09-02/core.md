# core 모듈 감사

**대상**: `aipt/core/{cache_protocol,capture,config,congestion,cwnd,idle_reset,netem,offload,quic_congestion,streaming,tcpinfo,wire}.py`, `aipt/core/__init__.py`
**참고**: `tests/core/*` (실제 동작 근거), `aipt/web/routes_gateway.py` (idle_reset 유일 호출부)
**방법**: 코드를 먼저 전량 읽고(1), 구현에서 요구사항을 역추론해 Task 카드를 만들고(2), 호출 그래프를 그린 뒤(3), 마지막에 DESIGN.md/ARCHITECTURE.md/MIGRATION.md와 대조(4)했다. 코드 수정 없음 — 감사 전용.

---

## 1. 구현 현황(함수별)

### 1.1 `__init__.py`
- 1줄 docstring만 존재: `"aipt.core -- measurement infrastructure shared by every AIPT backend."` 실질 코드 없음.

### 1.2 `config.py` — env var 파싱의 단일 소스
- `flag(name)` (L37-39): `os.environ.get(name)`을 `{"1","true","yes","on"}` 집합과 대소문자 무시 비교. 관대한 truthy 파서.
- `flag_any(*names)` (L42-49): 여러 이름 중 하나라도 truthy면 True — 정식명+deprecated alias 패턴에 사용(`offload.py`의 `NIC_OFFLOAD_DISABLE`/`TRAFFIC_PCAP_NO_OFFLOAD`가 실제 사용처, L106-110).
- `env_int(name, default)` (L52-69): int 파싱 실패/공백 시 default로 폴백(예외를 던지지 않음). `netem.parse_delay`와 의도적으로 대비되는 문서화(L55-60): 후자는 잘못된 값에 대해 **일부러 raise**.
- `env_str(name, default)` (L72-76): 단순 wrapper.
- `is_mock(provider="")` (L79-84): `TRAFFIC_MOCK` 전역 플래그 OR `{PROVIDER}_MOCK` 플래그.
- 모듈 docstring이 이 파일의 존재 이유를 명시: 과거 `TRAFFIC_MOCK=true`가 provider별로 다르게 파싱되어 "synthetic인데 live 버킷에 기록됨" 버그가 있었음(L5-12) — mock/live 오분류를 구조적으로 막기 위한 단일화.

### 1.3 `congestion.py` — 커널 TCP 혼잡제어 가용성
- `TCP_CONGESTION = getattr(socket, "TCP_CONGESTION", 13)` (L27): sockopt 상수, 없으면 13(Linux 고정값) 하드코딩 폴백.
- `available_algorithms()` (L38-58): `/proc/sys/net/ipv4/tcp_available_congestion_control`를 매 호출마다(임포트 시 캐시 아님) 읽어 `(names, reason)` 반환. 읽기 실패/빈 파일이면 빈 리스트+사유, **절대 하드코딩된 fallback 목록을 지어내지 않음**(L47-49) — 드롭다운에 없는 알고리즘을 보여줘서 나중에 조용히 실패하는 것을 방지.

### 1.4 `quic_congestion.py` — QUIC(유저스페이스) 버전
- `available()` (L25-36): `import aioquic`가 성공하는지만 확인. `congestion.py`(커널)와 분리된 이유가 문서화됨(L7-13): QUIC 혼잡제어는 커널이 아니라 aioquic 유저스페이스에 있어 `/proc` 파일이 없음.
- `available_algorithms()` (L39-72): aioquic의 `reno`/`cubic` 모듈을 import해 factory 등록을 트리거하고, 추가로 `aipt.backends.quic_mock.congestion`을 import해 프로젝트 자체의 `idle_probe` 알고리즘을 등록(best-effort, L61-67). `cc_base._factories.keys()`를 정렬해 반환.

### 1.5 `netem.py` — tc netem 지연 주입 (entrypoint 전용)
- `parse_delay(value)` (L39-44): 빈 문자열→0, 음수→0, 그 외 `int()` 파싱(비숫자는 raise — config.env_int과 대비되는 "실패 시 크래시" 정책).
- `build_commands(iface, delay_ms)` (L47-63): delay_ms==0이면 빈 리스트. 아니면 `tc qdisc del`(idempotent용) → `netem delay` 를 root(`1:`)에 설치 → **`fq`를 netem의 child(`10:`)로 체이닝**. 이유가 docstring에 정확히 명시(L12-21): `netem`을 root에 걸면 root qdisc 전체가 교체되어 `net.core.default_qdisc=fq`가 무의미해지고 BBR pacing이 조용히 사라짐. `parent 1:`으로 fq를 자식으로 붙여야 지연 주입과 BBR pacing이 동시에 적용됨.
- `apply(iface, delay_ms, dry_run=False)` (L70-76): `dry_run=True`면 명령만 반환하고 실행 안 함.
- `from_env()` (L79-85): `NETEM_DELAY_MS`(기본 "0")와 `NETEM_IFACE`(기본 "eth0") 읽음.

### 1.6 `offload.py` — NIC segmentation offload (두 API 공존)
두 개의 독립 계보가 병합된 모듈(docstring L4-27에 명시):
- **capture-time API** (`read`/`Window`/`describe`/`current`) — 캡처 1회 동안 선택적으로 끄고 정확히 복원.
- **entrypoint-time API** (`build_commands`/`apply`/`from_env`) — 컨테이너 시작 시 한 번에 영구적으로 끔.

구현:
- `enabled()` (L103-110): `config.flag_any("NIC_OFFLOAD_DISABLE", "TRAFFIC_PCAP_NO_OFFLOAD")` — 정식명+deprecated alias.
- `egress_iface(target)` (L117-132): `ip route get <target>`을 파싱해 실제 egress 인터페이스를 알아냄(`TRAFFIC_PCAP_IFACE`의 기본값 `any`는 ethtool이 이해 못 하는 pseudo-interface이므로 필요).
- `read(iface)` (L135-161): `ethtool -k <iface>` 파싱, `{feature: {"on": bool, "fixed": bool}}` 반환. `[fixed]`(변경 불가)를 별도로 표시.
- `_set(iface, values)` (L164-189): 한 번의 `ethtool -K` 호출로 여러 feature 동시 적용(링 재초기화 횟수 최소화 목적, L167-169).
- `class Window` (L192-271): `__enter__`에서 `before = read(iface)` 저장 → `enabled()`이면 on&not-fixed인 feature들만 off → `__exit__`/`restore()`에서 **정확히 이전 값으로**(단순히 "on"이 아니라) 복원(L195-198). `result()`가 pcap의 자기서술용 딕셔너리 생성.
- `current(target="1.1.1.1")` (L274-288), `describe(state)` (L291-300): 프리플라이트/UI용 요약.
- entrypoint 함수들 (L311-343): `build_commands`/`_run`/`apply`/`from_env` — `netem.py`와 동일한 구조. `ENTRYPOINT_FEATURES = ["tso","gso","sg","gro","lro"]`(capture-time의 `FEATURES=("tso","gso","gro")`보다 넓음).

### 1.7 `capture.py` — tcpdump 기반 pcap 캡처
- `apparmor_blocks(path)` (L67-87): `$HOME`의 dot-디렉토리 하위 경로면 True. Ubuntu tcpdump AppArmor 프로파일이 `@{HOME}/.*/** mrwkl`을 deny하기 때문(deny beats allow) — 실제로 시간을 낭비했던 이슈(module docstring L31-39).
- `pcap_dir()` (L101-109): `TRAFFIC_PCAP_DIR` 매 호출마다 읽음(import-time 캐시 금지 — 웹앱/테스트에서 첫 import가 항상 먼저 일어나므로).
- `PCAP_SNAPLEN = int(env "TRAFFIC_PCAP_SNAPLEN", "200")`: 헤더만 저장, TLS payload는 버림.
- `_filter_expr(ips, port, proto)` (L281-309): `tcp`(기본) 또는 `udp`(QUIC). **명시적 `and`** 사용 필수 — `any` 인터페이스(LINUX_SLL2)에서 shorthand(`udp port N`)가 BPF에서 "received by filter"엔 매치되지만 파일엔 0 packets가 기록되는 실측 버그가 있었음(L294-305).
- `can_raw_capture()` (L224-242): tcpdump 자체가 NET_RAW capability를 가졌는지 tcpdump를 직접 구동해(`_probe_tcpdump`) 확인. setcap은 바이너리에 부여되지 파이썬 프로세스엔 안 되므로 자체 소켓으로 프로빙하면 항상 실패로 나옴(L227-232).
- `available()` (L251-270): TRAFFIC_PCAP_DISABLE, tcpdump 미설치, NET_RAW 없음, AppArmor 차단 순서로 체크, 각 실패에 대해 해결 방법을 담은 사유 문자열 반환.
- `timestamp_source(iface)` (L335-371): `ethtool -T <iface>`로 하드웨어 vs 소프트웨어 타임스탬핑 판별. DESIGN.md B13에 대응(주석에 명시).
- `safe_pcap_path(name)` (L374-382): 파일명 정규식 검증 + path traversal 방지(디렉토리 탈출 체크).
- `class Capture` (L385-599): context manager.
  - `__init__`: label 조립 규칙(synthetic_mock 스타일 `label=` 직접 지정 vs external_api 스타일 `provider/arm/kind`로 자동 조립: `{provider}_{arm}_{kind}`). `proto` 파라미터(기본 "tcp") — QUIC은 "udp" 명시 필요.
  - `__enter__`: O_EXCL로 파일 원자적 예약(경합 방지) → `offload.Window` 시작(tcpdump 시작 **전**, offload 상태가 캡처 중간에 바뀌는 것 방지) → tcpdump Popen → `time.sleep(0.4)`로 초기화 대기.
  - `__exit__`: `time.sleep(0.6)` 후 SIGINT — 매우 빠른 트래픽(QUIC, <1ms)에서 tcpdump가 마지막 버스트를 read()하기 전에 죽는 실측 버그의 완화책(L502-521, 재현 보고 있음).
  - `result()`: pcap 크기, tcpdump stats(`_parse_tcpdump_stats`), offload 상태, timestamp_source를 포함한 딕셔너리.

### 1.8 `cwnd.py` — 연속 congestion-window 모니터
- C 헬퍼(`native/cwnd_monitor.c`, netlink sock_diag)를 서브프로세스로 실행하고 stdout NDJSON을 리더 스레드로 소비.
- `DEFAULT_INTERVAL_MS = 2`: RTO 이후 slow start가 10→65 segment로 회복하는 데 ~10ms 걸리는 실측 근거로 선택(모듈 docstring L76-84).
- `interval_from_rtt(rtt_ms, k=DEFAULT_RTT_K=5.0, min_interval_ms=1)` (L169-199): `interval_ms = max(min_interval_ms, rtt_ms/k)`. RTT 힌트 없으면 `(DEFAULT_INTERVAL_MS, "fixed")`. 매우 짧은 경로에서 `min_interval_ms`로 clamp되면 사유가 `"floor_clamped"`. DESIGN.md B12에 대응.
- `build()` (L216-238): C 컴파일러(cc/gcc)로 `native/cwnd_monitor.c`를 즉석 빌드.
- `available()` (L241-282): 캐시된(`_probe`) 프로브 결과. `TRAFFIC_CWND_DISABLE` 체크 → 리눅스인지 체크 → 바이너리 존재/빌드 → 실행권한 → 실제로 20ms 실행해서 `"type":"end"` trailer 확인(단순 stat이 아니라 실제 실행 — gVisor/seccomp에서 바이너리는 있지만 AF_NETLINK가 막힌 경우를 잡기 위함, L244-249).
- `class Monitor` (L300-598):
  - `__init__`: `interval` 명시 > `rtt_hint_ms` 주면 adaptive > 둘 다 없으면 `interval_ms()`(env 또는 고정 2ms). `PublicAIBackend`는 `rtt_hint_ms`를 절대 넘기지 않는다고 문서화(L331).
  - `__enter__`: 헬퍼 프로세스 spawn + reader 스레드 시작.
  - `announce(sock)` (L390-424): 소켓을 헬퍼에 `track <local_ip> <local_port> <peer_ip> <peer_port>` 라인으로 알림. `peer_port != self.port`면 무시. v4-mapped v6(`::ffff:`) 정규화.
  - `stop()` (L426-465): idempotent, SIGTERM(트레일러를 쓸 시간 확보) → timeout 시 kill.
  - `_drain()` (L467-500): NDJSON 파싱, 잘못된 줄은 skip(치명적 아님).
  - `result()` (L503-553): sockets, interval_reason, measurement_confidence, dumps/exact_queries(비용 검증용), `idle_resets(...)` 병합.
- `idle_resets(samples)` (L562-598): **이 모듈의 존재 이유가 되는 핵심 판정 함수**. 소켓별 peak cwnd 추적 → `prev > INIT_CWND(10) and cwnd <= INIT_CWND and ca_state == "open"`일 때만 리셋으로 카운트. **`ca_state == "open"` 조건이 핵심** — loss recovery로 인한 window 축소를 idle-reset과 구분(L565-568, "다른 원인이므로 합치면 손실 많은 네트워크가 idle-reset 문제로 위장하거나 숨겨짐").

### 1.9 `idle_reset.py` — sysctl 토글 (★정밀 분석 대상)
- `IDLE_RESET_PATH = "/proc/sys/net/ipv4/tcp_slow_start_after_idle"` (L43): 파라미터화되어 테스트가 스크래치 파일을 가리킬 수 있음.
- `read(path=IDLE_RESET_PATH)` (L57-70): "1"→(True,"ready"), "0"→(False,"ready"), 그 외/에러→(None, 사유). 절대 raise 안 함.
- `write(enabled, path=IDLE_RESET_PATH)` (L73-85): "1"/"0" 쓰기. OSError를 잡아 (False, 사유) 반환.
- `status(path=...)` (L88-92): `{"ok": enabled is not None, "enabled": ..., "reason": ...}`.
- **모듈 docstring이 이 파일의 설계 이력 전체를 명시적으로 담고 있음** (L13-26):
  > "원 설계(2026-09-01)는 *responding* side(mock-server/local-llm)를 토글했다... 그 설계에 대한 인과 실험(`docs/experiments/2026-09-01-idle-reset-results.md`)이 실제로 중요한 지표(다음 턴 요청 업로드 지연)는 응답측이 아니라 *client*(web)의 자체 송신측 cwnd에 의해 좌우됨을 발견했다. **2026-09-02 재설계**(operator 지시): 이 모듈은 이제 오직 `web` 자신에 의해서만 in-process로 import된다(`aipt/web/routes_gateway.py`의 GET/POST `/api/idle-reset`); 예전 mock-server의 `/admin/idle-reset` 라우트와 local-llm의 `docker/idle_reset_admin.py` 사이드카(이 모듈을 같은 방식으로 import하던)는 재설계로 도달 불가능해진 죽은 코드로서 삭제됨."

  → **결론: `idle_reset.py`는 오직 client(`web` 프로세스) 자신의 소켓/netns에만 적용된다. server/responding 측(mock-server, local-llm)에 적용되는 경로는 현재 코드베이스에 존재하지 않는다(과거엔 있었으나 제거됨).**

- 호출부 확인: `aipt/web/routes_gateway.py` L94 `from aipt.core import idle_reset as _idle_reset`, L97-106 `_web_client_idle_reset_status()`/`_web_client_idle_reset_write()`가 `_idle_reset.status()`/`_idle_reset.write()`를 직접(파라미터 없이 = 기본 `IDLE_RESET_PATH`, 즉 `web` 프로세스 자신의 `/proc/sys`) 호출. 주석(L88-93): `"web 자기 자신의 netns를 이미 소유하므로 별도 프록시 불필요"`.
- `docs/experiments/2026-09-01-idle-reset-results.md`가 실측 근거: 2026-09-01 재설계 실험에서 `web_client` 측 idle_reset을 토글(mock-server 측은 두 조건 모두 enabled=1로 **고정**, L52-54 `"mock-server 측 idle_reset은 양쪽 조건 모두 enabled=1(기본)로 고정 — 이번엔 응답 방향이 아니라 업로드 방향만 보는 것이므로 무관"`)했고, 업로드 지연이 최대 407배 차이(L155-158)로 나타남 — 이 실측 결과가 현재 코드의 "client-only" 설계를 정당화하는 증거.

### 1.10 `cache_protocol.py` — 요청 body leaf-hash 중복 제거
- 클라이언트(`aipt/backends/local_llm/gateway.py`)와 서버(`docker/engine_gateway.py`)가 공유하는 stdlib-only 프로토콜(hashlib/re만 사용, `aipt.core` 형제 모듈 의존 없음 — `local-llm` 이미지의 최소 슬라이스에 그대로 복사되기 때문, L7-10).
- `CACHE_HEADER="X-AIPT-Cache"`, `CACHE_HEADER_VALUE="enable"`: opt-in 헤더. 없으면 두 쪽 모두 이 프로토콜이 존재하지 않는 것처럼 동작(L56-58).
- `CACHE_MAP_FIELD="$aipt_cache_map"`: body 최상위에 추가되는 북키핑 필드.
- `HASH_LEN=20`(sha256 앞 20 hex=80bit), `DEFAULT_THRESHOLD_BYTES=200`(짧은 값은 dedup 대상 아님).
- `compute_hash`, `path_to_label`/`parse_label`(경로↔라벨 상호변환, 잘못된 라벨은 ValueError), `get_at_path`/`set_at_path`, `iter_string_leaves`(dict/list 재귀, `CACHE_MAP_FIELD` 자신은 재귀 skip).
- `class SessionCache` (L158-188): dumb dict 2개(hash↔value), TTL/eviction 없음 — 세션=TCP keep-alive 커넥션 생애와 일치(설계 문서 의도).
- `encode_body(body, cache, threshold_bytes=200)` (L196-226): **클라이언트측**. `body`를 deep-copy해 원본 불변 보장(caller의 `messages`가 멀티턴 히스토리라 평문 유지 필요, L198-201). threshold 이상 && 이미 본 값 → hash로 치환 + `$aipt_cache_map`에 기록. 처음 보는 값은 그대로 두되 기록(다음부터 치환 대상).
- `class CacheMiss(Exception)` (L229-238): `missing_paths` 보유.
- `decode_body(body, cache, threshold_bytes=200)` (L241-288): **서버측**. 먼저 모든 매핑된 hash가 알려져 있는지 **사전 검증**(부분 mutation 관찰 방지, L251-253) → 없으면 CacheMiss raise. 있으면 원복 + `$aipt_cache_map` 제거 + 매핑 안 된 leaf도 대칭적으로 학습(symmetric learning, L275-287).

### 1.11 `streaming.py` — SSE 응답 타이밍
- `StreamResult` dataclass: `req_sent_ms/ttfb_ms`(wire에서 채움)/`ttft_ms/ttlt_ms/turn_end_ms/text/events/raw/error`.
- `since(t0, mark, fallback=0)`: mark가 None이면 fallback(0을 "즉시"로 오독 방지).
- `_iter_data(resp)`: SSE `data:` 라인 파싱, `[DONE]`/파싱 불가는 이벤트로는 skip하지만 바이트 카운트엔 이미 반영됨.
- `read_stream(resp, text_of, t0)`: `text_of(event)`로 provider별 answer 텍스트 추출(reasoning/thought 텍스트는 answer가 아님 — TTFT를 부정확하게 만들기 때문, L24-29). 답변 텍스트가 전혀 없으면 ttft/ttlt를 turn_end로 고정(0이 아니라).

### 1.12 `tcpinfo.py` — 1회성 TCP_INFO 스냅샷
- ctypes/struct로 `struct tcp_info`를 부분 파싱(delivery_rate=offset 200, 4.9+ 커널까지 커버). `snapshot(sock)`이 `{cwnd, rtt_ms, rto_ms, delivery_rate}` 반환, 비-Linux/구버전 커널은 전부 0(크래시 없음).
- `cwnd.py`(연속 netlink 모니터)의 경량 대안으로, 다른 backend에서도 재사용 가능하도록 core에 위치(모듈 docstring 근거).

### 1.13 `wire.py` — 소켓 바이트 카운터 + congestion 알고리즘 pin
- `_wire_tally`(모듈 전역): 커넥션 풀링 생존을 위해 인스턴스가 아닌 전역에 카운트(L18-22).
- `wire_counter()` contextmanager: 블록 진입/이탈 시 tally 차이로 sent/recv, `last_send_at`/`first_recv_at` 계산.
- `_CountingSocket`/`_CountingReader`: sendall/send/recv/recv_into/makefile 오버라이드, 응답 첫 바이트만 마킹(`_mark_recv`).
- `_connect_watchers`/`watch_connections(fn)`/`_announce(sock)`: 신규 커넥션이 열리자마자 구독자(`cwnd.Monitor.announce`)에게 통지 — 100ms rediscovery 타이머로는 3ms RTT 경로에서 초기 윈도우를 놓치는 실측 문제 회피(L196-201).
- `set_congestion_algorithm(algo)`/`congestion_algorithm_result()`: 세션 전역 상태로 다음 커넥션에 적용할 알고리즘을 pin. 실제 반영 여부는 소켓에서 다시 읽어(`getsockopt`) `actual`로 검증(요청한 값을 믿지 않고 확인, L254-259).
- `_CountingConnection._new_conn`(L285-326): urllib3의 `socket_options`를 안 쓰고 직접 오버라이드하는 이유는 실패 복구 경로가 없기 때문(잘못된 알고리즘 이름이 `create_connection`에서 바로 raise되어 요청 전체가 실패) — 대신 실패를 `_ALGORITHM_STATE["error"]`에 기록하고 커널 기본값으로 계속 연결(L276-283).
- `session()`/`reset_session()`: 전역 requests.Session 싱글턴. `reset_session()`은 캡처 시작 전 반드시 호출해야 함(풀링된 소켓이 이미 있으면 tcpdump가 대화 중간부터 기록하게 됨, L392-399).

---

## 2. 역추적 Task 목록

| Task ID | 제목 | 추정 요구사항(코드에서 역추론) | 구현 파일:라인 | 관련 함수/클래스 |
|---|---|---|---|---|
| T-CORE-01 | env var 파싱을 프로젝트 전역 단일화 | "mock/live 오분류가 실제로 발생했던 결함(TRAFFIC_MOCK 파서 불일치)을 재발 불가능하게 만든다" | `config.py` 전체 | `flag`, `flag_any`, `env_int`, `is_mock` |
| T-CORE-02 | 커널이 실제로 로드한 TCP 혼잡제어 알고리즘만 노출 | "UI 드롭다운이 커널에 없는 알고리즘을 보여줘 선택 후 조용히 실패하는 것을 막는다" | `congestion.py:38-58` | `available_algorithms` |
| T-CORE-03 | QUIC 혼잡제어 알고리즘 가용성을 커널 API와 별개로 노출 | "QUIC은 유저스페이스(aioquic)이므로 /proc 개념이 없다 — 별도 판정 경로 필요, 동시에 자체 idle_probe 알고리즘도 드롭다운에 자동 등록" | `quic_congestion.py:25-72` | `available`, `available_algorithms` |
| T-CORE-04 | tc netem 지연 주입 시 BBR pacing이 죽지 않게 함 | "netem을 root에 걸면 fq가 사라져 BBR pacing이 조용히 없어지는 커널 동작을 우회해야 한다" | `netem.py:47-63` | `build_commands` |
| T-CORE-05 | pcap 캡처가 AppArmor로 인한 오진단을 만들지 않게 함 | "~/.something 경로에서 tcpdump가 NET_RAW는 얻었는데 출력 파일에서 Permission denied로 죽어 capability 문제처럼 보이는 실제 사고를 재발 방지" | `capture.py:67-98` | `apparmor_blocks`, `_default_pcap_dir` |
| T-CORE-06 | 캡처 시 pcap이 실제 wire frame을 반영하도록 offload를 껐다 정확히 복원 | "TSO/GSO/GRO가 켜져 있으면 pcap이 64KB 슈퍼패킷을 기록해 slow-start burst(10→20→40) 증거가 사라진다; 측정 종료 후 기기를 원상복구해야 한다" | `offload.py:192-271` | `class Window` |
| T-CORE-07 | 매우 빠른(서브ms) 트래픽에서 tcpdump가 마지막 버스트를 놓치지 않게 함 | "QUIC mock 5턴이 1ms 미만에 끝나는 실측에서 SIGINT가 tcpdump의 read() 전에 도착해 0 packets가 기록된 재현 버그 대응" | `capture.py:498-521` | `Capture.__exit__` |
| T-CORE-08 | pcap 필터가 UDP(QUIC)에서도 실제로 패킷을 기록하게 함 | "shorthand 필터 구문이 'any' 가상 인터페이스에서 BPF 매치는 되지만 파일 기록은 0인 실측 버그 대응" | `capture.py:281-309` | `_filter_expr` |
| T-CORE-09 | 짧은 RTT 경로에서 cwnd 샘플링 주기를 RTT에 맞게 적응 | "고정 2ms 주기는 CDN 엣지(수ms RTT) 기준으로 튜닝되어 있어, 컨테이너 직결/Gateway clean 프로파일의 수백μs RTT 경로에서는 이벤트를 놓친다(DESIGN.md B12)" | `cwnd.py:169-199` | `interval_from_rtt` |
| T-CORE-10 | pcap 타임스탬프의 신뢰도(하드웨어 vs 소프트웨어)를 결과에 명시 | "짧은 RTT 경로에서 gap_ms 신뢰도가 타임스탬프 소스에 의존한다는 것을 리포트 소비자가 판단할 수 있게 한다(DESIGN.md B13)" | `capture.py:335-371` | `timestamp_source` |
| T-CORE-11 | 손실로 인한 window 축소와 idle-reset을 구분해서 카운트 | "네트워크 손실이 많은 환경이 idle-reset 문제로 위장하거나(또는 실제 idle-reset을 숨기는) 것을 방지 — ca_state=open 조건 필수" | `cwnd.py:562-598` | `idle_resets` |
| T-CORE-12 | 새 소켓의 초기 윈도우(idle-window 첫 샘플)를 놓치지 않게 즉시 통지 | "100ms 재발견 타이머로는 3ms 경로에서 cwnd가 10ms 안에 10→60대로 커버려 초기 윈도우를 놓친 실측 사례 대응" | `wire.py:194-227`, `cwnd.py:390-424` | `watch_connections`/`_announce`, `Monitor.announce` |
| T-CORE-13 | 요청한 TCP 혼잡제어 알고리즘이 실제로 적용됐는지 소켓에서 재검증 | "setsockopt가 조용히 무시될 수 있으므로, 요청값을 믿지 않고 getsockopt로 실측값을 다시 읽어 불일치를 노출해야 한다" | `wire.py:242-326` | `set_congestion_algorithm`, `congestion_algorithm_result`, `_CountingConnection._new_conn` |
| T-CORE-14 | 캡처 시작 전 커넥션 풀을 반드시 리셋 | "풀링된 keep-alive 소켓이 이미 열려 있으면 tcpdump가 대화 도중부터 기록해 ACK가 보지 못한 세그먼트에 대한 것으로 나오는 읽을 수 없는 pcap이 생긴다" | `wire.py:391-406` | `reset_session` |
| T-CORE-15 | idle-reset sysctl 토글을 오직 클라이언트(web) 자신에게만 적용 | "2026-09-01 인과 실험이 responding side(서버) 토글은 효과가 미미(+3.7%)했고, next-turn 업로드 지연을 지배하는 건 client의 송신측 cwnd임을 실측으로 확인 — operator가 서버측 프록시 경로(/admin/idle-reset, idle_reset_admin.py 사이드카)를 죽은 코드로 명시적으로 제거 지시" | `idle_reset.py:1-95`, `aipt/web/routes_gateway.py:88-115` | `read`, `write`, `status`, `_web_client_idle_reset_status/_write` |
| T-CORE-16 | idle-reset sysctl 읽기/쓰기가 non-Linux/권한 부족 환경에서 크래시하지 않게 함 | "테스트/개발 환경(비-컨테이너, /proc 없음)이나 CAP_NET_ADMIN 없는 컨테이너에서도 실험 자체는 계속 진행되어야 하며, 실패 원인을 정확히 보고해야 한다" | `idle_reset.py:45-85` | `read`, `write` |
| T-CORE-17 | 요청 body 재전송량을 세션 단위 leaf-hash로 줄이되 옵트인, 완전 무의존 | "local-llm 이미지의 최소 aipt 슬라이스에 통째로 복사되므로 hashlib/re 외 의존성이 있으면 안 되고, 헤더가 없으면 완전히 투명해야 한다" | `cache_protocol.py:1-307` | `encode_body`, `decode_body`, `SessionCache` |
| T-CORE-18 | 캐시 디코드 실패(세션 재시작 등)를 부분 mutation 없이 안전하게 보고 | "재시작된 서버 프로세스가 예전 세션의 hash를 모르는 상황에서, 일부만 복원된 body를 하위로 흘려보내면 안 된다 — 사전 검증 후 raise" | `cache_protocol.py:241-264` | `decode_body`, `CacheMiss` |
| T-CORE-19 | 스트리밍 응답에서 reasoning/thought 텍스트가 TTFT를 왜곡하지 않게 함 | "reasoning 모델이 400ms 생각 후 답하는 경우를 TTFT~0으로 잘못 보고하면 사용자가 실제로는 더 느린 모델을 더 빠르다고 오판하게 된다" | `streaming.py:1-30, 87-115` | `read_stream` |
| T-CORE-20 | TCP_INFO 미지원 플랫폼에서도 실험이 계속되게 함(0 반환) | "비-Linux나 구버전 커널에서 크래시 대신 zero-value로 폴백해 실험 자체는 완주되어야 한다" | `tcpinfo.py:92-124` | `snapshot`, `_zeros` |

---

## 3. Mermaid 다이어그램

```mermaid
flowchart TD
    subgraph ENV["환경변수 (docker-compose.yml / entrypoint)"]
        E1[TRAFFIC_MOCK / *_MOCK]
        E2[NETEM_DELAY_MS / NETEM_IFACE]
        E3[NIC_OFFLOAD_DISABLE / TRAFFIC_PCAP_NO_OFFLOAD]
        E4[TRAFFIC_PCAP_* / TRAFFIC_CWND_*]
    end

    Config[config.py<br/>flag/flag_any/env_int/is_mock] -->|is_mock 판정| Backends
    Config -->|flag_any 재사용| Offload[offload.py]

    E2 --> Netem[netem.py<br/>from_env/apply]
    E3 --> Offload
    E4 --> Capture[capture.py]
    E4 --> Cwnd[cwnd.py]

    subgraph BOOT["컨테이너 부팅(entrypoint, 1회성)"]
        Netem -->|tc qdisc netem+fq| Kernel1[(커널 netdev)]
        Offload -->|ethtool -K 전체| Kernel1
    end

    subgraph RUNTIME["실험 실행 중 (aipt.web -> backends)"]
        Backends[backends/{mock,public_ai,local_llm,quic_mock}] --> Wire[wire.py<br/>session/wire_counter/reset_session]
        Wire -->|watch_connections 통지| Cwnd
        Wire -->|TCP_CONGESTION 설정+검증| Congestion[congestion.py<br/>available_algorithms]
        Backends -->|QUIC 경로| QuicCong[quic_congestion.py]

        Backends --> Capture
        Capture -->|캡처 시작 전 offload 끄고 복원| Offload
        Capture -->|타임스탬프 신뢰도| Capture

        Backends --> Cwnd
        Cwnd -->|native/cwnd_monitor.c netlink| Kernel2[(커널 sock_diag)]
        Cwnd -->|ca_state==open 필터| IdleResets[idle_resets 계산]

        Backends --> Streaming[streaming.py<br/>read_stream]
        Backends --> Tcpinfo[tcpinfo.py<br/>snapshot]
        Streaming -->|req_sent_ms/ttfb_ms 주입받음| Wire
    end

    subgraph CACHE["local_llm 전용, X-AIPT-Cache: enable"]
        GW[backends/local_llm/gateway.py<br/>클라이언트] -->|encode_body| CacheProto[cache_protocol.py]
        CacheProto -->|hash 치환된 body| EngineGW[docker/engine_gateway.py<br/>서버측, 별도 컨테이너]
        EngineGW -->|decode_body / 409 cache_miss| CacheProto
    end

    subgraph IDLERESET["idle-reset: 오직 web 프로세스 자신"]
        WebRoute["aipt/web/routes_gateway.py<br/>GET/POST /api/idle-reset"] -->|in-process, 파라미터 없음=기본 경로| IdleReset[idle_reset.py<br/>read/write/status]
        IdleReset -->|"/proc/sys/net/ipv4/<br/>tcp_slow_start_after_idle"| WebNetns[(web 컨테이너 자신의 netns)]
    end

    Backends -.->|과거엔 존재, 2026-09-02 삭제됨| DeadCode["mock-server /admin/idle-reset,<br/>local-llm docker/idle_reset_admin.py<br/>(현재 코드베이스에 없음)"]

    style IdleReset fill:#ffe0b3
    style DeadCode fill:#eeeeee,stroke-dasharray: 5 5
```

---

## 4. 문서-코드 불일치 (우선순위 높은순)

### ⚠️ 4.1 [최우선/일치처럼 보이나 정밀 확인 필요] idle-reset 적용 방향 — **문서 자체는 실제로 코드와 일치함, 단 다른 문서들이 이를 반영하지 않음**

- **routes_gateway.py 자체 주석 (사실상 1차 문서)**: `"idle-reset (net.ipv4.tcp_slow_start_after_idle) toggle -- ALWAYS on web itself (this process's own /proc/sys, via aipt.core.idle_reset, in-process, no network hop)."` — **코드와 정확히 일치**. `idle_reset.py` 모듈 docstring도 동일 서술.
- **DESIGN.md**: `grep -n "admin/idle-reset|idle_reset_admin|responding|서버 쪽|클라이언트 쪽"` 결과 0건 — **DESIGN.md 본문에는 idle-reset 적용 대상(client-only vs 양쪽)에 대한 서술 자체가 존재하지 않는다.** DESIGN.md는 `tcp_congestion` 프로젝트를 소개하며 "idle 구간 후 TCP cwnd 리셋(slow-start-after-idle)"이라고만 언급(L10)하고, 어느 쪽 소켓에 토글이 걸리는지는 서술하지 않음. → **문서에 없음** 분류.
- **ARCHITECTURE.md**: `idle-reset`/`idle_reset` 관련 서술은 `cwnd.csv`의 `reset_events`/`idle_resets` 카운트를 측정 지표로 언급하는 §6.2 표(L748)뿐이며, 여기도 "idle 구간 후 실제로 슬로우스타트 재진입이 발생하는지"만 서술하고 **토글 대상(client/server)에 대한 언급이 없다.** → **문서에 없음** 분류.
- **결론**: idle_reset.py/routes_gateway.py 자체의 인라인 문서는 코드와 100% 일치하지만(즉 자기 자신을 정확히 설명), 상위 설계 문서(DESIGN.md, ARCHITECTURE.md)는 이 재설계(2026-09-02, client-only로 전환)를 전혀 반영하지 않고 있다 — 즉 "불일치"라기보다 "정보 공백"이다. 다만 리스크는 실재한다: DESIGN.md는 `tcp_congestion`을 "idle 구간 후 TCP cwnd 리셋"이라고만 소개하므로, 이 문서만 읽은 사람은 **어느 소켓이 리셋되는지(그리고 무엇을 토글해야 효과가 있는지)를 전혀 알 수 없고**, 과거 존재했던 responding-side 토글(mock-server `/admin/idle-reset`, local-llm 사이드카)이 삭제됐다는 사실도 DESIGN.md/ARCHITECTURE.md 어디에도 기록되어 있지 않다. 이 audit이 작성되기 전까지는 `docs/experiments/2026-09-01-idle-reset-results.md`(실험 로그, 정식 설계 문서 아님)와 코드 docstring이 유일한 근거였다.
- **권고**: DESIGN.md 또는 ARCHITECTURE.md에 "idle-reset 토글은 client(web) 측에만 적용되며, 과거 존재했던 서버측 admin 라우트/사이드카는 2026-09-02 제거됨"을 명시적으로 기록해야 한다(현재는 실험 결과 문서에만 존재).

### ⚠️ 4.2 [불일치] `offload.py`의 두 API 세트가 DESIGN.md 표에는 단일 파일처럼만 기술됨

- **DESIGN.md L38**: `offload.py | 사실상 같은 기능, env var 네이밍만 다름 (TRAFFIC_PCAP_NO_OFFLOAD vs NIC_OFFLOAD_DISABLE) | 통합 후 두 이름 모두 지원(alias)` — 이 서술은 **병합 전 두 offload.py가 "사실상 같은 기능"이라고 단정**한다.
- **실제 코드**: `offload.py` 자체 docstring(L4-27)에 명시된 대로 두 API는 **기능이 다르다** — capture-time API(`Window`)는 3개 feature(`tso/gso/gro`)만 다루고 "정확히 이전 상태로 복원"하는 데 반해, entrypoint-time API(`build_commands`/`apply`)는 5개 feature(`tso/gso/sg/gro/lro`)를 다루고 "한 번 끄고 그대로 유지"한다(복원 없음). `FEATURES = ("tso","gso","gro")` (L79) vs `ENTRYPOINT_FEATURES = ["tso","gso","sg","gro","lro"]` (L91) — feature 집합 자체가 다르다(`sg`, `lro` 차이).
- **왜 다른가**: DESIGN.md는 병합 *이전* 시점(마이그레이션 계획 단계)의 비교이므로 "env var 네이밍만 다름"이라는 서술은 병합 착수 근거로는 맞을 수 있으나, 실제 병합 결과물인 `offload.py`는 두 개의 서로 다른 API/feature-set을 **공존**시키는 형태로 구현되었고 이는 DESIGN.md가 예고한 "통합"보다 더 복잡하다. MIGRATION.md L14에서도 `"env alias 양쪽 지원"`이라고만 적혀 있어 feature-set 차이는 어느 문서에도 명시되지 않았다.

### ✅ 4.3 [일치] `cwnd.py`/`capture.py` 병합 방침

- DESIGN.md L36-37("tcp_congestion의 단순화된 인터페이스를 채택하되 token_traffic의 상세 docstring/dumps/exact_queries 계측 필드 병합", "token_traffic의 AppArmor 감지 로직 보존")과 실제 `cwnd.py`(`dumps`/`exact_queries`가 `result()`에 존재, L540-541), `capture.py`(`apparmor_blocks` 존재, L67-98)가 정확히 일치한다.

### ✅ 4.4 [일치] B12(적응형 cwnd 샘플링), B13(타임스탬프 소스)

- DESIGN.md L521-526의 B12/B13 작업 항목이 `cwnd.py`의 `interval_from_rtt`(L169-199)와 `capture.py`의 `timestamp_source`(L335-371)로 정확히 구현되어 있으며, 코드 주석도 "B12"/"B13"을 명시적으로 참조한다.

### ✅ 4.5 [일치] cache_protocol 컴포넌트 배치

- ARCHITECTURE.md §3.3(L428-436)의 클라이언트/서버 역할 분담표(`aipt/backends/local_llm/gateway.py`가 클라이언트측 encode, `docker/engine_gateway.py`가 서버측 decode/409)가 `cache_protocol.py`의 `encode_body`/`decode_body`/`CacheMiss` 설계와 정확히 일치한다.

### 📄 4.6 [문서에 없음] `quic_congestion.py`의 `idle_probe` 자동 등록 side-effect

- `available_algorithms()`(L61-67)이 `aipt.backends.quic_mock.congestion`을 import해 `idle_probe` 알고리즘을 부작용으로 등록한다는 사실은 코드 docstring에만 있고, DESIGN.md L797-816(quic_congestion.py 신설을 다루는 절)에는 드롭다운이 `["cubic","idle_probe","reno"]`가 된다는 **결과**만 언급될 뿐, "import 부작용으로 등록된다"는 **메커니즘**은 서술되어 있지 않다. 우선순위는 낮음(동작 자체는 일치, 서술 상세도만 차이).

### 📄 4.7 [문서에 없음] `config.py`의 `env_int` vs `netem.parse_delay` 정책 차이

- `config.py`의 `env_int`(실패 시 default로 조용히 폴백)와 `netem.py`의 `parse_delay`(실패 시 raise)가 의도적으로 다른 정책을 갖는다는 것이 `config.py` 자체 docstring(L55-60)에는 있지만, DESIGN.md/ARCHITECTURE.md 어디에도 이 두 정책이 왜 다른지 서술되어 있지 않다. 실무 영향은 낮으나, 향후 새 env var 파싱 함수를 추가할 때 어느 정책을 따를지 판단 근거가 코드 주석에만 있다.
