# backends 모듈 감사

대상: `aipt/backends/{base,__init__,record}.py`, `public_ai/*`, `mock/*`,
`local_llm/*`, `quic_mock/*`, 및 `tests/backends/**`.
방법: 코드 전량 선독 → 역추적 Task 카드 작성 → Mermaid 다이어그램 → 마지막에
DESIGN.md/ARCHITECTURE.md/MIGRATION.md와 grep 대조. 코드는 수정하지 않았다.

---

## 1. 구현 현황

### 1.1 `base.py` — Backend 프로토콜 (인터페이스)

`aipt/backends/base.py:58-136`에 `runtime_checkable Protocol` `Backend` 정의.
구조적(덕타이핑) 계약이며 명시적 상속을 요구하지 않는다(`base.py:63-67`).

- 클래스 속성: `NAME`, `DEFAULT_MODEL`, `ARMS`, `HEADLINE_ARMS`, `transport`
  (`base.py:70-88`).
- 메서드: `ready() -> (bool, str)` (`base.py:90-93`),
  `api_host() -> str` (`base.py:95-103`),
  `connect(arm, model, system) -> None` (`base.py:105-114`),
  `send_turn(turn, question, measure, on_progress=None) -> TurnExchange`
  (`base.py:116-128`),
  `close() -> None` (`base.py:130-136`).
- `Transport = Literal["http1", "http3"]`, `DEFAULT_TRANSPORT = "http1"`
  (`base.py:53-55`) — "http3"은 슬롯일 뿐, base.py 자체는 아무 분기도
  구현하지 않는다(`base.py:48-52`의 주석이 명시).
- `progress()` 헬퍼(`base.py:139-165`)가 모든 backend에서 공통으로 쓰는
  진행 이벤트 emit 함수. `phase`가 `"steady"`/`"setup"`/`"teardown"` 등으로
  클라이언트의 캡처 윈도우 개폐를 제어한다는 계약이 docstring에 명시.

### 1.2 `record.py` — 공통 턴 레코드 스키마

- `TurnExchange` Protocol(`record.py:43-67`) — 모든 backend의 `send_turn()`
  반환값이 만족해야 하는 필드 집합 (`wire_sent/recv`, `req_payload_bytes`,
  `resp_payload_bytes`, 5개 타임마크, `text`, `request_json`,
  `response_json`, `error`).
- `Exchange` dataclass(`record.py:70-98`) — mock/local_llm이 실제로 사용하는
  구체 구현. `cache_bytes_saved` 필드(`record.py:94-98`)는 local_llm 전용,
  기본 0.
- `turn_record()`(`record.py:114-195`) — 4개 backend(public_ai 내부는
  gemini/openai 두 엔진)가 공통으로 거치는 단일 레코드 조립 함수. `backend`
  키에 실제로는 `"public_ai"`/`"mock"`이 들어가고 `quic_mock`도
  `NAME = "mock"`을 자칭한다(§1.6 참고).

### 1.3 `__init__.py` — 레지스트리

`_KNOWN = ("public_ai", "mock", "local_llm")` (`__init__.py:28`) — **3개만
등록**. `get(name)`은 `_KNOWN`에 없는 이름을 `KeyError`로 거부하고
(`__init__.py:41-42`), 있으면 `importlib.import_module(f"{__package__}.{name}")`로
동적 임포트한다(`__init__.py:43`). `quic_mock`은 이 레지스트리에 전혀
등록되어 있지 않다 — `aipt.backends.get("quic_mock")`은 `KeyError`.

### 1.4 `public_ai/` — Gemini/OpenAI, 실제 인터넷 경유

- **`__init__.py`**: `PublicAIBackend` 파사드(`public_ai/__init__.py:52-127`).
  gemini/openai 두 엔진을 하나의 레지스트리 슬롯으로 묶는다. `_select()`
  (`:85-92`)가 `arm` 이름으로 엔진을 자동 판별(`_engine_for_arm`,
  `:32-40`)하거나 생성자 `engine=` 인자로 고정. `connect`/`send_turn`/
  `close`는 전부 내부 `self._backend`(GeminiBackend 또는 OpenAIBackend
  인스턴스)로 위임(`:115-126`).
- **`_call.py`**: `send()`(`_call.py:223-251`)가 `measure` 값(`bytes`/
  `latency`/`both`)에 따라 blocking POST(`_blocking`, `:119-158`) 또는
  streamed POST(`_streamed`, `:161-220`)를 수행하거나 둘 다 실행 후
  `_merge()`(`:254-279`)로 합친다. `wire.wire_counter()`로 소켓 바이트를
  실측(`_call.py:129,174`). `aipt.core.wire`/`aipt.core.streaming` 사용
  (`_call.py:60-61`) — HTTP/1.1(requests 세션) 위에서 동작, TLS는
  `wire.session()`이 담당.
- **`_cachebust.py`**: 두 벤더의 암묵적 프리픽스 캐싱을 오염시키지 않기
  위해 런/턴 단위 마커를 시스템 프롬프트 앞에 붙이는 유틸
  (`_cachebust.py:1-107`). `TRAFFIC_CACHE_BUST` env로 on/off.
- **`gemini.py`**: `GeminiBackend`(`gemini.py:616-763`)가 `connect`/
  `send_turn`/`close` 구현. 6개 arm(`stateless`, `nocontext`, `cached`,
  `interaction`, `interaction_inline`, `interaction_stateless`,
  `gemini.py:67-68`)이 `send_turn`에서 `_send_turn_*` 메서드로 분기
  (`gemini.py:668-691`). `cached` arm은 원본 2-패스 알고리즘을 턴 단위
  프로토콜에 맞게 **온라인 캐싱**으로 재설계했다고 모듈 docstring에 명시
  (`gemini.py:37-49`) — turn 1은 캐시 없이 보내고, 매 턴 종료 후 그 시점까지의
  트랜스크립트로 캐시를 (재)생성해 다음 턴이 참조(`_send_turn_cached`,
  `gemini.py:716-737`). 원본 2-패스 `_arm_cached`(`gemini.py:495-` 부근)는
  parity 테스트용으로 그대로 남아 있다(`run_arm`, `gemini.py:589-611`).
  `is_mock()`이 참이면(`gemini.py:119-123`, `TRAFFIC_MOCK`/`GEMINI_MOCK`)
  실제 API를 전혀 호출하지 않고 `_mock_generate`/`_mock_interaction`
  (`:330-366`)이 합성 응답을 만든다.
- **`openai.py`**: `OpenAIBackend`(`openai.py:473-567`), 4개 arm
  (`chat_stateless`, `responses_stateless`, `responses`,
  `responses_inline`, `openai.py:65`). `responses_inline`은 `connect()`
  안에서 conversation-create 호출을 수행하고 그 결과 레코드를
  `pending_setup_records`에 적재(`openai.py:517-520,495-499`) — Backend
  프로토콜의 `connect()`가 반환값을 가질 수 없기 때문이라고 클래스
  docstring이 명시(`openai.py:474-480`). `mock_mode()`가 참이면
  `_mock_send`(`openai.py:435-468`)가 네트워크 없이 응답을 합성.
- **`recorder.py`**: `record_turn()`/`RecordWriter`/`recording_backend()`
  (`recorder.py:156-287`). 실제(비-mock) 호출의 request/response 원문을
  `mask_secrets()`(`:79-97`, 헤더/바디 재귀 마스킹)로 정제한 뒤
  `RecordedTurn.to_dict()`로 JSON화. `recording_backend()`는 기존 Backend
  인스턴스를 감싸 `send_turn`마다 자동으로 기록하는 프록시
  (`recorder.py:238-287`) — 기존 Gemini/OpenAIBackend 코드는 전혀 몰라도
  됨(옵트인, additive).

### 1.5 `mock/` — 고정/재생 JSON I/O, HTTP/1.1 TCP keep-alive

- **`server.py`**: `Server(socketserver.ThreadingTCPServer)`
  (`server.py:188-206`), stdlib만 사용하는 HTTP/1.1 keep-alive 서버.
  `/ping`, `/health`, `/inference-mock` 3개 엔드포인트(`server.py:11-13`).
  `record`가 바인딩되어 있고 유효한 `turn=<i>` 쿼리가 오면 그 턴의 실제
  답변 텍스트를 반환(`_record_answer`, `server.py:77-96`), 없으면 순수
  byte-dummy(`_pad_json_to_size`, `:54-74`)로 폴백.
- **`records.py`**: `ScenarioRecord`/`Turn` dataclass(`records.py:55-93`).
  `turns`-shaped(네이티브 Mock)과 `steps`-shaped(public_ai 레코더 스키마)
  양쪽을 모두 로드(`load_scenario_record`, `:152-178`). `byte_size_scenario()`
  (`:205-232`)는 원본 tcp_congestion의 순수 byte-sweep 모드를 옵션으로 보존.
- **`replay.py`**: `from_capture_doc`/`from_capture_file`/
  `from_public_ai_record_doc`(`replay.py:66-123`) — public_ai 레코더가
  캡처한 실제 트래픽을 Mock이 재생할 수 있도록 변환하되, **답변 텍스트는
  동일 길이 placeholder(`"x" * len`)로 치환**(`_placeholder`, `:44-47`),
  질문 텍스트만 원문 유지. 지연시간은 재현하지 않는다(모듈 docstring,
  `replay.py:8-16`) — `inference_delay_ms`가 별도 지연 제어 knob.
- **`conversation.py`**: `MockBackend`(`conversation.py:281-616`)가
  `connect`/`send_turn`/`close` 구현. `connect()`가 서버 스레드 기동(또는
  `MOCK_SERVER_HOST`/`MOCK_SERVER_PORT` env로 외부
  `mock-server` 컨테이너를 대신 타깃팅, `resolve_target`,
  `conversation.py:398-426`) + keep-alive 소켓 오픈 + `aipt.core.cwnd.Monitor`
  시작(`:455-465`). `_request_body_text()`(`:467-517`)가 시스템 프롬프트를
  매 턴 재전송하고 이전 (question, answer) 히스토리를 누적해 실제
  stateless 멀티턴 클라이언트의 누적 업로드 성장을 재현 — 이 로직은
  2026-08-31 실측 버그 수정으로 추가됐다고 docstring이 밝힘
  (`:486-500`). `set_congestion_algorithm()`(`:50-56`)로 TCP_CONGESTION
  소켓옵션을 connect 전에 직접 세팅.
- **`probe.py`**: idle 구간 RTT 측정용 HTTP `/ping` (`probe.py:21-29`).
  `delivery_rate`는 의도적으로 제외(작은 probe payload가 왜곡시키므로,
  `probe.py:8-11`).
- **`__init__.py`**: `MockBackend`만 재노출(`mock/__init__.py:18-24`).

### 1.6 `local_llm/` — 표준 OpenAI 호환 서빙 엔진 + 자체 애플리케이션 게이트웨이

- **`engine_adapter.py`**: `EngineAdapter`(`engine_adapter.py:109-215`)는
  추론을 재구현하지 않는 **얇은 HTTP 클라이언트** — llama.cpp/vLLM 둘 다
  동일한 `POST /v1/chat/completions` 스키마를 쓴다는 전제(모듈 docstring,
  `:1-13`). `LOCAL_LLM_ENGINE_URL`(기본 `http://127.0.0.1:40080`,
  `:65`)을 외부에서 이미 기동된 엔진으로 가정하고 붙기만 한다 — 엔진
  프로세스 자체를 띄우는 코드는 없음(`:15-19`).
- **`gateway.py`**: `Gateway`(`gateway.py:81-291`) — engine_adapter와
  클라이언트 사이의 **애플리케이션 레벨(L7) 프록시**. 실제 소켓 리스너는
  없음(in-process 레이어, `:87-94`). `on_request`/`on_response` 훅
  등록점(`:122-147`)만 마련하고 실제 실험 로직은 미구현(B4/B5 scope 그대로,
  모듈 docstring 확인). 유일하게 실제로 동작하는 "신기능"은
  `Backend.transport` 값을 `X-AIPT-Transport` 헤더로 반영하는 것
  (`:37-43,174-182`) — 이것도 실제 QUIC/HTTP3 전송은 아니고 그냥 HTTP/1.1
  요청에 표식 헤더 하나 추가하는 것. **request-body leaf-hash 캐싱**
  (`docs/engine_gateway_caching_seed.md`)이 이 위에 구현되어 있음
  (`gateway.py:202-245`, `cache_protocol.encode_body`/409 재시도 로직).
- **`__init__.py`**: `LocalLLMBackend`(`local_llm/__init__.py:75-256`).
  `connect()`가 `EngineAdapter` + `Gateway` 인스턴스를 만들고
  (`:175-185`), `aipt.core.wire.watch_connections()` 훅으로 실제 소켓이
  열리는 순간 `cwnd.Monitor.announce()`를 호출(`:192-197`) — Mock처럼
  자체 서버 소켓을 갖지 않고 `aipt.core.wire`의 풀드 세션이 지연 오픈하는
  소켓을 관찰하는 방식. `ARMS = ("chat",)` 단 하나(`:56`) — 표준화된
  서버측 세션/캐시 개념이 없어 다중 arm을 둘 근거가 없다는 설명
  (`:49-55`). `close()`가 감시 구독을 해제하고 cwnd 결과를 확정
  (`:238-248`).

**DESIGN.md 4.7의 "Network Gateway"(L3/L4, `aipt/gateway/`, `tc netem`)와
`local_llm/gateway.py`의 "engine gateway"(L7 애플리케이션 프록시)는 서로
다른 컴포넌트** — 코드상으로도 `local_llm/gateway.py`는 소켓을 열지 않고
`aipt.core.wire`의 세션을 그대로 쓰므로(위 확인), Network Gateway 컨테이너를
거치는지 여부는 이 모듈이 알지도 관여하지도 않는다(모듈 docstring,
`gateway.py:1-25`).

### 1.7 `quic_mock/` — 실제 QUIC(UDP/aioquic), Mock 전용, **레지스트리 미등록**

- 결론부터: **`quic_mock`은 아이디어 수준 시뮬레이션이 아니라 실제
  QUIC 프로토콜 스택**이다. `aioquic`(RFC 9000 QUIC 구현) 위에서
  `quic_connect`/`quic_serve`(`backend.py:68-70`)로 진짜 UDP 소켓과 TLS 1.3
  핸드셰이크(`_ensure_cert()`, `backend.py:105-125`)를 수행하며,
  `QuicMockBackend`(`backend.py:235-685`)가 `aipt.backends.base.Backend`
  프로토콜을 만족하는 실제 구현이다(`connect`/`send_turn`/`close`,
  `ready`/`api_host` 전부 존재). "mock"이라는 이름은 (a) 통신 상대가
  echo/dummy 응답 서버이고, (b) `NAME = "mock"`, `ARMS = ("dummy",
  "record")`로 HTTP/1.1 MockBackend와 동일한 UI 슬롯("Mock" 카드의
  Transport=http3 선택지)을 공유하기 때문이다(`backend.py:245-248`,
  클래스 docstring 240-243) — quic_mock은 별개 backend 이름이 아니다.
- **레지스트리 미등록**: `aipt/backends/__init__.py`의 `_KNOWN`에
  `quic_mock`은 없다(§1.3). `aipt.backends.get("quic_mock")`은
  `KeyError`. 대신 `aipt/web/routes_run.py:338-342`가
  `from aipt.backends.quic_mock.backend import QuicMockBackend`를
  **직접 import**해서 `req.backend == "mock" and transport == "http3"`일
  때만 생성한다(`routes_run.py` grep 결과) — `aipt.backends.get()`
  간접 조회 경로를 우회하는 특수 케이스.
- **congestion.py**: `IdleProbeCongestionControl`
  (`congestion.py:48-141`)이 aioquic의 표준
  `register_congestion_control("idle_probe", ...)` 레지스트리
  (`congestion.py:143`)에 실제로 등록되는 커스텀 혼잡제어 알고리즘.
  `mark_idle_probe_sent()`로 idle 진입 시 pre-idle RTT를 기록하고,
  다음 `on_rtt_measurement()`(PING ACK 콜백)에서 RTT 증가율에 비례해
  cwnd를 사전에 줄인다(`:92-129`). Reno 위임 + 오버레이 방식
  (`:56-90`) — kernel/C 코드 없이 순수 Python 클래스.
- **backend.py**: 서버측 `_MockEchoProtocol`(`:128-199`) / 클라이언트측
  `_MockClientProtocol`(`:202-232`). 프레이밍: `struct.pack(">II",
  response_bytes, delay_ms) + question_text`
  (`backend.py:549-552`), 4바이트 레거시(no-delay) 프레임도 하위호환
  (`:171-176`). `_cwnd_sample_loop()`(`:438-509`)이 aioquic의
  `QuicConnection._loss`(userspace 혼잡 상태)를 20ms 간격으로 폴링해
  cwnd 연속 트레이스를 만든다 — **`aipt.core.cwnd`의 netlink 기반
  모니터는 QUIC을 관측할 수 없다**(UDP라 소켓 inode가 없음)는 이유가
  모듈 docstring(`:26-42`)과 `cwnd_result()`(`:620-682`)에 명시. 이는
  local_llm/mock(TCP)과 quic_mock(UDP/userspace CC)이 cwnd 계측
  메커니즘 자체가 다르다는 뜻 — 후자는 `"measurement_confidence":
  "degraded"`를 스스로 보고한다(`:647`).
- **server.py**: 별도 최소 QUIC echo 서버(`EchoProtocol`,
  `server.py:25-33`) — spike 원형, `quic-mock-server` 컨테이너에서는 대신
  `backend.py`의 `_MockEchoProtocol`을 사용(`server.py:42-49`의
  `create_protocol` 설명).
- **congestion/experiment/spike_runner.py**: `spike_runner.py`/
  `experiment.py`는 독립 CLI(`python -m aipt.backends.quic_mock.spike_runner`
  등)로, `aipt/gateway/`(Network Gateway, tc netem)를 실제로 통과시켜
  idle_probe vs reno의 cwnd 궤적(spike_runner) 및 처리량/지연
  (experiment)을 A/B 측정한다. **`Backend` 프로토콜과 무관한 별도
  측정 도구** — `QuicMockBackend`(backend.py)와 달리 `send_turn()` 같은
  라이프사이클 메서드가 없다.

**동작 확인 요약**: quic_mock은 실제 QUIC(UDP+TLS1.3+aioquic congestion
control)이며, TCP HTTP/1.1과는 다른 L4/L7 스택을 진짜로 사용한다. "simulated"가
아니라 "real QUIC, mock content"(서버가 답변을 생성하지 않고 echo/canned
answer만 반환한다는 의미의 mock)다.

---

## 2. 역추적 Task 목록

| Task ID (추정) | 제목 | 추정 요구사항 | 구현 파일:라인 | 관련 함수/클래스 |
|---|---|---|---|---|
| A1 | Backend 프로토콜 정의 | 3개 backend가 공유할 connect/send_turn/close 최소 계약. token_traffic Provider Protocol 일반화 (DESIGN.md §5 A1 자기서술) | `aipt/backends/base.py:58-136` | `Backend` (Protocol), `progress()` |
| A2 | Gemini/OpenAI 어댑터를 Backend 프로토콜로 이관 | run_arm(전체 대화 일괄) → connect/send_turn/close 분할, 캐시 아키텍처 재설계 필요 | `aipt/backends/public_ai/gemini.py:616-763`, `openai.py:473-567` | `GeminiBackend`, `OpenAIBackend`, `run_arm()`(parity 보존) |
| A3 | Mock HTTP/1.1 서버 이관 + fixture 답변 서빙 확장 | tcp_congestion 서버 원형 유지, byte-dummy 외에 실제 Q&A 답변 재생 추가 | `aipt/backends/mock/server.py:1-206` | `Server`, `_Handler._handle_inference`, `_record_answer` |
| B1 | Mock fixture(ScenarioRecord) 포맷 통합 로더 | byte-sweep 전용 → Q&A JSON 스키마 지원, public_ai 레코더 스키마(`steps`)도 겸용 | `aipt/backends/mock/records.py:55-232` | `ScenarioRecord`, `load_scenario_record`, `byte_size_scenario` |
| B2 | Public AI 실측 트래픽 recorder + 민감정보 마스킹 | 실제 API 호출 원문을 디스크 저장하되 API 키가 절대 저장되지 않도록 강제 | `aipt/backends/public_ai/recorder.py:73-287` | `mask_secrets`, `record_turn`, `RecordWriter`, `recording_backend` |
| B3 | Mock replay(바이트 패턴만, 텍스트 미보존) | recorder가 캡처한 실측 트래픽을 Mock이 재생하되 실제 모델 답변 내용은 재사용하지 않음(이 실행이 만든 게 아니므로 오해 소지) | `aipt/backends/mock/replay.py:44-123` | `from_capture_doc`, `from_public_ai_record_doc`, `_placeholder` |
| B4 | LocalLLMBackend: 표준 서빙 엔진 + 자체 게이트웨이 | 추론 엔진 재구현 금지, OpenAI 호환 API로만 연동, HTTP 신기능 실험 훅 자리 마련 | `aipt/backends/local_llm/engine_adapter.py:109-215`, `gateway.py:81-291`, `__init__.py:75-256` | `EngineAdapter`, `Gateway`, `LocalLLMBackend` |
| B5 | Transport 확장 슬롯(QUIC 자리만, base.py) | http3 구현 없이 필드만 예약 — 후속 프로젝트 분리 | `aipt/backends/base.py:48-55` | `Transport`, `DEFAULT_TRANSPORT` |
| (§7 후속) | QUIC idle-probe 혼잡제어 스파이크 | TCP tcp_congestion_ops는 능동 idle probe를 구조적으로 지원 못 함(커널 조사 결론) → QUIC의 공개 send_ping API + pluggable CC 레지스트리로 우회 | `aipt/backends/quic_mock/congestion.py:48-145` | `IdleProbeCongestionControl`, `register_congestion_control("idle_probe")` |
| (§7 후속) | QuicMockBackend: Backend 프로토콜 편입 | 스파이크(spike_runner/experiment, 독립 CLI)를 웹 UI "Mock" 카드의 transport=http3 옵션으로 승격 | `aipt/backends/quic_mock/backend.py:235-685` | `QuicMockBackend`, `_MockEchoProtocol`, `_MockClientProtocol`, `_cwnd_sample_loop` |
| (미등록/버그성 격차) | quic_mock을 레지스트리에 정식 등록하지 않은 채 web 라우트에서 특수 케이스로 우회 | ARMS/이름이 "mock"과 겹치는 4번째 backend를 `_KNOWN`에 넣지 않고 `routes_run.py`가 직접 import — DESIGN.md 스스로 "번호 없는 4번째 backend 후보"라 인정(§4 diff 주석) | `aipt/backends/__init__.py:28`, `aipt/web/routes_run.py:338-342` | `_KNOWN`, `get()`, `_build_backend()`(routes_run.py) |
| (엔진 게이트웨이 캐싱) | request-body leaf-hash 중복 제거 캐싱 | transport 슬롯을 실제로 활용한 첫 신기능 실험(§7 열린 항목에 대한 응답), 멀티턴 대화의 반복 전송 바이트 절감 | `aipt/backends/local_llm/gateway.py:202-245`, `aipt/backends/record.py:94-98,185` | `Gateway.send()`, `cache_protocol.encode_body`, `Exchange.cache_bytes_saved` |

---

## 3. Mermaid 다이어그램

```mermaid
flowchart TB
    subgraph CLIENT["client 코드 (aipt.web/routes_run.py, aipt.core 계측)"]
        RR["_build_backend() / connect→send_turn*→close 드라이버"]
    end

    subgraph PROTO["aipt/backends/base.py — Backend Protocol (구조적, 상속 불요)"]
        BP["Backend\nNAME/DEFAULT_MODEL/ARMS/HEADLINE_ARMS/transport\nready() api_host()\nconnect() send_turn() close()"]
        REC["record.py: TurnExchange Protocol\nturn_record()"]
    end

    subgraph REG["aipt/backends/__init__.py — 레지스트리 (_KNOWN)"]
        GET["get(name) / names()\n_KNOWN = (public_ai, mock, local_llm)\n※ quic_mock 미등록"]
    end

    subgraph PUBLICAI["public_ai/ — 실제 인터넷, HTTP/1.1 (requests)"]
        PAF["PublicAIBackend (파사드)"]
        GB["GeminiBackend\n(6 arms, 온라인 캐싱 재설계)"]
        OB["OpenAIBackend\n(4 arms)"]
        CALL["_call.py: send()\nblocking/streamed/both"]
        CB["_cachebust.py"]
        RECDR["recorder.py\nmask_secrets → RecordedTurn"]
        PAF --> GB
        PAF --> OB
        GB --> CALL
        OB --> CALL
        GB --> CB
        OB --> CB
        RECDR -. wraps .-> GB
        RECDR -. wraps .-> OB
    end

    subgraph MOCK["mock/ — HTTP/1.1 TCP keep-alive, in-process 또는 mock-server 컨테이너"]
        MB["MockBackend"]
        MSV["server.py: Server\n(ThreadingTCPServer)"]
        MREC["records.py: ScenarioRecord"]
        MREPLAY["replay.py\n(byte pattern only)"]
        MPROBE["probe.py (idle RTT)"]
        MB --> MSV
        MB --> MREC
        MREPLAY --> MREC
        MB --> MPROBE
    end

    subgraph LOCALLLM["local_llm/ — OpenAI-compatible HTTP over aipt.core.wire"]
        LLB["LocalLLMBackend"]
        EA["EngineAdapter\n(얇은 클라이언트, 추론 미구현)"]
        GW["Gateway (L7 애플리케이션 프록시)\nX-AIPT-Transport 헤더\nleaf-hash 캐싱"]
        LLB --> GW
        GW --> EA
    end

    subgraph QUICMOCK["quic_mock/ — 실제 QUIC(UDP/aioquic/TLS1.3), NAME='mock' 재사용, 레지스트리 미등록"]
        QMB["QuicMockBackend\n(Backend Protocol 구현)"]
        IPC["congestion.py\nIdleProbeCongestionControl\n(aioquic register_congestion_control)"]
        QSRV["backend.py: _MockEchoProtocol\n(server) / _MockClientProtocol"]
        SPIKE["spike_runner.py / experiment.py\n(독립 CLI, Backend 프로토콜 미구현)"]
        QMB --> IPC
        QMB --> QSRV
        SPIKE -. shares congestion.py .-> IPC
    end

    RR -->|"aipt.backends.get(name)"| GET
    GET -->|"import_module"| PUBLICAI
    GET -->|"import_module"| MOCK
    GET -->|"import_module"| LOCALLLM
    RR -.->|"직접 import, backend=mock&transport=http3일 때만\n(get() 우회)"| QUICMOCK

    PROTO -.->|"구조적 계약(Protocol), 상속 없음"| PAF
    PROTO -.-> MB
    PROTO -.-> LLB
    PROTO -.-> QMB
    REC -.->|"turn_record() 공통 스키마"| PAF
    REC -.-> MB
    REC -.-> LLB
    REC -.-> QMB

    PUBLICAI -->|"HTTPS, 실제 인터넷\n(Network Gateway 미경유)"| EXT["generativelanguage.googleapis.com\napi.openai.com"]
    MOCK -->|"TCP, Gateway 경유 가능\n(MOCK_SERVER_HOST/PORT)"| GW3["aipt/gateway (L3 netem, 선택)"]
    LOCALLLM -->|"TCP, Gateway 경유 가능"| GW3
    QUICMOCK -->|"UDP/QUIC, Gateway 경유 가능\n(QUIC_MOCK_SERVER_HOST/PORT)"| GW3
    GW3 --> ENGINE["mock-server / local-llm(엔진) /\nquic-mock-server 컨테이너"]
```

---

## 4. 문서-코드 불일치

DESIGN.md / ARCHITECTURE.md / MIGRATION.md에서 `backends`/`public_ai`/`mock`/
`local_llm`/`quic_mock` 관련 절을 grep(약 42+170+272건 매치)한 뒤 코드와
대조한 결과.

### 4.1 불일치 (최우선)

**(1) ARCHITECTURE.md §1.1 아키텍처 다이어그램에 quic_mock이 없음 — 현재도
유효한 격차.**

- **문서 인용** (`ARCHITECTURE.md:34-40`):
  ```
  subgraph BACKENDS["aipt/backends — Backend 프로토콜 (컴포넌트 ①②③)"]
      PublicAI["① PublicAI 연동"]
      Mock["② Mock 트래픽"]
      LocalLLM["③ LocalLLM 연동"]
      PublicAI ~~~ Mock ~~~ LocalLLM
  end
  ```
  3개 backend만 등장 — quic_mock 서브그래프 없음.
- **코드 인용**: `aipt/web/routes_run.py:338-342`가
  `from aipt.backends.quic_mock.backend import QuicMockBackend`를 실제로
  import해 `backend="mock" & transport="http3"` 선택 시 생성하고,
  `aipt/web/static/app.js:36-61`과 `routes_config.py:389-395`가 이를 위한
  UI 토글(`quic_available`/`quic_congestion_algorithms`)까지 구현되어
  있다. `docker-compose.yml`에 `quic-mock-server` 5번째 서비스도 실존한다
  (DESIGN.md 자체 감사 §5.2 인용, 아래 (2) 참고).
- **판정**: 불일치. `ARCHITECTURE.md`는 스스로 "이 문서는 지금 이 코드가
  어떻게 동작하는가를 보여주는 참조 문서"(`ARCHITECTURE.md:5`)라고
  선언하는데, 정작 4번째로 실제 동작하는 backend 통합(quic_mock)이
  아키텍처 다이어그램에서 빠져 있다.
- **비고**: DESIGN.md §5.2(`DESIGN.md:587-591`)가 이미 이 정확한 문제를
  자체 인지하고 "남은 괴리" 항목으로 기록해 두었다("§4.8 아키텍처
  다이어그램이 quic_mock 미반영") — 다만 그 시점 이후 실제로 다이어그램이
  갱신되었는지는 이번 코드 감사로 재확인한 결과 **미수정 상태로 남아
  있음**(ARCHITECTURE.md 최신 버전 기준).

**(2) quic_mock이 `aipt.backends` 레지스트리(`_KNOWN`)에 등록되어 있지
않음 — 문서는 "4번째 backend 후보"라 표현하지만 이는 등록 여부에 대한
정확한 서술이고, 오히려 __init__.py 자체 docstring/모듈 구조가 "3개
backend"라는 서사를 유지하는 것과 대조된다.**

- **문서 인용** (`aipt/backends/__init__.py:1-17`, docstring):
  ```
  "aipt.backends -- the 3-backend common client structure (DESIGN.md 4.5)."
  ... talks to exactly one of three backends ...
    * public_ai ...
    * mock ...
    * local_llm ...
  ```
- **코드 인용**: `routes_run.py`가 `QuicMockBackend`를 실제 런타임 경로에서
  `get()`을 거치지 않고 직접 import해 사용(§1.7). `_KNOWN`
  (`__init__.py:28`)은 여전히 3개.
- **판정**: 불일치(모듈 자체 docstring vs 실제 동작하는 backend 개수).
  기능적으로는 동작하지만, "backend는 `aipt.backends.get()`을 통해서만
  조회한다"는 이 패키지 자신의 설계 원칙(`__init__.py:13-16`, "a caller
  must never be able to import an arbitrary module by passing a string
  straight to importlib")을 `routes_run.py`가 quic_mock에 한해 우회하고
  있다 — 원칙과 실제 구현이 어긋난 지점.

**(3) Network Gateway(L3, `aipt/gateway/`)와 engine Gateway(L7,
`local_llm/gateway.py`)의 이름 충돌은 문서·코드 양쪽에서 이미 명시적으로
경고되어 실제로는 일치.**

- **문서 인용** (`ARCHITECTURE.md:230-231`):
  ```
  ⚠️ 이 "engine gateway"(애플리케이션 레벨 프록시)와 아래 ④ "Network
  Gateway"(L3 커널 레벨)는 이름이 비슷하지만 서로 다른 컴포넌트.
  ```
- **코드 인용** (`local_llm/gateway.py:1-25`, 모듈 docstring)이 동일한
  경고를 코드 레벨에서도 반복.
- **판정**: 일치 (허위 불일치 후보를 배제하기 위해 기재). 명명 충돌
  자체는 실재하지만 문서와 코드 둘 다 명확히 경고하고 있어 "괴리"로
  분류하지 않는다.

### 4.2 문서에 없음(코드에만 존재)

- **`quic_mock/experiment.py`, `quic_mock/spike_runner.py`의 정확한 CLI
  사용법과 A/B 측정 방법론**은 DESIGN.md §7/7.1에 상세히 서술되어 있어
  "문서에 없음"이 아니라 오히려 코드보다 문서가 더 상세하다(설계
  문서 성격상 정상). 다만 **`ARCHITECTURE.md`(구현 완료 후 참조 문서)에는
  quic_mock 관련 섹션 자체가 전혀 없다** — §1.2 폴더 구조표
  (`ARCHITECTURE.md:104-176`)에는 `quic_mock/`이 등재되어 있지만
  (`:144-147`), 뒤따르는 §2 이후의 backend별 상세 설명 섹션(§2.1 public_ai,
  §2.2 mock, §2.3 local_llm 추정)에 quic_mock 전용 절이 없다(§4.1 (1)의
  다이어그램 누락과 같은 근본 원인).
- **`mock/probe.py`의 idle RTT probe 메커니즘**은 DESIGN.md에 A3 이관
  항목으로만 한 줄 언급되고(`DESIGN.md:239` 부근 표), `run_probes()`가
  `delivery_rate`를 의도적으로 배제하는 이유(작은 payload가 BW 추정을
  왜곡)는 코드 docstring(`probe.py:8-11`)에만 있고 DESIGN.md 본문에는
  이 근거가 재서술되어 있지 않다 — 사소하지만 문서에 없음으로 분류.
- **`public_ai/gemini.py`의 `cached` arm 온라인 캐싱 재설계 상세**
  (turn 1은 무캐시, 매 턴 후 캐시 재생성)는 MIGRATION.md
  (`MIGRATION.md:135` 부근)에 한 줄 요약만 있고, 정확한 알고리즘(turn별
  캐시 재생성 타이밍, `_send_turn_cached`의 online 스케줄이 원본
  `_arm_cached`와 다른 스케줄이라는 점)은 `gemini.py` 모듈 docstring에만
  상세 서술되어 있다 — 설계 근거 문서로는 정상이나, ARCHITECTURE.md
  (구현 후 참조 문서)에는 이 차이가 전혀 언급되지 않는다.

### 4.3 일치 확인 (참고용, 상세 생략)

다음은 문서 인용과 코드 인용이 실질적으로 일치함을 확인한 항목들 —
DESIGN.md/MIGRATION.md의 자기 서술이 이번 코드 감사 결과와 부합한다.

- Backend 프로토콜의 connect/send_turn/close 3분할과 그 이유(mock은
  무연결, local_llm은 연결 자체가 측정 대상) — `DESIGN.md:19-25` vs
  `base.py:19-25`(동일 문구 수준으로 코드 docstring에 재기술됨).
  로컬llm ARMS=("chat",) 하나뿐이라는 이유도 `MIGRATION.md:260-264` vs
  `local_llm/__init__.py:49-55` 일치.
- Mock 재생이 "바이트 패턴만, 텍스트/지연 미보존"이라는 정책
  (`DESIGN.md:183`, "확정된 설계 결정" 표) vs `mock/replay.py:8-28`
  구현(`_placeholder`가 답변을 동일 길이 filler로 치환) — 일치.
- public_ai recorder의 민감정보 마스킹 요구사항(`DESIGN.md:250`, B2
  행) vs `recorder.py:8-15,73-97`의 `mask_secrets` 구현 — 일치, 오히려
  코드가 헤더+바디 재귀+Bearer 패턴까지 더 포괄적으로 구현.
- local_llm이 "엔진을 재구현하지 않는다"는 확정 결정
  (`DESIGN.md:181`, 표) vs `engine_adapter.py:1-19` 실제로 OpenAI 호환
  클라이언트만 구현 — 일치.
- transport 슬롯이 인터페이스만 예약하고 미구현이라는 B5 서술
  (`DESIGN.md:253`) vs `base.py:48-52` 및 `local_llm/gateway.py`가
  `X-AIPT-Transport` 헤더 반영 이상의 실제 전송 계층 분기를 갖지 않음
  — 일치(단, 이후 quic_mock이 그 확장 슬롯을 실제로 http3까지 채웠다는
  점은 DESIGN.md §7이 후속 절로 별도 서술하고 있어 모순 아님).
