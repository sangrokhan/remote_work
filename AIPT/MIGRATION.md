# AIPT — Migration Checklist

파일 단위 이관 체크리스트. DESIGN.md §6 Phase 순서를 따른다. 각 항목은
"원본 경로 → 목적지 경로" + 필요한 변경 사항을 기록한다. `[ ]`는 미착수,
작업 시작 시 `[~]`, 완료+테스트 그린 시 `[x]`로 갱신.

원본 경로 표기: `TT/` = `remote_work/token_traffic/`, `TC/` = `remote_work/tcp_congestion/`

## Phase 1 — 공유 core (native + cwnd/capture/offload)

- [ ] `TT/native/cwnd_monitor.c` (= `TC/native/cwnd_monitor.c`, 동일 확인됨) → `AIPT/native/cwnd_monitor.c`
- [ ] `TC/tcp_congestion/cwnd.py` (단순 인터페이스) + `TT/core/cwnd.py` (상세 docstring, dumps/exact_queries 계측) → `AIPT/aipt/core/cwnd.py` (병합, §5-1 결정 필요)
- [ ] `TT/core/capture.py` (AppArmor 감지 로직 보존) + `TC/tcp_congestion/capture.py` (label 일반화) → `AIPT/aipt/core/capture.py`
- [x] `TT/core/offload.py` + `TC/tcp_congestion/offload.py` → `AIPT/aipt/core/offload.py` (env alias 양쪽 지원: `NIC_OFFLOAD_DISABLE` 정식, `TRAFFIC_PCAP_NO_OFFLOAD` deprecated alias)
- [x] `TC/tcp_congestion/tcpinfo.py` → `AIPT/aipt/core/tcpinfo.py` (그대로, synthetic_mock 전용이나 core에 위치 — 1회성 스냅샷은 범용 유틸)
- [x] `TC/tcp_congestion/netem.py` → `AIPT/aipt/core/netem.py` (그대로, token_traffic엔 대응물 없음)
- [x] `TT/core/config.py` → `AIPT/aipt/core/config.py` (env 플래그 판독 통합, `TC`의 `_flag()` 헬퍼 흡수: `cwnd.py`/`offload.py`에 흩어져 있던 정의를 `flag()`/`flag_any()`/`env_int()`로 통합)
- [ ] `AIPT/tests/core/test_cwnd.py`, `test_capture.py`, `test_offload.py` — 양쪽 테스트 합집합, 중복 제거
- [ ] `AIPT/tests/core/test_cwnd_live.py` / `test_conversation_live.py`의 live 스타일 → 마커 통일 (§5-4)

## Phase 2 — external_api 도메인 (구 token_traffic)

- [ ] `TT/core/wire.py` → `AIPT/aipt/core/wire.py`
- [ ] `TT/core/streaming.py` → `AIPT/aipt/core/streaming.py`
- [ ] `TT/core/record.py` → `AIPT/aipt/core/record.py`
- [ ] `TT/core/call.py` → `AIPT/aipt/labs/external_api/call.py`
- [ ] `TT/core/cachebust.py` → `AIPT/aipt/labs/external_api/cachebust.py`
- [ ] `TT/core/metrics.py` → `AIPT/aipt/labs/external_api/metrics.py`
- [ ] `TT/core/runner.py` → `AIPT/aipt/labs/external_api/runner.py`
- [ ] `TT/core/scenario.py` → `AIPT/aipt/labs/external_api/scenario.py`
- [ ] `TT/core/store.py` → `AIPT/aipt/labs/external_api/store.py`
- [ ] `TT/core/export.py` → `AIPT/aipt/labs/external_api/export.py`
- [ ] `TT/providers/base.py` → `AIPT/aipt/providers/base.py`
- [ ] `TT/providers/gemini.py` → `AIPT/aipt/providers/gemini.py`
- [ ] `TT/providers/openai.py` → `AIPT/aipt/providers/openai.py`
- [ ] `TT/fixtures/perf.json` → `AIPT/fixtures/perf.json`
- [ ] `TT/tests/test_{wire,streaming,record,call,cachebust,metrics,runner,scenario,store,export,provider_gemini,provider_openai,config,capture,cwnd,cwnd_live,app}.py` → `AIPT/tests/{core,providers,labs/external_api}/` 로 분배
- [ ] import 경로 일괄 치환: `from core import X` → `from aipt.core import X` 또는 `from aipt.labs.external_api import X`

## Phase 3 — synthetic_mock 도메인 (구 tcp_congestion)

- [ ] `TC/tcp_congestion/server.py` → `AIPT/aipt/labs/synthetic_mock/server.py`
- [ ] `TC/tcp_congestion/probe.py` → `AIPT/aipt/labs/synthetic_mock/probe.py`
- [ ] `TC/tcp_congestion/conversation.py` → `AIPT/aipt/labs/synthetic_mock/conversation.py`
- [ ] `TC/tcp_congestion/congestion.py` → `AIPT/aipt/labs/synthetic_mock/congestion.py`
- [ ] `TC/tcp_congestion/netem.py` → `AIPT/aipt/labs/synthetic_mock/netem.py`
- [ ] `TC/tcp_congestion/export.py` → `AIPT/aipt/labs/synthetic_mock/export.py`
- [ ] `TC/entrypoint_client.py`, `entrypoint_server.py` → `AIPT/aipt/labs/synthetic_mock/entrypoints.py` (또는 docker/ 아래 스크립트로)
- [ ] `TC/tests/test_{congestion,conversation,conversation_live,cwnd,export,netem,offload,probe,server,capture,app}.py` → `AIPT/tests/labs/synthetic_mock/`
- [ ] import 경로 일괄 치환: `from tcp_congestion import X` → `from aipt.labs.synthetic_mock import X` / `from aipt.core import X`

## Phase 4 — 웹 UI 통합 (Flask → FastAPI)

- [ ] `TT/core/app.py` (Flask, 365줄) → 라우트 분석 후 `AIPT/aipt/web/routes_external_api.py` (FastAPI APIRouter)로 포팅
- [ ] `TC/tcp_congestion/app.py` (FastAPI, 209줄) → `AIPT/aipt/web/routes_synthetic_mock.py` (그대로 이관 후 prefix만 조정)
- [ ] `TT/templates/index.html` → `AIPT/aipt/web/templates/external_api/index.html`
- [ ] `TC/templates/index.html` → `AIPT/aipt/web/templates/synthetic_mock/index.html`
- [ ] `TT/static/app.js`, `style.css` → `AIPT/aipt/web/static/external_api/`
- [ ] `TC/static/.gitkeep` → 확인 후 필요 시 유지
- [ ] `AIPT/aipt/web/templates/index.html` 신규 작성 (랜딩 페이지, 두 lab 링크)
- [ ] `AIPT/aipt/web/app.py` 신규 작성: `create_app()`이 두 라우터 mount
- [ ] `TT/tests/test_app.py` + `TC/tests/test_app.py` → `AIPT/tests/web/`

## Phase 5 — Docker/인프라

- [ ] `TT/Dockerfile` → `AIPT/docker/Dockerfile.web` 베이스로 사용, native C 빌드 스테이지 추가
- [ ] `TC/Dockerfile.client` → `AIPT/docker/Dockerfile.web`에 통합 (client 역할 = web 앱)
- [ ] `TC/Dockerfile.server` → `AIPT/docker/Dockerfile.mockserver`
- [ ] `TT/docker-compose.yml` + `TC/docker-compose.yml` → `AIPT/docker/docker-compose.yml` (2-service: web, mockserver)
- [ ] `TC/.env` → `AIPT/.env.example`로 승격 (실제 `.env`는 git-ignore)
- [ ] `TT/.dockerignore` → `AIPT/.dockerignore`
- [ ] `TC/Makefile`(있다면 확인) — native C 빌드 타겟 통합

## Phase 6 — 문서/마감

- [ ] `TT/docs/core-contracts.md` → `AIPT/docs/core-contracts.md` (synthetic_mock 부분 추가 반영)
- [ ] `TT/docs/outputs.md` → `AIPT/docs/outputs.md` (두 lab의 산출물 스키마 모두 기술)
- [ ] `TT/README.md` + `TC/README.md` → `AIPT/README.md` (통합, 빠른 시작 가이드)
- [ ] `remote_work/CLAUDE.md` 프로젝트 테이블 갱신 (`token_traffic` 행 → `AIPT`, `tcp_congestion` 행 제거)
- [ ] 원본 `TT/`, `TC/` 디렉터리 처리 방침 확정 (삭제 vs `_archive/`로 이동) — **사용자 확인 필요**
- [ ] `TT/data/`, `TC/data/pcaps/` 등 실측 데이터 처리 방침 확정 (이관 vs 폐기) — **사용자 확인 필요**

## 검증 기준 (각 Phase 공통)

1. `pytest tests/` 전체 그린 (live 마커 제외 기본 실행 + live 포함 실행 둘 다 확인)
2. `python -c "import aipt.core.cwnd"` 등 신규 경로 import 스모크 테스트
3. Phase 4 완료 시: `uvicorn aipt.web.app:create_app --factory` 로컬 기동 후 두 lab의 UI 모두 브라우저 확인
4. Phase 5 완료 시: `docker compose -f docker/docker-compose.yml up --build` 로 mock 서버 실험 1회 정상 실행 확인 (external_api는 API 키 필요하므로 dry-run/mock 모드로만 확인)

## Phase 4.6 — export 3-레이어 통합 (DESIGN.md §4.6, B6/B7/B8) — [x] 완료

이 문서 상단의 Phase 1~6은 DESIGN.md §1~4(구 lab별 이관) 기준으로 작성된
것이라 `aipt/labs/*/export.py` 경로를 가리키고 있으나, §4.5 개정으로
`aipt/labs/` 자체가 폐기되고 `aipt/backends/` + `aipt/export/`로
재편되었다(§4.5 폴더 구조 diff 참고). 아래는 그 개정판 경로 기준의
실제 작업 기록이다 — 위 Phase 2/3의 `export.py` 항목들은 이 섹션으로
대체된 것으로 간주한다.

- [x] `TT/core/export.py`(records.csv/summary.csv) + `TC/tcp_congestion/export.py`(cwnd.csv/turns.csv) → `AIPT/aipt/export/{connection,turns}.py`로 병합 (labs 경유 없이 backends → export 직결 구조)
- [x] `AIPT/aipt/export/connection.py` — cwnd.csv/cwnd_summary.csv. `aipt.core.cwnd.Monitor.result()` 그대로 소비, provider/arm/kind 3필드 대신 §6 결정#1의 단일 `label` 컬럼 채택
- [x] `AIPT/aipt/export/turns.py` — turns.csv. `aipt.backends.record.turn_record()` 스키마(3-backend 공통) 기준, tcp_congestion 쪽 전용 컬럼(prompt_bytes/idle_ms/probe_*)은 옵셔널로 병합. **`goodput_bps`(B7) 신규 구현**: `(wire_recv 또는 resp_payload_bytes) * 8 / (turn_end_ms - req_sent_ms)`, 0-나눗셈 가드
- [x] `AIPT/aipt/export/packets.py` — packets.csv (B6, 완전 신규). `dpkt` optional dependency + 순수 stdlib(`struct`) classic-pcap 파서 폴백 양쪽 구현, 실제 pcap 파일 없이 `write_pcap()` 헬퍼로 합성 픽스처 생성 후 라운드트립 테스트
- [x] `AIPT/aipt/export/bundle.py` — 세 CSV + pcap을 zip으로 묶는 유틸. `TC/tcp_congestion/app.py`의 `download_bundle_zip` 로직을 라우트 비의존 형태로 일반화(`build_bundle_zip()`)
- [x] `AIPT/pyproject.toml` — `[project.optional-dependencies] export = ["dpkt"]` 추가 (append만, 기존 `dev` 그룹은 변경 없음)
- [x] `AIPT/tests/export/test_{connection,turns,packets,bundle}.py` 신규 작성 — 35개 테스트, 합성 데이터/합성 pcap만 사용 (실제 소켓·tcpdump·netlink 불필요)
- [x] 검증: `pytest tests/ -q -m "not live"` → 226 passed, 1 skipped (dpkt 없을 때만 스킵되는 교차검증 테스트 1개, dpkt 설치 후 재실행 시 227 passed 확인), `from aipt.export import connection, turns, packets, bundle` 임포트 스모크 통과

## Phase 4.5 — aipt/backends/mock/ (DESIGN.md §4.5/5, A3/B1/B3) — [x] 완료

§4.5 개정으로 `aipt/labs/synthetic_mock/`가 폐기되고 `aipt/backends/mock/`로
재편된 경로 기준 실제 작업 기록. `aipt/backends/mock/__init__.py`의
`NotImplementedBackend` 스텁을 실제 `MockBackend` 구현으로 교체했다.

- [x] `TC/tcp_congestion/server.py` → `AIPT/aipt/backends/mock/server.py` — HTTP/1.1 keep-alive mock 서버 그대로 이관 + fixture 답변 텍스트 서빙 확장(B1): `/inference-mock?turn=<i>`에 `fixture` 바인딩 시 `answer` 필드 반환, `response_bytes` 미지정이면 답변 길이로 자동 패딩. `fixture=None`(기본)이면 원본과 동일한 순수 dummy-byte 동작
- [x] `AIPT/aipt/backends/mock/fixtures.py` — 신규(B1). Q&A JSON fixture 로더(`load`/`load_qa_fixture`, 스키마: `{name, system_prompt, turns:[{question,answer}]}`, `token_traffic/core/scenario.py` 정신 계승하되 mock 전용으로 단순화) + 기존 byte-size 스윕 방식(`byte_size_fixture`) 옵션 유지(DESIGN.md §5 "C. 폐기/대체" 방침대로 폐기 아닌 옵션화). `AIPT/fixtures/smoke.json` 신규 fixture 파일 추가(테스트용)
- [x] `AIPT/aipt/backends/mock/replay.py` — 신규(B3). 실측 캡처(fixture와 동일 스키마)를 로드해 **바이트 크기만** 보존하고 답변 텍스트는 동일 길이의 placeholder로 치환하는 `from_capture_doc`/`from_capture_file`. 지연시간은 재현하지 않음(`inference_delay_ms`로 별도 제어, `MockBackend`/`conversation.run()`의 기존 파라미터 그대로 사용)
- [x] `TC/tcp_congestion/conversation.py` → `AIPT/aipt/backends/mock/conversation.py` — 저수준 로직(`turn_prompt_size`/`build_turns`/`run()`) 그대로 이관, `aipt.core.cwnd`/`aipt.core.capture`와 연동 유지. 신규: `MockBackend` 클래스 추가 — `aipt.backends.base.Backend` 프로토콜(`connect`/`send_turn`/`close`) 구현, `connect()`에서 서버 스레드 기동 + keep-alive 소켓 오픈 + `cwnd.Monitor` 시작, `send_turn()`에서 fixture turn별 질문 전송·답변 수신을 `aipt.backends.record.Exchange`로 반환, `close()`에서 소켓/모니터/서버 정리(cwnd 결과는 `close()` 후에도 `cwnd_result()`로 조회 가능하도록 캐싱)
- [x] `TC/tcp_congestion/probe.py` → `AIPT/aipt/backends/mock/probe.py` — 그대로 이관 (idle-gap RTT HTTP PING, delivery_rate 미포함)
- [x] `AIPT/aipt/backends/mock/__init__.py` — `NotImplementedBackend` 스텁 제거, `MockBackend`를 `aipt.backends.get("mock").MockBackend`로 노출. `aipt/backends/__init__.py`의 레지스트리(`_KNOWN`/`get()`)와 호환 유지(변경 없음)
- [x] `AIPT/tests/test_backends_base.py` — `test_backend_registry_names_and_get`을 mock의 졸업(스텁 제거)을 반영하도록 패치(다른 두 backend는 여전히 스텁이므로 개별 backend별로 독립 검증), `test_mock_backend_is_implemented_and_satisfies_protocol` 신규 추가
- [x] `TC/tests/test_{server,probe,conversation,conversation_live}.py` → `AIPT/tests/backends/mock/test_{server,probe,conversation,conversation_live}.py` 로 이관·통합, live 테스트는 `@pytest.mark.live`로 마킹(모듈 레벨 `pytestmark = pytest.mark.live`). 신규: `test_fixtures.py`(B1), `test_replay.py`(B3), `test_mock_backend.py`(비-live 프로토콜/구성 스모크) — mock 스코프 총 60개 테스트
- [x] 검증: `pytest tests/backends/ -q` → 60 passed. `pytest tests/ -q -m "not live"` → mock 관련 217 passed(기존 export 스코프의 무관한 2건 실패는 다른 작업자 영역, `pytest tests/ -q -m "not live" --ignore=tests/test_backends_base.py` 및 `-k mock`으로 별도 확인). `from aipt.backends.mock import server, fixtures, replay, conversation` 임포트 스모크 통과

## Phase 4.5b — aipt/backends/public_ai/ (DESIGN.md §4.5/5, A2/B2) — [x] 완료

§4.5 개정으로 `aipt/labs/external_api/` + `aipt/providers/*`가 폐기되고
`aipt/backends/public_ai/`로 재편된 경로 기준 실제 작업 기록.
`aipt/backends/public_ai/__init__.py`의 `NotImplementedBackend` 스텁을
실제 `PublicAIBackend` 구현으로 교체했다.

- [x] `TT/core/wire.py` → `AIPT/aipt/core/wire.py` — 소켓 바이트 카운터, 변경 없이 그대로 이관(공유 core 유틸이므로 core에 위치, 이미 있으면 건드리지 않기로 했으나 미존재 확인 후 신규 작성)
- [x] `TT/core/streaming.py` → `AIPT/aipt/core/streaming.py` — SSE 리더, 변경 없이 그대로 이관
- [x] `TT/core/call.py` → `AIPT/aipt/backends/public_ai/_call.py` — public_ai 전용 내부 모듈로 이관(leading underscore; core로 승격하지 않음 — bytes-vs-latency dual-pass 정책은 billed public API에 특화된 것이라 재사용 범용 유틸이 아니라고 판단). import를 `aipt.core.wire`/`aipt.core.streaming`으로 갱신, `Exchange`가 `aipt.backends.record.TurnExchange` 덕타이핑 프로토콜을 만족하도록 필드 유지
- [x] `TT/core/cachebust.py` → `AIPT/aipt/backends/public_ai/_cachebust.py` — 동일하게 public_ai 전용 내부 모듈로 이관(provider→backend 파라미터명만 변경)
- [x] `TT/providers/gemini.py` → `AIPT/aipt/backends/public_ai/gemini.py` — arm 목록(`stateless`/`nocontext`/`cached`/`interaction`/`interaction_inline`/`interaction_stateless`)과 API 바디 빌드 로직 그대로 보존. 레거시 `run_arm()`(전체 대화 일괄 리플레이, 원본 fixture 테스트와의 patiry 검증용) 유지 + 신규 `GeminiBackend` 클래스(`aipt.backends.base.Backend` 프로토콜 구현: `connect`/`send_turn`/`close`). `cached` arm은 원본이 두-패스(전체 트랜스크립트 선-리플레이 후 캐시 일괄 생성)라 턴 단위 프로토콜에 맞지 않아 **온라인 캐싱으로 재설계**했음을 모듈 docstring에 명시 — turn 1은 캐시 없이 시스템 프롬프트+질문을 그대로 보내고, 매 턴 종료 후 그때까지의 트랜스크립트로 캐시를 (재)생성해 다음 턴이 참조하는 방식(정확히 같은 스케줄은 아니지만 "이후 턴은 서버측 캐시를 참조" 라는 비용 형태는 보존)
- [x] `TT/providers/openai.py` → `AIPT/aipt/backends/public_ai/openai.py` — arm 목록(`chat_stateless`/`responses_stateless`/`responses`/`responses_inline`)과 바디 빌드 로직 그대로 보존. `run_arm()` 유지 + 신규 `OpenAIBackend` 클래스. `responses_inline`의 conversation-create 준비 호출은 `connect()`에서 실행하고, `connect()`가 반환값을 가질 수 없는 Backend 프로토콜 제약 때문에 `pending_setup_records` 리스트에 적재해 클라이언트가 drain하도록 함(원본의 "records 리스트 맨 앞에 setup 레코드 prepend" 동작을 lifecycle로 옮긴 것)
- [x] `AIPT/aipt/backends/public_ai/recorder.py` — 신규(B2). 실제 API 호출의 request/response 원문을 `token_traffic/fixtures/perf.json`과 같은 모양(top-level `system`/`steps` + `turns`)의 fixture로 저장하는 `RecordedTurn`/`FixtureWriter`. **민감정보 마스킹**: `mask_secrets()`가 헤더(Authorization/x-goog-api-key/api-key/x-api-key)와 바디 내 어디든(`api_key`/`apiKey`/`token`/`secret` 등 키 이름 기반 + `Bearer ...` 패턴)을 재귀적으로 마스킹, 디스크에 쓰기 전에 항상 적용. `recording_backend()` — 기존 Backend 인스턴스를 감싸 매 `send_turn`을 자동으로 fixture에 기록하는 옵트인 프록시(원본 backend는 recorder에 의존하지 않음)
- [x] `AIPT/aipt/backends/public_ai/__init__.py` — `NotImplementedBackend` 스텁 제거, `PublicAIBackend` 파사드 신규 구현: gemini/openai 두 엔진이 하나의 `public_ai` 레지스트리 슬롯을 공유하므로(§4.5), `connect(arm=...)`에 전달된 arm 이름으로 엔진을 자동 판별(`_engine_for_arm`)하거나 생성자에 `engine="gemini"|"openai"`로 고정 가능. `aipt/backends/__init__.py`의 `get()`/`names()` 인터페이스는 변경 없이 그대로 호환
- [x] `TT/fixtures/perf.json` → `AIPT/fixtures/perf.json` — 그대로 복사(gemini 테스트 시나리오 픽스처)
- [x] `TT/tests/test_provider_gemini.py` → `AIPT/tests/backends/public_ai/test_gemini.py` — 원본 테스트 전량 이관(네트워크 불필요, mock 모드) + `GeminiBackend` lifecycle 신규 테스트 6개 추가
- [x] `TT/tests/test_provider_openai.py` → `AIPT/tests/backends/public_ai/test_openai.py` — 원본 테스트 전량 이관 + `OpenAIBackend` lifecycle 신규 테스트 5개 추가
- [x] `TT/tests/test_wire.py` → `AIPT/tests/core/test_wire.py` — 그대로 이관(로컬 keep-alive HTTP 서버만 사용, 외부 네트워크/`@pytest.mark.live` 불필요)
- [x] `TT/tests/test_call.py` → `AIPT/tests/backends/public_ai/test_call.py` — 그대로 이관, import를 `aipt.backends.public_ai._call`로 갱신
- [x] `AIPT/tests/backends/public_ai/test_recorder.py` — 신규(B2). 마스킹 유닛 테스트 + `recording_backend()`로 감싼 mock-mode `GeminiBackend` 호출이 실제 `GEMINI_API_KEY` 환경변수 값을 디스크에 전혀 쓰지 않음을 검증하는 종단 테스트
- [x] `AIPT/pyproject.toml` — `dependencies = ["requests"]`로 승격(기존 `dependencies = []` → base dependency; mock 모드에서도 `aipt.core.wire`가 모듈 로드 시점에 `requests`를 import하므로 optional이 아니라 base로 필요)
- [x] live(실제 API 키 필요) 테스트는 이관 대상에 없었음 — 원본 `test_provider_gemini.py`/`test_provider_openai.py`/`test_call.py`/`test_wire.py`가 이미 전부 mock-mode 또는 로컬 소켓 전용으로 작성되어 있어 `@pytest.mark.live` 마킹이 필요한 테스트가 없었음(실제 키가 필요한 live 통합 테스트는 이 스코프에 존재하지 않았고, 만약 향후 추가한다면 `@pytest.mark.live`로 마킹)
- [x] 검증: `pytest tests/ -q -m "not live"` → **296 passed, 1 skipped, 11 deselected** (python3.11 venv 인터프리터와 python3.12 시스템 인터프리터 양쪽에서 재확인, 두 인터프리터의 site-packages가 분리되어 있어 `requests`를 양쪽에 개별 설치함: `.venv/bin/pip install -e ".[dev]"` 및 `.venv/bin/python -m pip install -e ".[dev]"`). `from aipt.backends.public_ai import gemini, openai, recorder` 임포트 스모크 양쪽 인터프리터에서 통과

## Phase 4 — aipt/web/ (DESIGN.md §3/5, backend 선택형 UI) — [x] 완료

§4.5 개정으로 원래 계획했던 "external-api lab / synthetic-mock lab" 2-URL-네임스페이스
설계(§3, 위 Phase 4 섹션의 원안)는 폐기되고, **backend 선택형** 단일 웹 UI로
대체되었다. 클라이언트가 `aipt.backends.get(name)`으로 얻은 backend 인스턴스
하나를 골라서 실행하는 구조이므로, `routes_external_api.py`/`routes_synthetic_mock.py`
분리 대신 `routes_config.py`(랜딩+설정)/`routes_run.py`(실행)/`routes_runs.py`(조회+CSV/zip
다운로드) 3-라우터 구조로 구현했다. 이 섹션이 원래의 Phase 4 체크리스트를
대체한다(원안의 `TC/tcp_congestion/app.py` 그대로 이관 방침도 backend 선택형으로
바뀌면서 폐기).

- [x] `AIPT/aipt/web/app.py` — `create_app()` 팩토리: FastAPI 인스턴스 생성,
  `Jinja2Templates(aipt/web/templates)` 등록, `routes_config.register(app, templates)` +
  `routes_run.router`/`routes_runs.router` include, `aipt/web/static`를 `/static`에 mount.
  모듈 레벨 `app = create_app()`도 노출(`uvicorn aipt.web.app:app` 직접 기동 지원)
- [x] `AIPT/aipt/web/routes_config.py` — `GET /`(랜딩, backend 선택 카드 3개) +
  `GET /api/config`(congestion algorithm 목록, cwnd/capture 가용성, `aipt.backends.names()`
  기반 backend 목록 — 각 backend의 `ready()`/`ARMS`를 `PublicAIBackend`/`MockBackend`/
  `LocalLLMBackend` 파사드 클래스 존재 여부로 판별하여 `implemented` 플래그 계산.
  하드코딩된 이름 목록이 아니라 속성 존재 여부로 판별하므로, local_llm이 스텁에서
  실제 구현으로 교체되어도(실제로 이 작업 중 병렬 작업자가 완료함 — 아래 참고) 코드
  변경 없이 자동으로 "구현됨"으로 반영됨)
- [x] `AIPT/aipt/web/routes_run.py` — `POST /api/run`: `RunRequest` Pydantic 모델(backend/arm/
  turns/measure/mock 전용 옵션 등), `run_in_threadpool`로 동기 conversation 실행을 감쌈
  (DESIGN.md §3 방침 그대로). `aipt.backends.get(name)` → 파사드 인스턴스 생성 →
  `connect`/`send_turn`*/`close` 라이프사이클 구동 → `aipt.backends.record.turn_record()`로
  턴별 레코드 조립 → `aipt/web/store.py`에 저장. 알 수 없는 backend 이름은 400,
  미구현 backend(`NotImplementedError`)는 501로 변환(스텁 상태였던 local_llm을 염두에
  두고 작성했으나, 실행 시점엔 이미 병렬 작업자가 `LocalLLMBackend`를 구현 완료한
  상태였음 — 여전히 `NotImplementedError` 방어 코드는 유지, local_llm이 다시 스텁으로
  돌아가는 경우에도 501로 안전하게 처리됨을 테스트로 확인)
- [x] `AIPT/aipt/web/routes_runs.py` — `GET /api/runs`, `GET/DELETE /api/runs/{exec_id}`,
  `GET /api/runs/{exec_id}/{turns,summary,cwnd,cwnd_summary,packets}.csv`,
  `GET /api/runs/{exec_id}/bundle.zip`, `GET /api/pcaps/{name}`. CSV/zip은
  `aipt.export.{connection,turns,packets,bundle}`의 완성된 함수를 그대로 소비(신규 export
  로직 없음, 라우트는 얇은 어댑터). `summary.csv`만 원본에 없던 신규 최소 구현(런당
  1-arm 실행 구조라 provider/arm별 다중 행 대신 실행 1건당 1행으로 단순화)
- [x] `AIPT/aipt/web/store.py` — 신규. 메모리 내 최근 `MAX_RUNS=50`개 run만 유지하는
  `OrderedDict` 기반 store(`save_run`/`get_run`/`delete_run`/`list_runs`). **TODO(persistence)**:
  파일/DB 영속화는 이번 phase 범위 밖 — 모듈 docstring에 명시, 프로세스 재시작 시
  run 이력 소실됨
- [x] `AIPT/aipt/web/templates/index.html` — 랜딩 페이지: backend 선택 카드 3개(Public AI/
  Mock/Local LLM, 카드 렌더링은 `/api/config`의 `implemented` 플래그로 조건부 "구현 예정"
  배지 표시) + 실험 폼(`_experiment_form.html` include) + 결과 영역 + 최근 실행 테이블
- [x] `AIPT/aipt/web/templates/_experiment_form.html` — 신규. 공통 필드(backend/arm select,
  turns textarea, measure, capture 체크박스) + backend별 조건부 fieldset(mock: response
  bytes/inference delay/congestion algorithm, public_ai: model/system, local_llm: 안내 문구)
- [x] `AIPT/aipt/web/static/app.js` — 신규. backend select 변경 시 arm 드롭다운/조건부
  fieldset 토글, `POST /api/run` 호출, 결과를 텍스트+테이블로 렌더링(차트 없음, DESIGN.md
  §5 범위 밖 명시), `/api/runs` 폴링으로 최근 실행 테이블 갱신
- [x] `AIPT/aipt/web/static/style.css` — 신규. 최소 스타일(카드 레이아웃, 상태 색상,
  테이블), 프레임워크 없음
- [x] 구 `aipt/web/templates/external_api/`, `aipt/web/templates/synthetic_mock/`
  (§4.5 개정 이전 스켈레톤, 내용 없이 `.gitkeep`만 존재) 삭제 — backend 선택형 구조와
  더 이상 대응하지 않음
- [x] `AIPT/tests/web/test_app.py` — 신규. FastAPI `TestClient` 스모크 테스트 6개:
  `GET /`(200, backend 카드 렌더링 확인), `GET /api/config`(3-backend 레지스트리 전체
  반영 확인), `POST /api/run`(mock backend로 실제 conversation 2턴 실행 → `/api/runs`
  목록/`GET /api/runs/{id}`/6종 CSV·zip 다운로드/`DELETE` 라운드트립), 알 수 없는
  backend(400), local_llm(구현 상태와 무관하게 500 traceback 유출 없음을 확인),
  404 케이스
- [x] `AIPT/pyproject.toml` — `[project.optional-dependencies] web = ["fastapi",
  "uvicorn", "jinja2", "python-multipart"]` 신규 그룹 추가(append만, 기존 그룹 변경 없음)
- [x] 검증: `.venv/bin/python -m pip install -e ".[dev,web,export]"` → 설치 성공.
  `pytest tests/ -q -m "not live"` → **349 passed, 1 skipped, 12 deselected**(기존
  296 passed 기준선 대비 backends/local_llm·gateway 등 병렬 작업자 산출물 포함 전체
  스위트가 함께 증가한 수치). `uvicorn aipt.web.app:app` 로컬 기동 후
  `curl http://127.0.0.1:18080/`(200), `curl http://127.0.0.1:18080/api/config`(200,
  3-backend 응답), `curl -X POST http://127.0.0.1:18080/api/run -d '{"backend":"mock",...}'`
  (200, 2턴 conversation 실제 실행 결과 반환) 모두 확인 후 서버 종료

## Phase 4.5c — aipt/backends/local_llm/ (DESIGN.md §4.5/4.7/4.9/5, B4) — [x] 완료

`aipt/backends/local_llm/__init__.py`의 `NotImplementedBackend` 스텁을 실제
`LocalLLMBackend` 구현으로 교체했다. DESIGN.md 4.5 확정 방침대로 로컬
서빙엔진(llama.cpp/vLLM)을 직접 재구현하지 않고, 그 앞단에 자체
"engine gateway"(프록시)를 두는 구조로 구현했다 — DESIGN.md 4.7의 **Network
Gateway 컨테이너**(별도 컴포넌트, `tc netem` 기반 L3/L4 지연·손실 주입, B9로
이번 범위 밖)와는 개념적으로 다르며, 코드 주석/모듈 docstring에서
"engine gateway/proxy"(이 작업, 애플리케이션 레벨) vs "Network Gateway"(별도
컨테이너, L3/L4)로 명시적으로 구분해뒀다.

- [x] `AIPT/aipt/backends/local_llm/engine_adapter.py` — 신규. llama.cpp
  `llama-server`(OpenAI 호환 HTTP 서버) 또는 vLLM OpenAI 호환 API 서버를 향한
  **얇은 HTTP 클라이언트**로만 구현(`EngineAdapter`: `build_body`/`headers`/
  `chat_completions_url`/`text_of`/`usage_of`). 서버 프로세스를 직접 기동하지
  않음 — `LOCAL_LLM_ENGINE_URL`(기본 `http://127.0.0.1:8080`, llama-server 기본
  포트) 환경변수로 이미 떠 있는 엔진의 주소만 받는다. `LOCAL_LLM_ENGINE_KIND`
  (`llama_cpp`/`vllm`, 기본 `llama_cpp`)는 순수 레이블일 뿐 요청 처리 로직은
  두 엔진 모두 동일한 OpenAI 호환 스키마로 분기 없이 처리. `LOCAL_LLM_MODEL`,
  `LOCAL_LLM_API_KEY`(선택)도 지원
- [x] `AIPT/aipt/backends/local_llm/gateway.py` — 신규. `engine_adapter`와
  클라이언트 사이에 위치하는 자체 프록시 계층(`Gateway`). `on_request(hook)`/
  `on_response(hook)` 콜백 등록 포인트(구독 해제 함수 반환, 훅에서 예외가 나도
  전체 호출은 죽지 않음 — best-effort 계측이라는 기존 `wire.watch_connections`/
  `cwnd.announce` 관례와 동일)로 향후 HTTP 신기능 실험을 위한 훅만 마련하고
  실제 실험 로직은 구현하지 않음(B4/B5 scope 그대로). `Backend.transport`
  슬롯(`aipt.backends.base.Transport`)을 매 요청의 `X-AIPT-Transport` 헤더에
  반영하는 최소 구현으로 "신기능 실험 지점"을 실제로 1개 동작 예시로 남김.
  `aipt.core.wire.wire_counter()`로 소켓 바이트/타이밍 계측(public_ai의
  `_call._blocking`과 동일한 계측 패턴 재사용)
- [x] `AIPT/aipt/backends/local_llm/__init__.py` — `NotImplementedBackend`
  스텁 제거, `LocalLLMBackend`(`aipt.backends.base.Backend` 프로토콜:
  `connect`/`send_turn`/`close`) 신규 구현. `MockBackend`와 동일한 형태로
  연결 전체 수명 동안 하나의 `aipt.core.cwnd.Monitor`를 운용하되, `MockBackend`
  처럼 직접 소켓을 열지 않고(엔진 연결은 `aipt.core.wire`의 풀링 세션이 지연
  오픈) `aipt.core.wire.watch_connections()` 훅으로 소켓을 감지해
  `Monitor.announce()`(DESIGN.md 4.9: idle-window 첫 샘플을 놓치지 않기 위한
  기존 관례 그대로 준수)하는 방식 채택. `connect()`에서
  `wire.reset_session()`을 먼저 호출해 이전 run/test가 남긴 커넥션을 재사용하지
  않도록 함(재사용 시 이미 slow-start를 지난 소켓을 관찰하게 되어 cwnd 실험이
  무의미해짐). `ARMS = ("chat",)` 하나만 — 로컬 엔진에는 Gemini의 캐시/OpenAI의
  저장된 응답 같은 표준화된 서버측 세션 개념이 없어 public_ai처럼 다중 arm을
  둘 근거가 없음(멀티턴은 클라이언트가 매 턴 growing message list를 전송하는
  단일 방식으로 충분). `transport` 슬롯은 생성자 인자로 받아 그대로
  `gateway.Gateway`에 전달
- [x] `AIPT/tests/backends/local_llm/fake_server.py` — 신규 테스트 fixture.
  표준 라이브러리 `http.server`만으로 OpenAI 호환 `/v1/chat/completions`를
  흉내내는 fake 서버(`FakeOpenAICompatHandler`) — 실제 llama.cpp/vLLM 프로세스
  없이 engine_adapter/gateway/LocalLLMBackend를 종단 검증
- [x] `AIPT/tests/backends/local_llm/test_engine_adapter.py` — 신규, 17개
  테스트. env 기반 설정 해석, 요청 바디 조립(`build_body`/`extra`/`extra_body`
  병합), 스트리밍·블로킹·레거시 세 응답 shape 모두에 대한 `text_of()`, `usage_of()`
  매핑, 방어적 동작(dict가 아닌 입력에도 예외 없이 빈 문자열) 커버
- [x] `AIPT/tests/backends/local_llm/test_gateway.py` — 신규, 9개 테스트.
  fake 서버 대상 실제 소켓 왕복으로 wire_sent/recv 계측, `X-AIPT-Transport`
  헤더 반영, `on_request`/`on_response` 훅 등록·구독해제·예외 무시,
  HTTP 5xx/연결 거부(`ConnectionRefusedError`) 모두 예외 없이 `GatewayResult
  .error`로 보고됨을 검증
- [x] `AIPT/tests/backends/local_llm/test_local_llm_backend.py` — 신규,
  13개 테스트. `Backend` 프로토콜 만족, `connect` 전/후 상태, 알 수 없는 arm
  거부, `send_turn` 전 호출 시 `RuntimeError`, 전체 lifecycle(1턴/멀티턴 히스토리
  누적/progress 콜백/close 후 재호출 안전성/`cwnd_result()` 가용성/연결
  실패 시에도 예외 없이 에러가 담긴 `Exchange` 반환) 커버
- [x] `AIPT/tests/backends/local_llm/test_engine_live.py` — 신규, `@pytest.mark.live`
  1개. 실제 llama.cpp `llama-server`/vLLM OpenAI 호환 서버가
  `LOCAL_LLM_ENGINE_URL`에 떠 있을 때만 의미 있는 종단 테스트(기본 스위트에서는
  수집되지 않음, `pytest -m live`로만 실행)
- [x] `AIPT/tests/test_backends_base.py` — `test_local_llm_backend_is_implemented_and_satisfies_protocol`
  신규 추가(mock/public_ai 졸업 시와 동일한 패턴)
- [x] `AIPT/pyproject.toml` — 신규 의존성 불필요(기존 `requests`/`pytest`만
  사용, `http.server`는 표준 라이브러리) — 변경 없음
- [x] 검증: `pytest tests/backends/local_llm -q -m "not live"` → 36 passed, 1
  deselected. `pytest tests/ -q -m "not live"` → **349 passed, 1 skipped, 12
  deselected**(웹 레이어 등 다른 병렬 작업자 영역과 함께 그린, 3회 재실행으로
  flake 없음 확인). `from aipt.backends.local_llm import engine_adapter,
  gateway` 및 `from aipt.backends.local_llm import LocalLLMBackend` 임포트
  스모크 통과. `aipt.backends.get("local_llm").LocalLLMBackend()`가
  `aipt.backends.base.Backend` 프로토콜을 만족함을 `isinstance` 체크로 확인

## Network Gateway 컨테이너 (DESIGN.md 4.7, B9) — 신규 구현

- [x] `aipt/gateway/` — 신규 패키지. `aipt/core/netem.py`(delay 전용, 컨테이너
  기동 시 1회 적용)를 Gateway 컨테이너의 실제 제어 루프로 승격 —
  delay/jitter/loss/reorder 전체를 다루고 런타임에 여러 번 교체 가능하게 확장.
  fq 하위 qdisc 체이닝(BBR pacing 보존)은 원본 로직 그대로 유지
  - `aipt/gateway/profiles.py` — `Profile` dataclass + `PRESETS`
    (clean/broadband/3g/satellite/lossy, DESIGN.md 4.7 드롭다운 순서와 동일).
    `custom_profile()`/`resolve()`로 임의 파라미터 조합 지원. `from_env()`:
    `GATEWAY_PROFILE`(프리셋 직접 선택) 또는 `GATEWAY_DELAY_MS`/
    `GATEWAY_JITTER_MS`/`GATEWAY_LOSS_PCT`/`GATEWAY_REORDER_PCT`(개별 커스텀
    값) 읽음. 기존 `CLIENT_NETEM_DELAY_MS`/`SERVER_NETEM_DELAY_MS`는
    deprecated delay-only alias로 지원(canonical `GATEWAY_DELAY_MS`가 우선)
  - `aipt/gateway/netem_control.py` — `tc qdisc netem` 커맨드 구성/실행.
    `apply_profile(iface, profile)`/`current_profile(iface)`/`clear(iface)`.
    `aipt.core.offload`/`aipt.core.capture`와 동일한 "예외로 죽지 않고
    `{"ok": bool, "reason": "..."}` 보고" 패턴 — `tc` 미설치나
    CAP_NET_ADMIN 부재(이 샌드박스의 실제 상태) 모두 이 경로로 보고됨.
    `tc qdisc del ... root`가 빈 인터페이스에서 nonzero를 반환하는 정상
    케이스는 별도로 흡수(idempotent 재적용 보장)
  - `aipt/gateway/app.py` — 독립 FastAPI 미니앱(`aipt/web`과 별도 프로세스로
    배포 가정). `GET /health`(netem 가용성 포함), `GET /gateway/profile`,
    `POST /gateway/profile`(프리셋 이름 또는 `{"profile":"custom", ...}`).
    netem 적용 실패도 500이 아니라 `{"ok": false, "reason": ...}` 200 응답으로
    보고
- [x] `docker/Dockerfile.gateway` — 신규. `python:3.12-slim` + `iproute2`(tc
  포함) 설치 + `pip install ".[web]"`(fastapi/uvicorn 재사용, 별도
  `gateway` extra 정의하지 않음) + `uvicorn aipt.gateway.app:app` 구동.
  런타임에 `--cap-add=NET_ADMIN`/`cap_add: [NET_ADMIN]` 필요하다는 주석 명시
  — docker-compose.yml 통합(B10)은 이번 범위 밖, 별도 작업자 진행 예정
- [x] `AIPT/tests/gateway/` — 신규, 48개 테스트.
  `test_profiles.py`(프리셋 값 정의, `from_env()`의 env 우선순위/deprecated
  alias, `custom_profile()` 음수 클램프 등), `test_netem_control.py`
  (subprocess.run 전량 mock — 커맨드 구성 로직 + CAP_NET_ADMIN 부재/`tc`
  미설치/idempotent del 흡수 등 실패 경로), `test_app.py`(FastAPI
  TestClient로 4개 라우트 + 422 검증 + 알 수 없는 프로파일명도 500 없이
  `ok:false`로 응답하는지 확인)
- [x] `AIPT/pyproject.toml` — 변경 없음(`web` extra의 fastapi/uvicorn을
  Gateway 앱도 재사용, `gateway` extra 별도 정의 안 함)
- [x] 실제 환경 검증(이 샌드박스): `tc`는 설치돼 있으나 CAP_NET_ADMIN 없음 —
  `netem_control.apply_profile("lo", PRESETS["3g"])` 실제 실행 결과
  `{"ok": false, "reason": "... RTNETLINK answers: Operation not
  permitted ... CAP_NET_ADMIN ..."}`로 정직하게 보고됨(예외 없이) 확인
- [x] 검증: `pytest tests/gateway -q` → 48 passed. `pytest tests/ -q -m "not
  live"` → **410 passed, 1 skipped, 12 deselected** (전체 스위트 그린,
  다른 병렬 작업자 영역과 함께). `from aipt.gateway import profiles,
  netem_control, app` 임포트 스모크 통과

## Phase 4.9 — B13 pcap 타임스탬프 정밀도 검토 (DESIGN.md §4.9) — [x] 완료

- [x] `AIPT/aipt/core/capture.py` — `ethtool_path()`(`offload.py`의 동명
  헬퍼와 동일 패턴, capture.py 자체 스코프) + `timestamp_source(iface:
  str = "eth0") -> dict` 신규 추가. `ethtool -T <iface>`를 실행해
  `Capabilities:` 블록에서 `hardware-transmit`/`hardware-receive` 라인
  존재 여부로 하드웨어 타임스탬프 지원을 판별,
  `{"iface", "hardware_timestamping", "raw", "available", "reason"}` 반환.
  ethtool 미설치/실행실패/비정상 종료 모두 예외를 던지지 않고
  `available=False` + 사유 문자열로 보고 — `available()`/`offload.describe()`와
  동일한 "가용성 감지 후 안내" 관례
- [x] `Capture.__init__`에서 `self.timestamp_source = timestamp_source(self.interface)`로
  1회 캐싱(ethtool 재호출 없이 `result()`에서 재사용), `result()`의 정상/에러
  두 경로 모두에 `"timestamp_source"` 필드 포함
- [x] `AIPT/aipt/export/packets.py` — 기존 `packets_csv()`의 `PACKET_COLUMNS`
  스키마(컬럼 순서/개수)는 건드리지 않음. 대신 신규 `gap_confidence_summary(pcap_path,
  timestamp_source=None) -> dict` 별도 함수 추가: pcap의 inter-arrival gap
  중앙값(`median_gap_ms`)을 계산하고, 중앙값이 1ms 미만이면서 타임스탬프
  소스가 소프트웨어(또는 불명)일 때 `timestamp_precision_reason`에 경고
  문장을 채움(하드웨어 타임스탬프거나 gap이 충분히 크면 빈 문자열). `aipt.core.capture`를
  import하지 않고 `timestamp_source` dict를 파라미터로만 받아 export/core
  간 의존성 방향 유지
- [x] `AIPT/tests/core/test_capture.py` — `timestamp_source()` 신규 테스트
  7개(하드웨어 있음/소프트웨어만/ethtool 없음/비정상 종료/실행 예외/Capture
  캐싱+result() 반영/에러 경로 result()에도 포함), 기존 테스트는 전량 무수정
- [x] `AIPT/tests/export/test_packets.py` — `gap_confidence_summary()` 신규
  테스트 6개(짧은 gap+소프트웨어 경고/하드웨어 무경고/긴 gap 무경고/타임스탬프
  소스 불명 경고/pcap 없음·1패킷 처리) + `PACKET_COLUMNS` 불변 확인 테스트 1개,
  기존 테스트는 전량 무수정
- [x] 검증: `pytest tests/core/test_capture.py tests/export/test_packets.py -v`
  → **60 passed**(기존 52 + 신규 8). `pytest tests/ -q -m "not live"` →
  **410 passed, 1 skipped, 12 deselected**(다른 병렬 작업자의 `aipt/gateway/`
  영역과 함께 그린, `aipt/core/capture.py`/`aipt/export/packets.py`/관련
  테스트 외 파일은 수정하지 않음)

## Phase 4.7 — B10 Docker 토폴로지 확장 (web + gateway + mock-server) — [x] 완료

- [x] `docker/Dockerfile.web` 신규: `python:3.12-slim` 2-stage 빌드
  (builder 스테이지에서 `native/cwnd_monitor.c`만 컴파일 → 런타임 스테이지에
  바이너리만 복사, tcp_congestion `b7cf75cb fix(docker)` 교훈 반영). 런타임
  스테이지엔 `iproute2`(netem/offload)+`tcpdump`(capture)만 설치(gcc 없음).
  `pip install ".[web,export]"`. `CMD uvicorn aipt.web.app:create_app
  --factory --host 0.0.0.0 --port 10000`. `NET_ADMIN`(cwnd netlink/offload)
  + `NET_RAW`(tcpdump) 요구사항을 헤더 주석으로 명시(Dockerfile.gateway와
  동일한 "capability 없으면 기능만 비활성, 이미지는 항상 빌드/기동" 계약)
- [x] `docker/Dockerfile.mockserver` 신규: `aipt.backends.mock.server.Server`
  구동용 경량 이미지(FastAPI/uvicorn 불필요, base `requests` 의존성만).
  `aipt.backends.mock.server`에 `__main__` 진입점이 없어서
  `docker/entrypoint_mockserver.py` 신규 작성 —
  `Server(host=MOCK_HOST, port=MOCK_PORT).serve_forever()`
- [x] `docker-compose.yml` 신규(AIPT 루트) — `mock-server`(8888, 호스트
  미노출, `expose`만) → `gateway`(8080, `NET_ADMIN`, `depends_on:
  mock-server`) → `web`(10000→`${WEB_HOST_PORT:-10000}`, `NET_ADMIN`+
  `NET_RAW`, `depends_on: gateway`, `./data/pcaps` 볼륨, `GEMINI_API_KEY`/
  `OPENAI_API_KEY`/`LOCAL_LLM_ENGINE_URL` 등 env passthrough,
  `GATEWAY_HOST=gateway`/`GATEWAY_PORT=8080` 예약). `local-llm` 서비스는
  이번 phase에서 생략(주석으로 `LOCAL_LLM_ENGINE_URL`을 외부 llama-server/
  vLLM으로 향하게 하면 local_llm backend 사용 가능하다고 안내). DESIGN.md
  4.7 "미해결 세부사항 1"(gateway L3 vs L4 포워딩 미정)을 compose 파일
  최상단 주석으로 명시 — 이번 phase는 컨테이너 토폴로지(서비스 이름,
  depends_on 순서, mock-server 비공개)까지만 구현, `gateway`→`mock-server`
  실제 트래픽 relay는 TODO로 남김
- [x] `.env.example` 신규(AIPT 루트) — `GEMINI_API_KEY`/`OPENAI_API_KEY`/
  `GATEWAY_IFACE`/`GATEWAY_PROFILE`/`GATEWAY_DELAY_MS` 등 netem 프리셋/
  `LOCAL_LLM_ENGINE_URL` 등 local_llm 옵션/`WEB_HOST_PORT`
- [x] `README.md`에 "Docker로 실행하기" 섹션 추가 — `docker compose
  up --build` 안내, 접속 URL `http://localhost:10000`, 3개 서비스 역할 요약
- [x] 검증: `docker compose config` → 정상 파싱(서비스 3개, `depends_on`
  순서 `web→gateway→mock-server` 확인). `docker compose build` → 3개
  이미지(`aipt-web`, `aipt-gateway`, `aipt-mock-server`) 전부 빌드 성공.
  `docker compose up -d`(호스트에 이미 다른 프로젝트가 10000 포트를 점유 중이라
  `WEB_HOST_PORT=10001`로 재시도) → 3개 컨테이너 전부 기동, `curl
  `localhost:10001/` → 200, 컨테이너 내부에서 `web→gateway:8080/health`
  (`netem_available: true`, `NET_ADMIN` 정상 동작 확인) 및
  `gateway→mock-server:8888/health` (`{"status": "ok"}`) 양쪽 모두 확인 후
  `docker compose down`으로 정리. `pytest tests/ -q -m "not live"` →
  **410 passed, 1 skipped, 12 deselected**(기존 베이스라인과 동일, Python
  코드는 전혀 건드리지 않았으므로 회귀 없음 확인)

## Network Gateway — L3 IP 포워딩 확정 구현 (DESIGN.md 4.7 "미해결 세부사항" 1, 2026-08-26 확정) — [x] 완료

DESIGN.md 4.7 미해결 세부사항 1이 "L3 라우팅"으로 확정됨에 따라, 기존
Gateway(netem 프로파일 API만 있던 상태)를 실제 커널 IP 포워딩 컨테이너로
구현했다. 애플리케이션 레벨 프록시/relay 코드는 만들지 않음 — 순수
`net.ipv4.ip_forward=1` + 분리 브리지 네트워크 + 명시적 `ip route add`.

- [x] `docker-compose.yml` — 최상단에 `networks:` 섹션 신규 추가:
  `net-client`(172.28.1.0/24, `web`+`gateway`), `net-backend`
  (172.28.2.0/24, `mock-server`+`gateway`). `web`은 `networks: [net-client]`
  만, `mock-server`는 `networks: [net-backend]`만, `gateway`는 두 네트워크
  모두에 고정 IP(`ipv4_address`, 기본 `172.28.1.2`/`172.28.2.2`, env로
  override 가능)로 연결. `gateway` 서비스에 `sysctls: [net.ipv4.ip_forward=1]`
  추가. `web`/`mock-server`에 `cap_add: [NET_ADMIN]` 추가(entrypoint의
  `ip route add`용). `web`/`mock-server` 각각에 `GATEWAY_PEER_SUBNET`/
  `GATEWAY_ROUTE_VIA` env 추가(entrypoint wrapper가 읽어 상대 네트워크로
  가는 경로를 gateway 경유로 명시 라우팅). 기존 "L3 vs L4 미정" TODO 주석은
  확정 설계 설명으로 교체
- [x] `aipt/gateway/netem_control.py` — 기존 `apply_profile(iface, profile)`은
  하위호환으로 그대로 유지. 신규 `apply_profile_both(client_iface,
  backend_iface, profile, dry_run=False)` 추가 — 두 인터페이스에 각각
  `apply_profile`을 호출(한쪽 실패해도 다른 쪽 계속 시도), 결과를
  `{"ok": bool(둘 다 성공해야 True), "client": {...}, "backend": {...},
  "reason": "client_iface=...: ...; backend_iface=...: ..."}`로 반환 — 어느
  쪽이 실패했는지 항상 구분 가능. `current_profile_both()`/`clear_both()`도
  같은 패턴으로 추가. 신규 `DEFAULT_CLIENT_IFACE`/`DEFAULT_BACKEND_IFACE`
  (env `GATEWAY_CLIENT_IFACE`/`GATEWAY_BACKEND_IFACE`, 기본 eth0/eth1) —
  Docker가 컨테이너에 여러 네트워크를 붙일 때 인터페이스 순서를 보장하지
  않으므로 하드코딩 대신 명시적 env로 받음. 기존 `DEFAULT_IFACE`/`GATEWAY_IFACE`는
  deprecated로 유지(하위호환)
- [x] `aipt/gateway/app.py` — `GET`/`POST /gateway/profile`이
  `apply_profile_both`/`current_profile_both`를 사용하도록 변경(양쪽
  인터페이스에 동일 프로파일 적용). 응답 shape이 `{"client": {...},
  "backend": {...}, ...}`로 바뀜(기존 단일 `{"profile": ..., "delay_ms":
  ...}` 평면 구조에서 변경 — 이 변경으로 `tests/gateway/test_app.py`의
  관련 테스트 최소 수정 필요, 아래 참고). `GET /health`에
  `client_iface`/`backend_iface`/`ip_forward_available`/`ip_forward_reason`
  필드 추가(기존 `iface` 필드는 하위호환으로 유지)
- [x] `aipt/gateway/forwarding.py` — 신규. `net.ipv4.ip_forward`가 실제로
  1인지 `/proc/sys/net/ipv4/ip_forward`를 직접 읽어 확인하는
  `read_ip_forward(path)`/`available(path)`/`status(path)`. sysctl이
  docker-compose 설정 누락/권한 부족 등으로 안 먹었을 때 예외로 죽지 않고
  `netem_control.available()`과 동일한 `(ok, reason)`/`{"ok": bool,
  "reason": ...}` 패턴으로 보고. `aipt.gateway.app`의 `GET /health`에서 사용
- [x] `docker/entrypoint_web.py` — 신규. `GATEWAY_PEER_SUBNET`/
  `GATEWAY_ROUTE_VIA` env를 읽어 컨테이너 시작 시 `ip route add
  <net-backend subnet> via <gateway의 net-client IP>` 실행 후
  `uvicorn aipt.web.app:create_app --factory ...`로 exec. env 미설정 시
  라우팅 설정을 건너뛰고 그대로 앱 기동(standalone/dev 실행 호환).
  `ip route add`가 이미 존재하는 경로("File exists")나 NET_ADMIN 부재로
  실패해도 컨테이너를 죽이지 않고 로그만 남김(netem_control과 동일한
  honesty-over-crash 원칙)
- [x] `docker/entrypoint_mockserver.py` — 기존 파일에 동일한 라우팅 로직
  추가(`GATEWAY_PEER_SUBNET`=net-client subnet, `GATEWAY_ROUTE_VIA`=gateway의
  net-backend IP), 그 다음 기존 `Server(...).serve_forever()` 그대로 실행.
  기존 동작(env 미설정 시 mock server만 기동)은 완전히 보존
- [x] `docker/Dockerfile.gateway` — `ENV GATEWAY_CLIENT_IFACE=eth0
  GATEWAY_BACKEND_IFACE=eth1` 추가(기존 `GATEWAY_IFACE=eth0`는 유지), 상단
  주석에 `net.ipv4.ip_forward=1` sysctl 요구사항 설명 추가
- [x] `docker/Dockerfile.web` — `COPY docker/entrypoint_web.py`,
  `CMD`를 `uvicorn ...` 직접 호출에서 `python entrypoint_web.py`로 변경(내부적으로
  동일한 uvicorn 커맨드를 `os.execvp`로 실행하므로 최종 프로세스는 동일)
- [x] `docker/Dockerfile.mockserver` — 주석만 갱신(entrypoint 파일 자체는
  기존과 동일 경로, 내용만 라우팅 로직 추가)
- [x] `tests/gateway/test_forwarding.py` — 신규, 8개 테스트. 실제
  `/proc/sys/net/ipv4/ip_forward`를 건드리지 않고 `tmp_path` scratch 파일로
  대체(1/0/파일없음/PermissionError 각 경로), `status()`의 dict shape 확인
- [x] `tests/gateway/test_netem_control.py` — 기존 테스트는 전량 무수정.
  `TestApplyProfileBoth` 클래스 신규 추가(8개 테스트): 양쪽 성공(커맨드
  6개=인터페이스당 3개 확인)/dry_run/tc 미설치 시 양쪽 reason 모두 포함/
  한쪽만 실패 시 실패한 쪽만 `reason`에 명시(다른 쪽 iface명은 안 들어감을
  확인)/`current_profile_both`/`clear_both`/`DEFAULT_CLIENT_IFACE`·
  `DEFAULT_BACKEND_IFACE` 상수 존재 확인
- [x] `tests/gateway/test_app.py` — `apply_profile_both` 응답 shape 변경에
  맞춰 3개 테스트 최소 수정(`test_get_profile_defaults_to_clean`,
  `test_post_then_get_reflects_applied_profile_when_tc_available`이
  `body["profile"]`→`body["client"]["profile"]`/`body["backend"]["profile"]`
  참조로 변경) + `test_health_ok`에 신규 필드
  (`client_iface`/`backend_iface`/`ip_forward_available`/`ip_forward_reason`)
  존재 확인 추가. `test_post_profile_preset`/`test_post_profile_custom`/
  `test_post_profile_unknown_name_rejected_without_500`/
  `test_post_profile_missing_field_is_422`는 응답 최상위 shape이
  `apply_profile_both`와 호환(여전히 최상위 `ok`/`profile` 키 존재)이라
  무수정
- [x] 검증: `pytest tests/gateway -q` 포함 `pytest tests/ -q -m "not live"`
  → **430 passed, 1 skipped, 12 deselected**(다른 병렬 작업자 영역 포함
  전체 그린, 회귀 없음). `from aipt.gateway import app, netem_control,
  profiles, forwarding` 임포트 스모크 통과.
  `netem_control.apply_profile_both("eth0","eth1", PRESETS["clean"],
  dry_run=True)` 실제 호출 → `ok=True`, 양쪽 `dry_run=True` 확인.
  `docker compose -f docker-compose.yml config` → 정상 파싱(네트워크 2개
  `net-client`/`net-backend`, `web`이 `net-client`에만, `mock-server`가
  `net-backend`에만, `gateway`가 양쪽에 고정 IP로 연결된 것 확인). 실제
  `docker compose up`으로 컨테이너 2개 띄워 `ip route`/포워딩 왕복 검증은
  이번 작업 범위 밖(다음 단계에서 사용자가 직접 확인 예정) — 코드/설정
  파일 정확성에 집중

## fixtures/perf.json 20턴 확장 + Mock 백엔드 재생 지원 (2026-08-27) — [x] 완료

`fixtures/perf.json`(구 `token_traffic/fixtures/perf.json`, ATLAS SRE 인시던트
대응 페르소나 시나리오, public_ai 백엔드용 `system`(리스트)+`steps` 스키마)을
10턴에서 20턴으로 확장하고, 신규로 답변(`answer`)까지 채워 Mock 백엔드가
그대로 재생할 수 있도록 로더를 확장했다.

- [x] `AIPT/fixtures/perf.json` — `steps` 10개→20개로 확장. 오전 인시던트
  (payments-api p99 breach → ledger 커넥션풀 고갈 → pool 260 증설 + us-east-1→
  us-west-2 10% 트래픽 shift → root cause: `ledger.async_retry_v2` 플래그가
  재시도마다 커넥션을 계속 잡고 있던 것 → 플래그 비활성화)를 turn 1~12에서
  마무리한 뒤, turn 13~20에서 같은 근무 시간대의 후속 인시던트(settlement-worker
  큐 증가 → query plan regression → 최장수명 replica 1대 drain_and_restart →
  isolated 확인 → 양쪽 인시던트 종합 SLO+최종 노트)로 대명사 참조를 유지하며
  자연스럽게 연결. `description` 필드도 "ten-turn"→"twenty-turn...spanning
  two related incidents in one shift"로 갱신. 각 `steps[i]`에 ATLAS 페르소나
  스타일(결론 우선 정량 수치 → 근거 → 다음 액션, mutating action은 항상
  현재값/목표값/롤백/관찰지표 명시) `answer` 필드 신규 추가(20개 전부,
  569~1,459바이트/턴, 평균 약 782바이트, 총 15,650바이트) — 이 필드는
  기존 `aipt.backends.public_ai.gemini.model_steps_from_response()` 등
  public_ai 로더에는 영향 없음(`text`만 참조), Mock 백엔드 재생 전용으로
  추가된 것
- [x] `AIPT/aipt/backends/mock/fixtures.py` — `load_qa_fixture()`가 기존
  `turns`(`{question,answer}` + `system_prompt` 문자열) 스키마에 더해
  `steps`(`{text,answer}` + `system` 리스트) 스키마도 인식하도록 확장.
  신규 `_turn_of_step()`(steps 항목을 `Turn`으로 변환, `answer` 누락 시
  "mock replay requires a canned answer" 명시적 에러) + `_system_prompt_of()`
  (`system`이 리스트면 `aipt.backends.public_ai.gemini`가 시스템 프롬프트를
  합치는 것과 동일하게 `"\n\n".join()`, 문자열이면 그대로, 없으면
  `system_prompt` 폴백). `turns` 키가 있으면 우선(하위호환), 없으면 `steps`로
  폴백하는 순서. `Fixture`/`Turn`/`load()`/`names()`/`byte_size_fixture()`
  등 기존 공개 API·시그니처는 무수정
- [x] 검증: `fixtures.load_qa_fixture("fixtures/perf.json")` → 20 turns
  정상 로드, `system_prompt` 20,653바이트(캐싱 실험 목적대로 4096 토큰
  이상 유지) 확인. `aipt.backends.mock.server.Server`를 실제로 기동해
  `/inference-mock?turn=0/9/19` 3개 호출 → 응답 JSON의 `answer` 필드에
  perf.json에 넣은 실제 텍스트가 그대로 반환됨을 wire 응답(657~1,514바이트)
  까지 확인. `pytest tests/ -q -m "not live"` → **433 passed, 1 skipped,
  12 deselected**(기존 430 passed 기준선 대비 회귀 없이 그린; `steps` 신규
  브랜치 자체를 커버하는 유닛 테스트는 아직 미작성 — 다음 후속 작업 후보)

## Run store 디스크 영속화 (2026-08-27, 남은 작업 리스트 #3) — [x] 완료

`aipt/web/store.py`가 프로세스 재시작 시 run 이력을 잃던 문제(Phase 4의
`TODO(persistence)`)를 해소했다. 인메모리 `OrderedDict`(`MAX_RUNS=50`)는
그대로 유지하고, 그 옆에 디스크 미러를 추가하는 방식.

- [x] `AIPT/aipt/web/store.py` — `save_run()`이 메모리 갱신 후 락 밖에서
  `<RUN_STORE_DIR>/<exec_id>.json`에 동기 파일 쓰기(`RUN_STORE_DIR` env,
  기본 `data/runs/`, `PUBLIC_AI_RECORDS_DIR`와 동일 패턴). `MAX_RUNS` 초과로
  evict된 run은 디스크 파일도 함께 삭제. 프로세스 최초 호출 시
  `_ensure_loaded_locked()`가 디스크를 1회 스캔해 `_runs`를 재구성
  (`saved_at` 기준 정렬 후 최신 `MAX_RUNS`개만). `get_run()`은 메모리 미스
  시 디스크 파일을 직접 읽는 폴백 추가. 디스크 I/O 실패(권한/디스크 풀)는
  로그만 남기고 절대 실행을 죽이지 않음(`aipt.gateway.netem_control`과
  동일한 honesty-over-crash 원칙). `clear()`는 테스트 격리를 위해 메모리+
  디스크 모두 비우고 재로드 플래그 리셋
- [x] `AIPT/aipt/web/app.py` — 모듈 docstring 갱신(영속화 완료 명시)
- [x] 기존 `tests/web/test_app.py`/`test_store.py`가 `tmp_path` 기반
  `RUN_STORE_DIR` 격리를 이미 쓰고 있어 추가 fixture 변경 불필요
- [x] 검증: `pytest tests/web -q` 그린(재시작 시나리오는 `_ensure_loaded_locked`
  단위 테스트로 커버)

## `/api/run/stream` SSE 엔드포인트 (2026-08-27, 남은 작업 리스트 #4) — [x] 완료

Phase 4에서 범위 밖으로 명시했던 스트리밍 진행상황 표시(현재는 폴링만
지원)를 구현. `POST /api/run`(블로킹, 전체 턴 완료 후 응답 1회)은 그대로
유지하고, `POST /api/run/stream`(SSE, 턴마다 이벤트)을 신규 추가.

- [x] `AIPT/aipt/web/routes_run.py` — 기존 `_run_conversation()`의 connect/
  send_turn 루프를 `_run_conversation_stream()` 제너레이터로 리팩터링:
  `{"type":"start",...}`(연결 직후 1회) → 턴마다 `{"type":"turn","turn":i,
  "record":{...}}` → 마지막 `{"type":"done","result":{...}}`(기존
  `_run_conversation()`이 반환하던 것과 동일한 run-document dict) 순서로
  yield. `_run_conversation()`은 이제 이 제너레이터를 드레인해서 `done`
  이벤트의 `result`만 반환하는 얇은 래퍼로 축소 — `/api/run` 및 기존
  테스트는 무수정으로 통과
- [x] 신규 `POST /api/run/stream` 라우트: `backend.send_turn()`이 blocking
  소켓 I/O이므로 이벤트 루프에서 직접 돌릴 수 없음 — `_drive_stream_to_queue()`
  헬퍼가 threadpool 워커에서 `_run_conversation_stream()`을 드레인하며
  각 이벤트를 `queue.Queue`에 push, `_STREAM_DONE` sentinel로 종료를
  알림. 라우트 코루틴은 `anyio.to_thread.run_sync(q.get)`로 큐를 한 개씩
  읽어(이것도 threadpool 슬롯을 쓰지 이벤트 루프를 막지 않음) SSE
  `data: <json>\n\n` 라인으로 변환. `run_store.save_run()`은 `done` 이벤트가
  나오는 시점에 큐잉 스레드 안에서 호출(`/api/run`과 동일한 저장 시점을
  스트리밍 구조에 맞게 이동)
  - EventSource가 아니라 POST+fetch/ReadableStream으로 소비해야 함
    (표준 `EventSource`는 GET+요청바디 없음만 지원, 이 라우트는 `RunRequest`
    JSON 바디가 필수) — 프론트 통합 시 유의사항으로 라우트 docstring에 명시
  - unknown backend(400 대신)나 local_llm `NotImplementedError`(501 대신)는
    스트림이 이미 200으로 시작했으므로 status code를 바꿀 수 없어
    `{"type":"error","error":...}` 단일 이벤트로 대체 보고
- [x] `AIPT/tests/web/test_app.py` — 신규 테스트 3개: mock 백엔드 3턴 실행
  시 `start`→`turn`×3→`done` 순서 및 `run_store` 저장 확인
  (`test_api_run_stream_mock_backend_emits_start_turn_done`), unknown
  backend가 200+단일 error 이벤트로 응답함(`test_api_run_stream_unknown_backend_emits_error_event_not_400`),
  local_llm이 error 또는 done 이벤트 중 하나로 안전하게 응답함(500 traceback
  유출 없음, `test_api_run_stream_local_llm_emits_error_or_done_event`)
- [x] `AIPT/aipt/web/app.py` — 모듈 docstring 갱신(`/api/run/stream` 반영,
  TODO 문구 제거)
- [x] 검증: `pytest tests/web/test_app.py -q` → **11 passed**(기존 8 +
  신규 3). `pytest tests/ -q -m "not live"` → **457 passed, 1 skipped, N
  deselected**(`tests/backends/local_llm/test_engine_live.py`의
  `@pytest.mark.live` 아닌 `test_local_llm_backend_against_real_engine` 1건은
  실제 로컬 llama-server/vLLM 프로세스가 안 떠 있는 이 샌드박스 환경 문제로
  기존부터 실패하던 것 — 이번 SSE 작업과 무관, `docker compose up`으로 실제
  엔진을 띄운 뒤에만 그린이 되는 사전 조건부 테스트임을 확인)


## "fixture" 용어 전면 리네임 → "record" (2026-08-27) — [x] 완료

사용자 지적: "fixture"라는 이름을 계속 쓰지 말고 "record"로 바꾸라는 지시를
앞서 "MIGRATION.md에 기록"으로 잘못 해석했던 것을 정정 — 실제로는 코드/파일명
자체의 "fixture" 네이밍을 "record" 계열로 바꾸라는 뜻이었다. 이 코드베이스의
Q&A 시나리오 개념("fixture")을 전부 "record"/"scenario record"로 리네임했다.
`pytest.fixture` 데코레이터(테스트 프레임워크 자체 기능, srv/ethtool 등 셋업)는
이 프로젝트 도메인 개념과 무관하므로 리네임 대상에서 제외했다. 기존
`aipt/backends/record.py`(`turn_record()`/`TurnExchange`, CSV 턴 로우 스키마)는
이름은 같지만 완전히 다른 개념이라 혼동 방지를 위해 새 클래스명은
`ScenarioRecord`로 지어 구분했다(단순 `Record`가 아님).

- [x] `AIPT/fixtures/` → `AIPT/records/`(디렉터리 rename, `perf.json`/`smoke.json`/
  `.gitkeep` 그대로 이동)
- [x] `AIPT/aipt/backends/mock/fixtures.py` → `AIPT/aipt/backends/mock/records.py`.
  `Fixture` 클래스 → `ScenarioRecord`, `FIXTURE_DIR` → `RECORD_DIR`,
  `load_qa_fixture()` → `load_scenario_record()`, `byte_size_fixture()` →
  `byte_size_scenario()`. `load()`/`names()`는 이름 유지(이미 도메인 중립적).
  docstring/주석 전체 "fixture"→"record"/"scenario record" 갱신
- [x] `AIPT/aipt/backends/mock/server.py` — `Server(fixture=...)` →
  `Server(record=...)`, 내부 헬퍼 `_fixture_answer()` → `_record_answer()`,
  `self.server.fixture` → `self.server.record`
- [x] `AIPT/aipt/backends/mock/conversation.py` — `MockBackend(fixture=...)` →
  `MockBackend(record=...)`, `self.fixture` → `self.record`,
  `DEFAULT_MODEL = "mock-fixture"` → `"mock-record"`, progress 이벤트의
  `arm="fixture"` → `arm="record"`
- [x] `AIPT/aipt/backends/mock/replay.py` — `Fixture`/`Turn` import를
  `ScenarioRecord`/`Turn`으로, 반환 타입·docstring 전부 갱신
  (`from_capture_doc`/`from_capture_file`/`from_public_ai_record_doc`)
- [x] `AIPT/aipt/backends/mock/__init__.py` — 모듈 docstring의
  `fixtures.py`/`fixture 답변`/`replay fixture` 언급을 `records.py`/
  `scenario-record 답변`/`replay record`로 갱신
- [x] `AIPT/aipt/backends/public_ai/recorder.py` — `FixtureWriter` 클래스 →
  `RecordWriter`(속성/메서드는 무수정: `system`/`steps`/`add()`/`to_dict()`/
  `write()`), `recording_backend(writer: FixtureWriter, ...)` 시그니처도
  `RecordWriter`로 갱신. docstring의 "fixture format"/"perf.json shape"
  언급을 "scenario record" 표현으로 갱신
- [x] `AIPT/aipt/web/routes_run.py` — `MockBackend(fixture=...)` 호출부를
  `record=...`로, 내부 헬퍼 `_load_record_fixture()` → `_load_record_scenario()`,
  로컬 변수 `fixture` → `scenario_record`, `public_ai_recorder.FixtureWriter` →
  `RecordWriter`, 주석의 "fixtures/ 트리" 언급을 "records/ 트리"로 갱신
  (이 파일은 사이드 작업자가 동시 편집 중이던 `RunRequest`/`_resolve_turns`
  로직 자체는 건드리지 않고 네이밍만 교체)
- [x] `AIPT/aipt/web/routes_runs.py` — docstring 1곳("raw persisted fixture
  JSON") 표현 갱신
- [x] 테스트 전량 갱신: `tests/backends/mock/test_fixtures.py` →
  `tests/backends/mock/test_records.py`(리네임 + `records` 모듈 API로 재작성,
  `steps`-shaped 레코드 로딩 신규 테스트 3개 추가), `test_server.py`
  (`fixture_srv` → `record_srv`, `Server(record=...)`), `test_conversation_live.py`
  (`Fixture`→`ScenarioRecord`, `MockBackend(record=...)`),
  `tests/backends/public_ai/test_recorder.py`(`FixtureWriter`→`RecordWriter`,
  테스트 함수명도 `test_record_writer_*`로), `tests/web/test_public_ai_records.py`
  (`FixtureWriter`→`RecordWriter` 1곳), `tests/backends/public_ai/test_gemini.py`
  (`FIXTURE` 경로 상수를 `records/perf.json`로 갱신 — 디렉터리 rename으로 깨졌던
  경로를 바로잡음)
- [x] 검증: `aipt.backends.mock.records.load_scenario_record("records/perf.json")`
  → 20 turns 정상 로드 재확인, `Server(record=...)` 실제 기동 후
  `/inference-mock?turn=0` 호출 → `answer` 필드 정상 반환 재확인.
  `pytest tests/ -q -m "not live"` → **446 passed, 1 skipped, 12 deselected**
  (이 작업 시작 시점 기준선 433 대비, 사이드 작업자가 동시에 추가한
  `tests/web/test_store.py`(3개) 포함 전체 그린, 이번 리네임으로 인한 회귀 없음).
  전역 검색으로 `pytest.fixture` 데코레이터를 제외한 도메인 "fixture" 잔존
  참조가 코드에 없음을 재확인(문서의 과거 이력 서술 및 옛 `token_traffic/
  fixtures/perf.json` 원본 경로 언급만 의도적으로 보존)


