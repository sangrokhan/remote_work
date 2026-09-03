# AIPT — Design Document

**AIPT** (AI Protocol Traffic lab) merges two previously separate projects that
measure the same underlying phenomenon — TCP behaviour under LLM multi-turn
traffic patterns — from two different angles:

| 원 프로젝트 | 관측 대상 | 트래픽 소스 |
|---|---|---|
| `token_traffic` | 요청/응답 **byte·token·latency** (arm 간 비교) | 실제 Gemini/OpenAI API — 과금 발생 |
| `tcp_congestion` | idle 구간 후 **TCP cwnd 리셋** (slow-start-after-idle) | 로컬 mock 서버 — 무료, 재현 가능 |

`tcp_congestion`은 실제로 `token_traffic/core/{cwnd,capture,offload}.py`와
`native/cwnd_monitor.c`를 이식(포크)해서 만들어졌다 — 소스 코드 주석에
"Adapted from token_traffic/core/cwnd.py" 라고 명시되어 있다. 즉 이미 하나의
계보였던 코드가 두 개의 독립 디렉터리로 갈라져 있었고, 이번 작업은 그것을
원래대로 하나의 코어로 되돌리는 일에 가깝다.

## 1. 현황 분석 요약

### 1.1 코드 규모

| | token_traffic | tcp_congestion |
|---|---|---|
| 웹 프레임워크 | Flask (`core/app.py`, 365줄) | FastAPI (`tcp_congestion/app.py`, 209줄) |
| 코어 모듈 | 16개, ~3,600줄 | 9개, ~1,340줄 |
| 프로바이더 어댑터 | gemini.py(715) + openai.py(634) | 없음 (mock 서버 자체 구현) |
| 테스트 | 19개 파일 | 13개 파일 |
| 컨테이너 구성 | 단일 Dockerfile (앱만) | Dockerfile.client + Dockerfile.server (실제 소켓 페어) |
| native C | `native/cwnd_monitor.c` | 동일 파일 (완전 동일, diff 없음) |

### 1.2 모듈별 관계 (diff 기반 확인)

| 모듈 | 관계 | 병합 방침 |
|---|---|---|
| `native/cwnd_monitor.c` | **완전 동일** | 그대로 1개만 유지 |
| `cwnd.py` | tcp_congestion판이 token_traffic판의 단순화 파생 (`core.config`/`core.wire` 의존 제거, `announce(sock)` API로 정리) | **tcp_congestion의 단순화된 인터페이스**를 채택하되, token_traffic의 상세 docstring(설계 근거)과 `dumps`/`exact_queries` 계측 필드를 병합 |
| `capture.py` | tcp_congestion판은 token_traffic판에서 AppArmor 회피 로직 등을 제거하고 "run 1개당 pcap 1개"로 단순화 | token_traffic의 **AppArmor 감지 로직은 반드시 보존** (실제로 시간 낭비했던 이슈, 주석에 경고 있음). label 파라미터를 일반화해서 두 lab 모두 지원 |
| `offload.py` | 사실상 같은 기능, env var 네이밍만 다름 (`TRAFFIC_PCAP_NO_OFFLOAD` vs `NIC_OFFLOAD_DISABLE`) | 통합 후 **두 이름 모두 지원**(alias) — 기존 docker-compose.yml/문서 호환. **(T7, 2026-09-03 갱신)** 병합 직후에는 feature-set(capture-time 3개 vs entrypoint-time 5개)과 복원 정책(capture-time만 복원, entrypoint-time은 미복원)이 서로 달라 이 서술이 실제로는 참이 아니었다(`docs/audit-2026-09-02/core.md` §4.2). T7에서 두 API 모두 `FEATURES = (tso, gso, sg, gro, lro)` 5개로 통일하고, 기존 `build_commands()`/`apply()`(무복원, 하위호환 유지)와 별개로 `Toggle` 클래스를 추가해 entrypoint-time에도 `Window`와 동일한 관찰→선택적 해제→복원 계약을 제공함으로써 이 표의 서술을 실제로 참이 되게 했다 |
| `export.py` | 서로 다른 레코드 스키마(제공자별 arm vs 턴별 요약)라 병합이 아니라 **공존** | `export/` 서브패키지로 분리, `records_csv()`(external_api용), `turns_csv()`(synthetic_mock용) 공존 |
| `probe.py`, `netem.py`, `congestion.py`, `tcpinfo.py`, `server.py`, `conversation.py` | tcp_congestion 전용, token_traffic에 대응물 없음 | synthetic_mock lab 전용으로 그대로 이관 |
| `wire.py`, `streaming.py`, `call.py`, `record.py`, `metrics.py`, `store.py`, `scenario.py`, `cachebust.py`, `providers/*` | token_traffic 전용, tcp_congestion에 대응물 없음 | external_api lab 전용으로 그대로 이관 |

## 2. 목표 폴더 구조

```
AIPT/
├── DESIGN.md                     # 이 문서
├── MIGRATION.md                  # 파일 단위 이관 체크리스트 (실행 단계에서 갱신)
├── README.md                     # 프로젝트 개요, 빠른 시작
├── pyproject.toml                # 단일 의존성 정의 (fastapi, uvicorn, requests, google-genai, openai 등)
├── Makefile                      # native C 빌드, 테스트 실행
│
├── aipt/                         # 설치 가능한 패키지 루트
│   ├── __init__.py
│   ├── core/                     # 두 lab이 공유하는 측정 인프라
│   │   ├── config.py             # env 플래그 판독 (양쪽 통합)
│   │   ├── cwnd.py               # 연속 netlink cwnd 모니터 (통합판)
│   │   ├── capture.py            # tcpdump 캡처 (AppArmor 감지 포함, label 일반화)
│   │   ├── offload.py            # NIC offload 토글 (env alias 지원)
│   │   ├── tcpinfo.py            # 1회성 TCP_INFO 스냅샷 (경량 대안)
│   │   ├── wire.py               # 소켓 바이트 카운터 (external_api 전용, 재사용 가능하게 core에 위치)
│   │   ├── streaming.py          # SSE 리더 (external_api 전용)
│   │   ├── record.py             # 레코드 스키마 (external_api 전용)
│   │   └── export_base.py        # CSV export 공통 유틸 (컬럼 정의, writer 헬퍼)
│   │
│   ├── providers/                # external_api 전용 — 그대로 이관
│   │   ├── base.py
│   │   ├── gemini.py
│   │   └── openai.py
│   │
│   ├── labs/
│   │   ├── external_api/         # 구 token_traffic 도메인 로직
│   │   │   ├── call.py           # 1~2-pass HTTP 호출
│   │   │   ├── cachebust.py
│   │   │   ├── metrics.py
│   │   │   ├── runner.py         # 여러 (provider, arm) 조합 실행
│   │   │   ├── scenario.py       # fixtures/perf.json 재생
│   │   │   ├── store.py          # 런 저장 + 보존 정책
│   │   │   └── export.py         # records.csv / summary.csv
│   │   │
│   │   └── synthetic_mock/       # 구 tcp_congestion 도메인 로직
│   │       ├── server.py         # HTTP keep-alive mock 서버
│   │       ├── probe.py          # idle 구간 RTT HTTP PING
│   │       ├── conversation.py   # 누적 컨텍스트 멀티턴 시나리오
│   │       ├── congestion.py     # 알고리즘(cubic/reno/bbr/vegas) + qdisc 점검
│   │       ├── netem.py          # tc netem 지연 주입
│   │       └── export.py         # cwnd.csv / turns.csv
│   │
│   └── web/                      # 단일 FastAPI 앱
│       ├── app.py                # create_app(): 루트에 랜딩 페이지, /external-api, /synthetic-mock 마운트
│       ├── routes_external_api.py    # 구 core/app.py(Flask) 라우트 → FastAPI로 포팅
│       ├── routes_synthetic_mock.py  # 구 tcp_congestion/app.py 라우트 이관
│       ├── templates/
│       │   ├── index.html            # 랜딩: 두 lab 선택
│       │   ├── external_api/index.html
│       │   └── synthetic_mock/index.html
│       └── static/
│           ├── app.js               # 구 token_traffic 정적 자산
│           └── style.css
│
├── native/
│   └── cwnd_monitor.c             # 유일한 사본 (양쪽 동일 확인됨)
│
├── fixtures/
│   └── perf.json                  # external_api 시나리오 fixture
│
├── docker/
│   ├── Dockerfile.web             # 웹앱 (native C 빌드 스테이지 포함)
│   ├── Dockerfile.mockserver      # synthetic_mock의 상대편 서버 컨테이너
│   └── docker-compose.yml         # web + mockserver (+ netem용 NET_ADMIN/NET_RAW)
│
├── tests/
│   ├── core/                      # cwnd, capture, offload, wire, streaming 등
│   ├── providers/                 # gemini, openai
│   ├── labs/external_api/
│   ├── labs/synthetic_mock/
│   └── web/
│
└── docs/
    ├── core-contracts.md          # token_traffic/docs/core-contracts.md 갱신판
    └── outputs.md                 # 두 lab의 산출물 포맷 통합 문서
```

## 3. 웹 UI 통합 방침 (FastAPI 단일화)

- `token_traffic/core/app.py`는 **Flask**, `tcp_congestion/app.py`는 **FastAPI**.
  하나로 합치기로 결정했으므로 Flask 라우트 전체를 FastAPI로 포팅한다.
- 라우트 이관 시 매핑 원칙:
  - Flask `@app.route("/api/run", methods=["POST"])` → FastAPI `@router.post("/api/run")`
  - Flask의 `request.get_json()` → FastAPI Pydantic 모델 또는 `await request.json()`
  - Flask의 동기 blocking 실행(외부 API 호출은 원래도 동기)은 FastAPI에서 `run_in_threadpool`로 감싸 이벤트 루프 블로킹 방지 (synthetic_mock의 `conversation.run()`도 동일하게 스레드풀 위임)
- URL 네임스페이스: `/external-api/*` (구 token_traffic), `/synthetic-mock/*` (구 tcp_congestion). 루트 `/`는 두 실험을 선택하는 랜딩 페이지.
- 정적/템플릿 파일은 하위 폴더로 분리해서 두 UI가 서로의 CSS/JS를 침범하지 않게 한다.
- 다운로드 엔드포인트(`/api/download/*`, `/api/runs/<id>/*`)는 각 lab 아래로 이관하되 경로 프리픽스만 붙인다 — 응답 스키마는 변경하지 않는다 (외부에서 참조 중일 수 있음).

### 3.1 시각 디자인 방침 (2026-09-02 개정)

초기 구현(`aipt/web/static/style.css`)은 "화려하게 만들 필요 없음 — 기능하는
최소 UI"를 원칙으로 시작했으나, 실제 사용 중 비주얼(색상·타이포·여백)이
성의없어 보인다는 피드백에 따라 이 원칙을 **폐기**한다. 이 화면은 마케팅
랜딩(Decide/Learn 성격)이 아니라 **Configure + Operate + Monitor**가 섞인
실무 운영 도구이므로, 히어로/카드-그리드형 장식 대신 다음 기준으로 다시
설계한다:

- 다크 배경 + 모노스페이스 accent(라벨/코드성 값)로 개발자 도구 톤 유지.
- 카드/폼/테이블의 밀도와 정렬을 높여 상태·에러·값이 한눈에 들어오게 함.
- 색은 상태 의미론(ok/warn/pending)에만 사용하고 장식적 그라디언트는 배제.
- `app.js`가 참조하는 모든 id/class 셀렉터는 그대로 유지 — 로직 변경 없이
  비주얼/레이아웃(CSS+HTML)만 개선.

## 4. Docker/인프라 통합 방침

- `token_traffic`은 단일 컨테이너(앱만, 외부 API 호출), `tcp_congestion`은 client+server 페어 컨테이너.
- 병합 후: `docker-compose.yml`에 **web**(FastAPI 앱, external-api 기능 포함) + **mockserver**(synthetic_mock의 keep-alive 서버) 2개 서비스로 구성.
- `native/cwnd_monitor.c`는 web 컨테이너 빌드 시 1회만 컴파일 (멀티스테이지 빌드로 분리 — tcp_congestion의 `b7cf75cb fix(docker)` 커밋 교훈 반영).
- 포트: mockserver 기본 8888 유지, 웹UI 기본 10000 유지 (tcp_congestion 관례), external-api 기능은 같은 웹앱 프로세스 내 라우트이므로 별도 포트 불필요.
- `CLIENT_NETEM_DELAY_MS` / `SERVER_NETEM_DELAY_MS` env는 그대로 유지.
- NIC offload env는 `NIC_OFFLOAD_DISABLE`(기존 tcp_congestion 이름)을 정식으로 채택하고 `TRAFFIC_PCAP_NO_OFFLOAD`를 deprecated alias로 지원.

## 4.5 아키텍처 개정 v2 — 3-Backend 공통 클라이언트 구조

**배경**: §1~4의 최초 설계는 "external_api lab / synthetic_mock lab, 두 실험을
나란히 두고 core만 공유"하는 구조였다. 사용자 피드백에 따라 이를
**"클라이언트가 3개 backend 중 하나를 선택해서 동일한 인터페이스로 호출"**하는
구조로 개정한다. 기존 §2 폴더 구조의 `aipt/labs/{external_api,synthetic_mock}`
분리는 **기능(무엇을 측정하나) 기준**이었는데, 이제는 **backend(무엇을
상대하나) 기준**으로 재편한다.

```
Client 측 (측정 로직: cwnd/capture/stats export — 3개 backend에 공통 적용)
  └── Backend 프로토콜 (신규 추상화)
        ├── PublicAIBackend   — Gemini / ChatGPT      (기존 token_traffic providers/* 재사용)
        ├── MockBackend       — 고정 JSON I/O 재생      (기존 tcp_congestion server.py 확장)
        └── LocalLLMBackend   — 표준 서빙엔진 + 자체 프록시 (신규 구현)
```

### 확정된 설계 결정

| 결정 사항 | 확정 내용 |
|---|---|
| 로컬 LLM 스택 | **llama.cpp/vLLM 같은 표준 서빙 프레임워크를 그대로 사용**하고, 그 앞단에 자체 프록시/게이트웨이를 둔다. 프록시가 HTTP 신기능 확장 지점을 담당하고, 토큰 생성 자체는 표준 엔진에 위임 — 추론 엔진을 직접 재구현하지 않는다 |
| 신규 HTTP 실험 범위 | **이번 AIPT 병합에는 포함하지 않는다.** `Backend`/프록시 인터페이스에 transport 확장 지점(`transport: "http1"` 슬롯)만 마련해둔다 |
| Mock 재생 충실도 | **바이트 패턴만 재현**한다. 지연시간(추론 대기)은 실측값을 그대로 재생하지 않고, 기존 tcp_congestion처럼 설정값(`inference_delay`)으로 별도 제어 — 재생 로직의 복잡도를 낮춘다 |

### 폴더 구조 변경 (§2 대비 diff)

```diff
 aipt/
   core/                        # 변경 없음: cwnd, capture, offload, tcpinfo
+  backends/                    # 신규 — 구 aipt/labs/*, aipt/providers/* 를 흡수
+    base.py                    # Backend 프로토콜: connect/send_turn/close, transport 확장 슬롯 포함
+    public_ai/
+      gemini.py                # 구 aipt/providers/gemini.py
+      openai.py                # 구 aipt/providers/openai.py
+      recorder.py              # 신규 — 실측 요청/응답 캡처 (B2)
+    mock/
+      server.py                # 구 tcp_congestion server.py, 고정 byte → fixture 재생으로 확장 (B1)
+      fixtures.py              # 신규 — fixture 포맷 로더 (dummy bytes ↔ 실측 재생 데이터 공통 스키마)
+      replay.py                # 신규 — 실측 캡처 데이터를 fixture로 변환해 재생 (B3, 바이트 패턴만)
+    local_llm/
+      gateway.py                # 신규 — 표준 서빙엔진 앞단 프록시 (B4)
+      engine_adapter.py         # 신규 — llama.cpp/vLLM 등 백엔드 엔진 선택 어댑터
-  labs/
-    external_api/...            # → backends/public_ai/ 로 흡수
-    synthetic_mock/...          # → backends/mock/ 로 흡수
-  providers/...                 # → backends/public_ai/ 로 흡수
   export/                       # 신규 통합 — 구 aipt/labs/*/export.py 통합 (§4.6 참고)
     turns.py                    # 턴 단위 CSV (bytes/tokens/latency/goodput)
     packets.py                  # 신규 — pcap → 패킷 간격 CSV (B6)
     connection.py               # cwnd.csv (기존 그대로)
   web/                          # 변경 없음: FastAPI 단일 앱, 다만 라우트는 backend 선택 파라미터로 통합
```

`aipt/labs/`, `aipt/providers/`는 폐기하고 `aipt/backends/`로 통합한다.
MIGRATION.md의 Phase 2/3 대상 파일들은 목적지 경로만 `backends/public_ai/`,
`backends/mock/`으로 바뀌고 이관 대상 자체는 동일하다.

## 4.6 통계/CSV 3-레이어 통합 (3개 backend 공통)

이전 설계는 lab별로 export.py를 유지했으나, 3-backend 공통 구조에서는
**어떤 backend를 상대하든 동일한 3-레이어 CSV 세트**를 내야 한다.

| 레이어 | 파일 | 내용 | 출처 | 상태 |
|---|---|---|---|---|
| 1. Connection-level | `cwnd.csv` | tick별 snd_cwnd/rtt/delivery_rate | 기존 `cwnd.py` (양쪽 동일) | 이관만 하면 됨 |
| 2. Turn-level | `turns.csv` | prompt_bytes, wire_sent/recv, tokens, ttft/ttlt/turn_end, **goodput(신규)** | token_traffic records.csv + tcp_congestion turns.csv 병합 | **신규 병합 필요** |
| 3. Packet-level | `packets.csv` | 패킷 간격(inter-arrival gap), 패킷 크기 분포 — pcap에서 추출 | 없음 (지금은 pcap 저장만 함) | **완전 신규** |

세 CSV + pcap을 기존 `bundle.zip` 방식으로 묶어서 다운로드하는 구조는 유지.

## 5. 신규/머지 작업 리스트업 (최종)

### A. 머지 작업 (기존 코드 재사용/재구성)

| # | 작업 | 원본 | 목적지 |
|---|---|---|---|
| A1 | `Backend` 프로토콜 정의 | `token_traffic/providers/base.py` 일반화 | `aipt/backends/base.py` |
| A2 | Gemini/OpenAI 어댑터 이관 | `token_traffic/providers/{gemini,openai}.py` | `aipt/backends/public_ai/{gemini,openai}.py` |
| A3 | Mock 서버 이관 (확장은 B1) | `tcp_congestion/tcp_congestion/server.py` | `aipt/backends/mock/server.py` |
| A4 | cwnd 연속 모니터링 통합 | `cwnd.py` (양쪽) | `aipt/core/cwnd.py` |
| A5 | tcpdump 캡처 통합 | `capture.py` (양쪽, AppArmor 감지 보존) | `aipt/core/capture.py` |
| A6 | 턴별 트래픽 통계 정의 일반화 | `token_traffic/core/{wire,record,metrics}.py` | `aipt/core/` + `aipt/export/turns.py` |
| A7 | offload/netem 이관 | `offload.py`, `netem.py` | `aipt/core/` |

### B. 신규 구현 작업

| # | 작업 | 내용 | 비고 |
|---|---|---|---|
| B1 | Mock fixture 포맷 설계 + 구현 | 고정 byte dummy → Q&A 쌍 JSON 시나리오 재생 (`token_traffic/fixtures/perf.json` 개념 확장) | `aipt/backends/mock/fixtures.py` |
| B2 | 실측 데이터 recorder | Public AI backend 호출 시 request/response 원문을 B1 포맷으로 저장 | `aipt/backends/public_ai/recorder.py`. API 키 등 민감정보 마스킹 필요 |
| B3 | Mock replay (바이트 패턴만) | B2로 캡처한 실측 데이터를 Mock backend가 재생. **지연은 재현 안 함**, 설정값(`inference_delay`)으로 별도 제어 | `aipt/backends/mock/replay.py` |
| B4 | LocalLLMBackend (표준 엔진 + 자체 프록시) | llama.cpp/vLLM을 서빙 엔진으로 세우고, 앞단에 자체 게이트웨이를 둬서 HTTP 신기능 실험 지점 마련 | `aipt/backends/local_llm/{gateway,engine_adapter}.py`. 엔진 선택은 `huggingface-hub`/`llama-cpp`/`serving-llms-vllm` 스킬 활용 |
| B5 | Transport 확장 슬롯 | `Backend`/게이트웨이에 `transport` 파라미터 슬롯만 마련, 구현은 후속 프로젝트로 이관 | 이번 범위: 인터페이스 설계만 |
| B6 | 패킷 간격 통계 (`packets.csv`) | pcap 파싱(`dpkt`/`scapy`)으로 inter-arrival gap, 패킷 크기 분포 계산 | `aipt/export/packets.py` — 신규 의존성 추가 필요 |
| B7 | Goodput 계산 | 기존 wire_sent/recv + 마크(req_sent_ms~turn_end_ms)로 턴별 goodput 산출 | `aipt/export/turns.py`에 컬럼 추가 |
| B8 | 통합 CSV 스키마 확정 | §4.6의 3-레이어를 실제 컬럼 단위로 확정 | Phase 진행 중 `docs/outputs.md`에 기술 |

### C. 폐기/대체

- tcp_congestion의 고정 byte dummy 생성 로직 → B1 완료 시 대체 (다만 "순수 바이트 크기만 스윕"하고 싶은 실험을 위해 옵션으로는 유지)
- `aipt/labs/`, `aipt/providers/` 네임스페이스 → `aipt/backends/`로 전량 이전, 폐기

## 4.7 Network Gateway 컨테이너 — mock/local_llm 경로의 지연/손실 모사

**배경**: `PublicAIBackend`는 이미 실제 인터넷을 거치므로 진짜 RTT/지연/손실을
겪는다. 반면 `MockBackend`와 `LocalLLMBackend`는 지금까지 컨테이너 간 직결
(loopback 수준)이라 "완벽한 네트워크"에서만 측정되어 왔다 — idle-reset이나
cwnd 실험의 사실성이 떨어지는 지점이다. tcp_congestion에 있던
`CLIENT_NETEM_DELAY_MS`/`SERVER_NETEM_DELAY_MS`도 컨테이너 자체의 인터페이스에
건 ad-hoc 설정이었을 뿐, 독립된 아키텍처 구성요소는 아니었다.

**결정**: 클라이언트(측정 코드)가 **mock 또는 local_llm backend를 상대할 때는
반드시 Gateway 컨테이너를 경유**하도록 만든다. Gateway는 두 종단(client ↔
mock-server / client ↔ local-llm) 사이에 위치하는 별도 컨테이너로, 내부에
지연·손실·지터·재정렬 설정을 갖고 트래픽에 주입한다. `PublicAIBackend`는
이미 실제 인터넷이 이 역할을 하므로 Gateway를 경유하지 않는다.

```
Client (측정 코드: cwnd/capture/export — 공통)
  ├── PublicAIBackend ────────────────────────────► 실제 인터넷 (Gemini/OpenAI)
  │                                                    (이미 실제 네트워크 특성 보유)
  │
  └── ┌─────────────────────────────┐
      │  Network Gateway 컨테이너      │   tc netem 기반: delay/jitter/loss/reorder/duplicate
      │  (aipt/core/netem.py 승격)    │   런타임 프로파일 전환 지원 (§구성 참고)
      └──────────────┬────────────────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
  MockBackend                LocalLLMBackend
  (mock-server 컨테이너)      (프록시/게이트웨이 + 표준 서빙엔진 컨테이너)
```

### 구성

| 항목 | 방침 |
|---|---|
| 위치 | `aipt/gateway/` 신규 패키지 + `docker/Dockerfile.gateway` — 별도 컨테이너로 배포 |
| 트래픽 제어 수단 | `tc qdisc netem` (지연/지터/손실/재정렬/중복), 기존 `aipt/core/netem.py`를 얇은 wrapper가 아니라 **Gateway 컨테이너의 제어 루프 본체**로 승격 |
| 적용 대상 | `MockBackend`, `LocalLLMBackend`만 경유. `PublicAIBackend`는 경유하지 않음 (실제 인터넷이 이미 그 역할) |
| 설정 방식 | (a) 컨테이너 기동 시 env 프리셋(`GATEWAY_DELAY_MS`, `GATEWAY_JITTER_MS`, `GATEWAY_LOSS_PCT`, `GATEWAY_REORDER_PCT` 등, 기존 `CLIENT_NETEM_DELAY_MS` 계열 대체) — (b) **런타임 API** `POST /gateway/profile` 로 실행 중 프로파일 교체 지원 (실험 웹 UI에서 "wired 프로파일", "wireless 프로파일" 같은 프리셋 선택 가능하게) |
| 웹 UI 연동 | `aipt/web` 실험 설정 폼에 "Network profile" 드롭다운 추가 (`clean`/`wired`/`wireless`/`custom`, 2026-09 재설계 — 값 근거는 ARCHITECTURE.md §4.2 참고) — 선택값을 Gateway의 `/gateway/profile`로 전달 |
| 계측과의 관계 | Gateway 자체는 순수 L3/L4 트래픽 셰이핑만 하고 애플리케이션 로직에는 개입하지 않는다. `aipt/core/cwnd.py`/`capture.py`가 관찰하는 소켓은 client↔gateway 구간이며, Gateway↔backend 구간은 별도 관찰 대상이 아니다 (필요시 후속 확장) |
| Docker 토폴로지 | `docker-compose.yml`에 서비스 3~4개: `web`(client), `gateway`, `mock-server`, `local-llm-gateway`(+엔진). `web`은 `gateway`의 주소로 mock/local_llm에 접속하고, `gateway`가 `mock-server`/`local-llm-gateway`로 forward하며 그 경로에 netem을 건다 |
| PublicAI와의 대칭성 참고 | 이 구조 덕분에 3개 backend 모두 "클라이언트가 겪는 경로 특성"이 명시적으로 기술된다: PublicAI=실제 인터넷, Mock/LocalLLM=Gateway가 흉내낸 네트워크. 지금까지처럼 "로컬은 공짜 네트워크"라는 암묵적 가정이 제거된다 |

### 미해결 세부사항 (Phase 진행 중 확정)

1. ~~Gateway가 L4(TCP 프록시)로 동작할지, L3(라우팅 경유, netem만 인터페이스에 적용)로 동작할지~~
   **확정 (2026-08-26): L3 라우팅.** Gateway는 TCP 상태를 보지 않는 순수 IP
   포워딩 컨테이너로 동작한다 — 애플리케이션 레벨 프록시/relay 코드를 만들지
   않고, 커널의 IP forwarding(`net.ipv4.ip_forward=1`)과 두 개의 분리된
   Docker 브리지 네트워크(`net-client`, `net-backend`)로 구현한다.
   - `web`은 `net-client`에만 속하고, `net-backend` 서브넷으로 가는 경로를
     Gateway의 `net-client` IP를 통해 명시적으로 라우팅한다 (`ip route add`).
   - `mock-server`(및 향후 `local-llm`)는 `net-backend`에만 속하고, 마찬가지로
     `net-client` 서브넷 경로를 Gateway 경유로 라우팅한다 (왕복 트래픽이 반드시
     Gateway를 통과하게 하기 위함 — 이게 없으면 응답 패킷이 Gateway를
     우회해서 되돌아갈 수 있다).
   - Gateway 자체는 두 네트워크 모두에 속하며, `net.ipv4.ip_forward=1` +
     `NET_ADMIN`으로 커널 레벨 포워딩만 수행한다. TCP 페이로드나 헤더를
     들여다보지 않는다 — 순수 L3/L4-무관 패킷 라우팅.
   - `tc netem`은 **client-facing leg에만** (egress 직접 + ingress는 IFB
     리다이렉트, 2026-09 client-link-only 재설계) 적용된다 — 왕복
     (request/response) 모두 client-facing leg를 겪게 하기 위함.
     backend-facing leg는 사용자 선택과 무관하게 항상 고정된
     `ETHERNET_BASELINE`(사실상 무손상)만 적용 — Gateway↔backend가 실제로는
     같은 데이터센터/호스트 내부의 Ethernet 홉이라는 토폴로지를 반영한
     것. 자세한 근거와 IFB 구현은 ARCHITECTURE.md §4.2 참고. (이전
     2026-08-26 확정 설계는 양쪽 인터페이스에 동일 프로파일을 걸었으나,
     실제 토폴로지와 맞지 않는다는 지적으로 재설계됨.)
2. 런타임 프로파일 전환 시 기존 연결(keep-alive)에도 즉시 반영되는지 — tc netem은
   인터페이스 단위라 기존 연결에도 즉시 적용됨 (재현성에 유리, 확정 유지).
3. LocalLLMBackend(B4)의 "자체 프록시"(engine gateway, 애플리케이션 레벨)와
   이 Network Gateway(L3, 커널 레벨)는 여전히 서로 다른 컴포넌트 —
   `client → Network Gateway(L3 forward) → engine gateway(애플리케이션 프록시) →
   서빙 엔진` 순서로 체인된다.

## 4.7.1 실행 결과 저장 정책 (확정, 2026-09-02 개정 — 모든 backend run 디스크 영속화가 기본 정책)

기존 §6 미해결 결정 5번("data/ 저장 위치")을 아래로 확정한다(2026-08-27 "Run
store 디스크 영속화" 작업으로 실제 동작이 이렇게 바뀌었고, 2026-09-02부로 이
절을 그 실제 동작에 맞춰 정식 개정함 — MIGRATION.md, docs/audit-2026-09-02/web.md
§5.2 참고):

- **`public_ai`/`mock`/`local_llm` 3개 backend의 모든 run이 예외 없이 디스크에
  영속 저장되는 것이 기본 정책이다.** `aipt/web/routes_run.py`는 backend
  종류와 무관하게 실행 결과를 `run_store.save_run()`에 넘기고
  (`/api/run` 동기 경로, `/api/run/stream`의 `done` 이벤트 경로 모두 동일),
  `aipt/web/store.py`의 `save_run()`이 그 JSON 전체를 `RUN_STORE_DIR`(기본
  `data/runs/`, env로 재정의 가능)에 `<exec_id>.json`으로 동기 파일 쓰기한다.
  인메모리 `OrderedDict` 캐시(모든 읽기가 우선 거치는 경로, `MAX_RUNS=50`
  cap)는 여전히 존재하지만 그 자체가 저장소가 아니라 디스크 위의 캐시이며,
  프로세스 재시작 시 `RUN_STORE_DIR`를 재스캔해 다시 채워진다(`_load_from_disk_locked`).
  `MAX_RUNS`를 넘겨 축출되는 run은 메모리와 디스크 파일 양쪽에서 함께
  삭제된다 — "최근 N개만 유지"는 여전히 사실이지만, 그 N개가 재시작을
  버텨낸다는 점이 핵심 변경이다.
- Public AI(상용 API) 요청/응답 기록은 위와 별도로 `aipt/backends/public_ai/recorder.py`를
  통해 `data/public_ai_records/<exec_id>.json`에도 저장된다 — 과금이 발생한
  실제 API 호출 기록이라 재현 불가능하므로, run 자체의 영속화와 무관하게
  독립적으로 유지되는 감사 기록이다.
- 원래의 "한 대의 머신에서 도는 실험실이라 run은 영속 저장 안 하고 사용자가
  bundle.zip을 직접 다운로드해 보관한다"는 철학은 폐기되었다. run 디스크
  영속화 자체는 별도 DB 없이 파일 시스템에 JSON을 직접 쓰는 방식이므로
  "한 대의 머신에서 도는 실험실" 철학과 상충하지 않는다 — 다만 재시작해도
  run 목록이 남는 쪽으로 정책이 바뀌었다는 뜻이다.
- Docker 볼륨은 `./data/pcaps` + `./data/public_ai_records` + `./data/runs`
  (`RUN_STORE_DIR`) 3개를 마운트한다(`docker-compose.yml`: `RUN_STORE_DIR=/app/data/runs`).

## 5.1 리스트업 갱신 — Gateway 관련 신규 작업

| # | 작업 | 내용 |
|---|---|---|
| B9 | Gateway 컨테이너 신규 구현 | `aipt/gateway/`: netem 제어 루프 + 프로파일 프리셋 + `/gateway/profile` API. `aipt/core/netem.py` 로직을 재사용/승격 |
| B10 | Docker 토폴로지 확장 | `docker-compose.yml`에 `gateway` 서비스 추가, `mock-server`/`local-llm`이 `gateway`를 통해서만 도달 가능하도록 네트워크 구성 |
| B11 | 웹 UI Network profile 선택 | 실험 설정 폼에 프로파일 드롭다운 + Gateway API 연동 — **구현 완료**(`aipt/web/routes_gateway.py`, 커밋 `39c4ea78`, `85dc19fc`에서 idle-reset 리팩터와 함께 갱신; `GET/POST /api/gateway/profile`이 Gateway의 `/gateway/profile`을 프록시, `tests/web/test_routes_gateway.py`로 테스트됨. §6 참고) |

## 4.8 전체 아키텍처 다이어그램 (Mermaid)

```mermaid
flowchart TB
    subgraph USER["사용자"]
        Browser["웹 브라우저<br/>(실험 설정 · 결과 대시보드)"]
    end

    subgraph WEBAPP["aipt/web — FastAPI 단일 앱"]
        Routes["routes_config / routes_run /<br/>routes_runs /<br/>routes_gateway<br/>(구현 완료, B11)"]
        Templates["templates + static<br/>(backend 선택 + 공통 실험 폼)"]
    end

    subgraph BACKENDS["aipt/backends — Backend 프로토콜 (connect / send_turn / close)"]
        direction LR
        PublicAI["PublicAIBackend<br/>gemini.py / openai.py<br/>(6+4 arm)"]
        Mock["MockBackend<br/>server.py / records.py /<br/>replay.py / conversation.py"]
        LocalLLM["LocalLLMBackend<br/>gateway.py(프록시) +<br/>engine_adapter.py<br/>(구현 완료, §4.5c)"]
    end

    subgraph GATEWAY["Network Gateway 컨테이너 (신규, B9)"]
        Netem["tc netem 제어 루프<br/>delay / jitter / loss / reorder"]
        ProfileAPI["/gateway/profile API<br/>clean·wired·wireless·custom"]
    end

    subgraph EXT["외부 실제 인터넷"]
        Gemini["generativelanguage.googleapis.com"]
        OpenAI["api.openai.com"]
    end

    subgraph MOCKSRV["mock-server 컨테이너"]
        MockServer["HTTP/1.1 keep-alive 서버<br/>고정 byte 또는 Q&A fixture 응답"]
    end

    subgraph LLMSRV["로컬 서빙 엔진 컨테이너"]
        Engine["llama.cpp / vLLM<br/>(표준 엔진, 추론 위임)"]
    end

    subgraph CORE["aipt/core — 3-backend 공통 계측"]
        direction LR
        Cwnd["cwnd.py<br/>netlink 연속 cwnd 모니터<br/>(native/cwnd_monitor.c)"]
        Capture["capture.py<br/>tcpdump pcap 캡처<br/>(AppArmor 감지 포함)"]
        Offload["offload.py<br/>NIC TSO/GSO 토글"]
        Wire["wire.py / streaming.py<br/>소켓 바이트 카운터 · SSE 리더"]
    end

    subgraph EXPORT["aipt/export — 3-레이어 통합 산출물"]
        direction LR
        Connection["connection.py<br/>cwnd.csv"]
        Turns["turns.py<br/>turns.csv (+ goodput_bps)"]
        Packets["packets.py<br/>packets.csv (pcap 파싱)"]
        Bundle["bundle.py<br/>bundle.zip"]
    end

    Browser <--> Routes
    Routes --> Templates
    Routes --> PublicAI
    Routes --> Mock
    Routes --> LocalLLM
    Routes -->|"GET/POST /api/gateway/profile (구현 완료, B11)"| ProfileAPI

    PublicAI <-->|"실제 인터넷 경로<br/>(이미 실제 RTT/손실 보유)"| Gemini
    PublicAI <-->|"실제 인터넷 경로"| OpenAI

    Mock <--> Netem
    LocalLLM <--> Netem
    Netem <--> MockServer
    Netem <--> Engine

    PublicAI -. 계측 훅 .- CORE
    Mock -. 계측 훅 .- CORE
    LocalLLM -. 계측 훅 .- CORE

    CORE --> EXPORT
    EXPORT --> Routes

    style GATEWAY fill:#2d2d3a,stroke:#e0a030,stroke-width:2px,color:#fff
    style CORE fill:#243447,stroke:#4a90d9,stroke-width:2px,color:#fff
    style EXPORT fill:#2f3b2f,stroke:#5cb85c,stroke-width:2px,color:#fff
    style EXT fill:#3a2626,stroke:#c0392b,stroke-width:1px,color:#fff
```

**읽는 법**:
- **점선 화살표**(`-. 계측 훅 .-`)는 "이 backend가 core 모니터링을 훅으로 사용한다"는 관계 — 3개 backend 모두 동일한 `aipt/core`를 공유하며, backend별로 별도의 cwnd/capture 구현을 갖지 않는다.
- **PublicAIBackend**만 Gateway를 거치지 않고 실제 인터넷으로 직행 — 이미 진짜 네트워크 특성(RTT/손실/혼잡)을 갖고 있기 때문 (§4.7).
- **Mock/LocalLLM**은 반드시 **Gateway**를 경유 — Gateway가 `tc netem`으로 지연/손실/재정렬을 주입해 "완벽한 로컬 네트워크"라는 암묵적 가정을 제거한다. Gateway는 L3 IP 포워딩이라 프로토콜에 무관하게 동일하게 적용된다.
- **`routes_gateway`는 구현 완료(B11)** — `aipt/web/routes_gateway.py`가 `GET/POST /api/gateway/profile`을 완전히 구현해 Gateway 컨테이너의 `/gateway/profile` API로 프록시하며(`GATEWAY_HOST`/`GATEWAY_PORT` 실사용), `tests/web/test_routes_gateway.py`로 테스트도 갖췄다(커밋 `39c4ea78`, 이후 `85dc19fc`에서 idle-reset 리팩터와 함께 갱신). 위 다이어그램의 실선 화살표가 이를 반영한다.
- **LocalLLMBackend**의 게이트웨이(프록시, 애플리케이션 레벨 HTTP 신기능 실험 지점)와 **Network Gateway 컨테이너**(순수 네트워크 특성 주입, L3/L4)는 서로 다른 컴포넌트다 — 이름이 비슷해 혼동하기 쉬우므로 문서/코드에서 전자는 "engine gateway/proxy", 후자는 "network gateway"로 구분 표기할 것을 권장.
- 모든 backend의 turn 결과는 `aipt/core`가 관찰한 데이터와 함께 `aipt/export`의 3-레이어 CSV(connection/turns/packets) + bundle.zip으로 수렴한다.

## 4.9 정밀 측정: 짧은 RTT 환경을 위한 C 기반 수집기

**문제의식**: TCP/네트워크 통계는 Python 레벨에서 측정하면 GIL 스케줄링,
syscall 왕복, 타이머 해상도 때문에 측정 자체가 지터를 갖는다. 이 지터는
RTT가 수십 ms인 실제 인터넷(PublicAIBackend)에서는 무시할 수준이지만,
**RTT가 짧을수록 지터가 신호 자체를 삼켜버린다** — 애초에 `cwnd.py`의
2ms netlink 샘플링 주기가 이 문제 때문에 C 헬퍼(`native/cwnd_monitor.c`)로
분리되어 있는 것이 그 증거다: token_traffic 원본 주석에 "api.openai.com
엣지까지 3.3ms 경로에서, cwnd가 10→65로 슬로우스타트 복귀하는 데 걸리는
시간이 약 10ms인데, 10ms 주기 샘플러는 그 사건 자체를 건너뛴다"고
명시되어 있다.

**v2 아키텍처에서 이 문제가 더 커지는 이유**: PublicAIBackend만 다루던
때는 RTT가 항상 수 ms~수십 ms 범위(실제 인터넷)였다. 그런데 이제
MockBackend/LocalLLMBackend는 컨테이너 간 직결이거나 Gateway의 `clean`
프로파일(지연 0)일 때 RTT가 **수백 μs 이하**로 떨어진다. 기존 2ms 고정
주기로는 이런 경로에서 발생하는 idle-reset이나 slow-start 이벤트를
Python은커녕 C 헬퍼조차 놓칠 수 있다 — "3.3ms 경로에 10ms 샘플러"와
같은 실수를 "0.3ms 경로에 2ms 샘플러"로 반복하게 되는 셈이다.

### 결정: 네이티브 수집기를 정식 아키텍처 계층으로 승격 + 적응형 주기

| 항목 | 방침 |
|---|---|
| 원칙 | **TCP/네트워크 타이밍 관련 모든 정밀 측정(cwnd 샘플링, 패킷 도착 타임스탬프)은 Python이 아니라 C로 구현된 별도 프로세스가 수행**하고, Python은 그 결과를 소비만 한다. GIL/스케줄링의 영향을 받지 않는 별도 OS 프로세스/스레드로 분리하는 것이 핵심이며, 이미 `cwnd.py` + `native/cwnd_monitor.c`가 이 원칙을 구현하고 있다 — 이번 결정은 그 원칙을 **명시적 아키텍처 요소로 승격**하고 **RTT가 짧은 backend까지 확장**하는 것 |
| 샘플링 주기 적응 (신규) | 고정 2ms 대신, **경로의 예상 RTT에 비례해 주기를 정하는 적응형 로직** 도입: `interval_ms = max(MIN_INTERVAL_MS, measured_or_declared_rtt_ms / K)` (K는 슬로우스타트 burst 하나를 최소 몇 회 샘플링할지 결정하는 상수, 기존 문서의 "3.3ms 경로 → 10ms 복귀 → 2ms 주기로 5회 샘플" 비율을 기준으로 역산). Gateway가 주입한 지연값(`GATEWAY_DELAY_MS`)이나 mock/local_llm의 실측 RTT를 실행 전에 알 수 있으므로, 실행 시작 시 이 값을 `cwnd.Monitor`에 넘겨 주기를 자동 산정 |
| 하한 주기 | 순수 loopback/컨테이너 직결(RTT < 0.1ms) 같은 극단적으로 짧은 경로에서는 C 헬퍼도 물리적 하한(스케줄링 tick, netlink 왕복 비용)에 부딪힌다. 이 경우 "측정 불가/신뢰 구간 밖"임을 결과에 명시적으로 표시(`interval_below_reliable_floor: true` 같은 플래그) — 없는 정밀도를 있는 것처럼 보고하지 않는다 |
| 패킷 타임스탬프도 동일 원칙 적용 (신규, 확장) | `aipt/export/packets.py`의 inter-arrival gap 계산은 지금 tcpdump가 pcap에 기록한 타임스탬프에 의존한다. tcpdump 자체는 커널 캡처(AF_PACKET)라 userspace Python보다는 정확하지만, **짧은 RTT 경로에서 패킷 간격이 μs 단위로 좁아지면 pcap 타임스탬프의 커널 클록 해상도(통상 1μs, NIC에 따라 다름)가 병목**이 될 수 있다. 하드웨어 타임스탬프(`SO_TIMESTAMPNS`/`ETHTOOL_GET_TS_INFO`)가 있는 환경에서는 이를 우선 사용하도록 `capture.py`에 감지 로직 추가 검토 |
| 결과 스키마 | `cwnd.Monitor.result()`와 `packets.csv`에 **주기 산정 근거를 기록**: `interval_ms`, `interval_reason`("fixed" / "adaptive:rtt=<x>ms" / "floor_clamped"), `measurement_confidence`("high"/"degraded") — 나중에 어떤 실행이 신뢰할 만한지 사후에 판별 가능하게 (2026-09-03: export 계층인 `aipt/export/connection.py`의 `CONNECTION_SUMMARY_COLUMNS`/`connection_summary_csv()`에도 두 필드가 반영되어 `cwnd_summary.csv`까지 전달됨) |
| 적용 범위 | PublicAIBackend는 원래도 실제 인터넷 RTT(수~수십 ms)라 기존 2ms 고정 주기로 충분 — 적응형 로직은 **MockBackend/LocalLLMBackend에서만 활성화**해도 됨 (단, 공통 코드 경로이므로 구현은 `aipt/core/cwnd.py`에 위치하고 파라미터로 제어) |

### 아키텍처 다이어그램 반영

§4.8 다이어그램의 `CORE` 서브그래프를 아래처럼 이해할 것: `Cwnd`(cwnd.py +
native C 헬퍼)와 향후 확장될 패킷 타임스탬프 수집기는 **Python 프로세스와
독립된 네이티브 프로세스**로 그려야 정확하다 — 화살표가 Python 쪽에서
"제어(시작/중지, 주기 파라미터 전달)"만 하고, 실측 자체는 C 프로세스가
자신의 클록으로 수행한 뒤 NDJSON으로 결과만 넘기는 단방향 데이터 흐름이다.

```mermaid
flowchart LR
    subgraph PY["Python (aipt.core.cwnd.Monitor)"]
        Ctrl["제어: start/stop, interval_ms 파라미터<br/>(경로 RTT로부터 적응 산정)"]
        Reader["리더 스레드: NDJSON 소비만,<br/>측정에는 관여하지 않음"]
    end

    subgraph NATIVE["native/cwnd_monitor.c — 별도 프로세스"]
        Loop["자체 클록의 netlink sock_diag 폴링 루프<br/>GIL/Python 스케줄링과 완전 분리"]
    end

    Ctrl -->|"spawn + track 명령"| Loop
    Loop -->|"NDJSON (stdout)"| Reader
```

### 신규 작업 리스트업 (B12)

| # | 작업 | 내용 |
|---|---|---|
| B12 | 적응형 cwnd 샘플링 주기 | `aipt/core/cwnd.py`에 `interval_from_rtt(rtt_ms, k=...)` 헬퍼 추가, Mock/LocalLLM backend의 `connect()`가 경로 RTT(Gateway 프로파일의 delay 설정값 또는 실측 RTT)를 넘기도록 연동. 결과에 `interval_reason`/`measurement_confidence` 필드 추가 |
| B13 (검토) | pcap 타임스탬프 정밀도 확인 | `capture.py`가 캡처 인터페이스의 타임스탬프 소스(소프트웨어 vs 하드웨어)를 확인하고 결과에 기록. 짧은 RTT 경로에서 `packets.csv`의 inter-arrival gap 신뢰도 판단 근거로 사용 |

## 4.10 engine Gateway 요청 leaf-hash 중복 제거 캐싱 (2026-09-01 설계+구현 완료)

**배경**: §4.5 v2 개정에서 `LocalLLMBackend`용으로 마련해 둔 "표준 서빙
엔진 앞단 자체 프록시(engine gateway)"의 첫 실제 활용 사례. 최초에는 이
프록시의 `on_request`/`on_response` 훅을 "LLM *응답*을 캐싱"하는
용도로 상상했으나, 2026-09-01 Slack 설계 논의에서 실제 필요는 "매 턴
재전송되는 요청 *컨텍스트*의 중복 제거"임이 확인되어 정정됐다 —
`docker/engine_gateway.py`의 `on_cacheable_request`/`on_cacheable_response`
(응답 캐싱용 훅, 여전히 no-op)와는 다른, 별도의 요청 dedup 프로토콜로
구현됨.

**Seed 문서**: `docs/engine_gateway_caching_seed.md`가 SSoT — 문제 정의,
opt-in 헤더, 세션 경계(TCP 커넥션), 와이어 포맷(leaf-hash 치환 +
`$aipt_cache_map`), 해시 함수/길이(SHA-256 앞 20자, 충돌확률 계산 포함),
캐시 미스 처리(HTTP 409 + 부분 재전송), 임계값(200 bytes), 구현
위치(`Gateway.send()`의 `on_request` 훅 실행 지점) 등 모든 설계 결정과
근거를 담고 있다. 이 절은 아키텍처 문서 관점에서 **어디에 무엇이
배치됐는지**만 요약한다 — 상세 근거는 Seed 문서를 따라갈 것.

**핵심 구현 산출물**:

| 파일 | 역할 |
|---|---|
| `aipt/core/cache_protocol.py` (신규) | 클라이언트/서버 공유 stdlib-only 프로토콜 모듈 — leaf 순회, hash 치환/복원, `SessionCache`, `CacheMiss` |
| `aipt/backends/local_llm/gateway.py` (수정) | `Gateway.send()`에 캐싱 로직 통합: 캐싱 시 uncached baseline과 실제 전송량을 같은 호출에서 비교해 `GatewayResult.cache_bytes_saved` 산출, 409 수신 시 1회 재전송 |
| `docker/engine_gateway.py` (수정) | `_Handler`에 커넥션 단위 `SessionCache` 부착, `_relay_cacheable()`가 캐시 디코드/409 응답 처리 |
| `aipt/backends/local_llm/__init__.py`, `aipt/web/routes_run.py` | `cache_enabled`/`cache_threshold_bytes`를 env(`LOCAL_LLM_CACHE_ENABLED` 등) 또는 `RunRequest` 필드로 opt-in, 웹 폼 체크박스까지 연결 |
| `aipt/backends/record.py`, `aipt/export/turns.py`, `aipt/web/static/app.js` | `cache_bytes_saved`를 `turn_record()`/`turns.csv`/웹 결과 테이블에 신규 컬럼으로 노출 |
| `docker-compose.yml`, `docker/Dockerfile.local_llm`, `docker/entrypoint_local_llm.py` | engine Gateway를 `local-llm` 컨테이너 안 별도 sidecar 프로세스(포트 40079)로 승격, `web`이 바라보는 `LOCAL_LLM_ENGINE_URL` 기본값을 40080(llama-server 직접)에서 40079(engine Gateway 경유)로 변경 |

**실측 검증**: `scripts/measure_perf_cache_savings.py`가 실제
docker-compose 4-서비스 토폴로지(web → Network Gateway L3/L4 →
engine Gateway L7 → llama-server) 위에서 동일 20턴 시나리오
(`records/perf_short_smoketest.json`)를 캐싱 off/on 두 번 실행해 턴별
`req_payload_bytes`/`wire_sent`를 비교 — 결과는 `data/runs/
cache_savings_multiturn.csv`, 요약 수치는 ARCHITECTURE.md §3.3 참고
(요청 payload 87.2%, 실제 wire 전송량 86.3% 절감).

**§4.8 다이어그램 갱신 필요 사항(반영 완료)**: `LocalLLM` 노드가 이제
`gateway.py`(클라이언트측 캐싱 로직 포함)를 가리키고, `Netem`(Network
Gateway) 뒤에 있던 `Engine`(llama.cpp/vLLM) 노드 앞에 engine Gateway
L7 sidecar 홉이 추가로 존재한다 — 상세 시퀀스는 ARCHITECTURE.md §3.3의
Mermaid 다이어그램 참고 (DESIGN.md의 §4.8은 backend 조합의 개괄
다이어그램이라 L7 sidecar 세부 홉까지는 그리지 않고, "LocalLLM <-->
Netem <--> Engine" 화살표가 실제로는 "LocalLLM → Netem → engine
Gateway(L7, 캐싱 처리) → Engine"임을 텍스트로만 명시).

## 4.11 idle-reset(slow-start-after-idle) 토글 — client-only 재설계 (2026-09-02)

**대상**: `net.ipv4.tcp_slow_start_after_idle` sysctl. 이 값이 `1`(Linux
기본값)이면 커널은 소켓이 한 RTO만큼 idle 상태였다가 다시 보낼 때 cwnd를
버리고 slow start로 되돌린다 — `tcp_congestion`(§1의 idle-reset 실험) 및
`aipt/core/cwnd.py`의 `idle_resets()` 판정 로직이 관측하는 바로 그 현상.

**원 설계 (2026-09-01)**: *responding* 쪽(`mock-server`/`local-llm`)의
sysctl을 토글했다 — 서버가 보내는 응답이 더 크므로 그쪽 cwnd가 지배적일
것이라는 가정. 이를 위해 mock-server에 `/admin/idle-reset` 라우트를,
local-llm에 `docker/idle_reset_admin.py` 사이드카를 두어 각각 자기 컨테이너의
sysctl을 쓰게 했다.

**2026-09-01 인과 실험 결과 (재설계의 근거)**: `docs/experiments/2026-09-01-idle-reset-results.md`가
실제로 원 설계를 검증했다. mock-server 측 idle-reset은 실험 내내
`enabled=1`(기본값)로 **고정**한 채, 오직 `web`(client) 자신의
idle-reset만 두 조건(on/off)으로 바꿔 **다음 턴 요청 업로드 지연**
(next-turn request upload latency)을 측정했다 — 이 실험이 보려던 것은
응답 방향이 아니라 업로드 방향이므로 서버측 설정은 의도적으로 무관하게
고정했다. 결과: 이 지표는 responding side가 아니라 `web` 자신의
**송신측(send-side) cwnd**에 의해 좌우되며, on/off 조건 간 업로드 지연이
최대 407배 차이났다 — `web`이 idle 구간 후 다음 요청을 보내는 것은
언제나 `web` 자신의 소켓이므로, 그 소켓의 cwnd가 idle-reset으로 초기화됐는지
여부가 직접적인 원인이다. 즉 원 설계(서버측 토글)는 측정하려는 인과관계와
무관한 소켓을 건드리고 있었다.

**2026-09-02 재설계 (operator 지시)**: `aipt/core/idle_reset.py` 모듈은
그대로 재사용하되, 호출 주체를 **오직 `web` 프로세스 자기 자신**으로
한정했다. `aipt/web/routes_gateway.py`의 `GET`/`POST` `/api/idle-reset`이
`aipt.core.idle_reset.read()`/`write()`를 **in-process로, 네트워크 홉 없이**
직접 호출하며, 대상 경로는 언제나 `IDLE_RESET_PATH =
"/proc/sys/net/ipv4/tcp_slow_start_after_idle"`(파라미터 기본값), 즉
`web` 컨테이너 자신의 netns다. `web`이 이미 자기 자신의 netns를 소유하므로
별도 프록시/사이드카가 필요 없다(`idle_reset.py` 모듈 docstring 근거). API
시그니처에도 `backend=` 같은 대상 선택 파라미터가 없다 —
`set_idle_reset(enabled: bool)`뿐이며, 어떤 backend(mock/local_llm/public_ai)를
선택했든 이 토글은 항상 `web` 자신에게만 적용된다.

이 재설계로 다음이 **삭제된 죽은 코드**가 됐다(2026-09-02):
- mock-server의 `/admin/idle-reset` admin 라우트
- local-llm의 `docker/idle_reset_admin.py` 사이드카 프로세스
- 위 둘을 위해서만 존재하던 서버측 admin 프록시 경로 전체

(과거 커밋 히스토리에서 확인 가능. `tests/web/test_routes_gateway.py`의
`test_idle_reset_never_makes_http_call`이 "예전 프록시 경로로 되돌아가면
즉시 실패"하도록 회귀 방지 중.)

**주의(제약)**: `public_ai` backend는 이 프로젝트가 컨테이너 netns를
소유하지 않는 실제 인터넷 종단이므로, `public_ai`를 선택한 상태에서
이 토글을 조작해도 Gemini/OpenAI 서버에는 아무 영향이 없다 — 여전히
`web` 자신의 송신측 cwnd만 바뀐다. 이는 코드가 의도한 대로 정확히
동작하는 것이지만, UI가 backend와 무관하게 항상 이 필드를 노출하므로
"현재 선택한 backend의 응답 경로에 영향을 준다"는 오해를 유발할 수 있다
(`docs/seed-2026-09-01-idle-reset-experiment.md` E3 항목 참고).

**참고 문서**: `docs/experiments/2026-09-01-idle-reset-results.md`(실험
로그), `docs/audit-2026-09-02/core.md` §4.1, `docs/audit-2026-09-02/web.md`
§5.1(코드 근거 정밀 검증).

## 5.2 문서-코드 정합성 점검 (2026-09-01, ooo 인터뷰 기반 전수 감사)

AIPT를 ooo(Ouroboros) 워크플로우로 재정의하면서, 실제 코드를 병렬 서브에이전트로
전수 조사(빌드/기동/실측 포함)해 DESIGN.md와의 괴리를 확정했다. §6의 6개
미해결 결정은 **모두 확정 완료**(코드 레벨로 재확인, 아래 §6에 확정 내용 갱신).
당시 발견된 괴리는 아래 2건이며, **둘 다 이후(2026-09-02, T2/T8) 정식
개정으로 이미 해소되었다** — 이 절은 감사 당시 시점의 기록으로 보존한다:

1. **§4.7.1 저장 정책 stale (해소됨, T8)** — 당시엔 "Public AI 기록만 영속
   저장, 나머지는 인메모리 최근 50개"라고 확정돼 있었으나, 실제로는
   `RUN_STORE_DIR`(`data/runs/`)에 모든 backend의 run을 디스크 영속화하고
   있었다(2026-08-27 "Run store 디스크 영속화" 작업). §4.7.1은 2026-09-02
   T8로 이 실제 동작에 맞춰 정식 개정 완료됨.
2. **B11(웹 UI Network Profile 선택) 구현 완료 (해소됨, T2)** — Gateway 컨테이너의
   `GET/POST /gateway/profile` API는 완성되어 실동작 확인됐고(§4.7),
   `aipt/web/routes_gateway.py`가 `GET/POST /api/gateway/profile`로 이를
   프록시하며 실험 폼에도 프로파일 드롭다운이 연결되어 있다(커밋
   `39c4ea78`, 이후 `85dc19fc`에서 idle-reset 리팩터와 함께 갱신;
   `tests/web/test_routes_gateway.py`로 테스트됨). `GATEWAY_HOST`/
   `GATEWAY_PORT` env는 `web` 서비스에 주입되어 `routes_gateway.py`가
   실제로 읽어 쓰는 live config다 — 사용자가 웹 UI에서 직접 프로파일을
   바꿀 수 있다.

또한 이번 감사에서 **Docker HEALTHCHECK 버그**를 신규 발견했다: `local-llm`
컨테이너의 llama.cpp 서버는 `--port 40080`으로 정상 기동·응답하지만
(`curl -s http://127.0.0.1:40080/health` → `{"status":"ok"}`, 컨테이너 내부에서
직접 확인), base 이미지가 물려주는 HEALTHCHECK가 여전히 기본 포트 8080을
찔러 항상 `unhealthy`로 표시된다. 기능 장애는 아니지만(2026-09-01 실측:
Qwen2.5-0.5B-Instruct-GGUF 모델로 `local_llm` 백엔드 end-to-end 3턴 실행
성공 — TTFT 583ms, wire_sent/recv 정상 계측) `docker compose ps`만 보고
"고장났다"고 오판할 여지가 있어 `docker/Dockerfile.local_llm`의
HEALTHCHECK 정의를 40080으로 고쳐야 한다.

**HEALTHCHECK 비대칭 (설계 의도, 2026-09-02 감사 native.md §3.2로 문서화)**:
네 개 서비스 이미지 중 `docker/Dockerfile.local_llm`에만 HEALTHCHECK
지시문이 있고 `docker/Dockerfile.web` / `docker/Dockerfile.mockserver` /
`docker/Dockerfile.gateway`에는 전혀 없다. 이는 결함이 아니라 각 서비스의
기동 특성 차이를 반영한 의도된 비대칭이다.

- `local-llm`: 업스트림 `ghcr.io/ggml-org/llama.cpp:server` 이미지를 그대로
  쓰며, 실제 추론 엔진(llama-server)이 모델 가중치를 로드하고 나서야 요청에
  응답할 수 있다 — `start-period=60s`가 이 로딩 지연을 감안한 값이다. 즉
  컨테이너가 "떠 있음"과 "요청을 받을 준비가 됨" 사이에 의미 있는 시간차가
  있고, `docker compose ps`/오케스트레이션이 그 차이를 구분해야 할 실질적
  필요가 있어 HEALTHCHECK를 정의(및 40080 포트로 재정의)했다.
- `web` / `mock-server` / `gateway`: 각각 FastAPI(uvicorn)·stdlib
  `http.server`·FastAPI(uvicorn) 기반의 얇은 애플리케이션 서버로, 무거운
  모델 로딩이나 외부 프로세스 기동 대기 없이 프로세스 시작과 거의 동시에
  요청을 받을 준비가 된다. 기동-완료와 요청-가능 사이의 지연이 사실상
  없으므로 별도 HEALTHCHECK로 얻는 이득이 local-llm만큼 크지 않아 지금까지
  추가되지 않았다.
- 다만 이 세 서비스도 HEALTHCHECK가 전혀 없으므로 `docker compose ps`가
  이들의 healthy/unhealthy 상태를 표시하지 못한다는 점은 실사양의 한계로
  남아 있다(native.md §2.7/§3.2). 필요해지면(예: 오케스트레이션 레벨의
  readiness gate) 각 서비스의 `/health`류 엔드포인트를 대상으로 가벼운
  HEALTHCHECK를 추가하는 것을 향후 과제로 고려한다.

## 6. 미해결 설계 결정 (구현 전 확인 필요) — [x] 2026-09-01 전항목 확정 완료 (아래 각 항목 참고)

1. **`aipt/core/cwnd.py` 최종 API** — token_traffic의 `provider/arm/kind` 3필드 라벨링 vs tcp_congestion의 단일 `label` 문자열. 제안: `label` 하나로 통일하고 호출측에서 `f"{provider}:{arm}:{kind}"` 형태로 조립 (synthetic_mock은 조립 없이 그대로 label 사용).
   **확정 (2026-09-01, 코드 재확인): 제안대로 `label: str` 단일 문자열로 통일됨.** `aipt/core/cwnd.py`의 `Monitor.__init__(label, ...)`. 호출측이 `f"{provider}:{arm}:{kind}"` 조립(docstring 명시), synthetic_mock은 조립 없이 그대로 사용.
2. **`core/capture.py`의 caller당 pcap 개수** — external_api는 (provider, arm, kind)당 1개, synthetic_mock은 run당 1개. `label` 파라미터로 이미 일반화 가능해 보이나, external_api의 dual-pass(bytes/latency 분리 캡처) 요구사항까지 커버되는지 이관 시 재검증 필요.
   **확정 (2026-09-01): `Capture` 클래스가 `label` 직접 전달(synthetic_mock, run당 1개) / `provider・arm・kind` 자동 조립(`{provider}_{arm}_{kind}`, external_api) 두 방식 모두 지원. dual-pass 요구사항도 `kind`가 파일명에 포함되어 별도 pcap으로 분리되는 것으로 재검증됨.**
3. **의존성 통합** — token_traffic은 `requests` 기반, google-genai/openai SDK는 사용 안 함(SDK가 httpx라 소켓 카운터 훅이 안 걸림). synthetic_mock은 표준 라이브러리 위주. `pyproject.toml` 하나로 합칠 때 optional-dependency 그룹(`[external-api]`, `[dev]`)으로 나눌지 결정 필요.
   **확정 (2026-09-01): `[external-api]` 대신 기능별 4개 그룹으로 결정.** base `dependencies=["requests"]`, optional `dev`(pytest) / `export`(dpkt) / `web`(fastapi/uvicorn/jinja2/python-multipart/httpx).
4. **테스트 마킹** — 기존 `test_conversation_live.py`, `test_cwnd_live.py` 등 "live"(실제 소켓/커널 필요) 테스트를 pytest 마커(`@pytest.mark.live`)로 통합 표시할지, 두 프로젝트 관례가 달랐다면 통일 필요.
   **확정 (2026-09-01): `@pytest.mark.live`로 통일.** `pyproject.toml`의 `[tool.pytest.ini_options] markers`에 등록, 9개 테스트 파일이 사용 중(`test_cwnd.py`, `test_capture.py`, `test_conversation_live.py`, `test_backend_live.py`, `test_engine_live.py` 등). CI 기본 실행은 `-m "not live"`.
5. **`data/` 저장 위치** — token_traffic은 `TRAFFIC_DATA_DIR`(런 JSON), tcp_congestion은 `data/pcaps/`(pcap만, 메모리에 최근 1건만 유지). 병합 후 두 lab이 저장소를 공유할지, `data/external-api/`·`data/synthetic-mock/`으로 분리할지 결정 필요. **권장: 분리** — 두 lab의 보존 정책이 다르다(external_api는 20개 런 유지 pruning, synthetic_mock은 최근 1건만 메모리 유지).
   **확정 (2026-09-01, 권장안과 다르게 결정): lab별 분리가 아니라 기능별 공유 구조로 최종 결정됨.** `data/pcaps/`(capture.py), `data/runs/`(web/store.py — 전 backend 공통, §4.7.1과 달리 영속화됨, §5.2 참고), `data/public_ai_records/`(recorder.py) 3종 — 3개 backend가 이 디렉터리들을 모두 공유하며 lab 단위 구분은 없음. 3-backend 통합 아키텍처(§4.5 v2)로 개정되며 애초에 "lab"이라는 개념 자체가 폐기되었으므로 이 편차는 자연스러운 결과.
6. **모노레포 `CLAUDE.md` 갱신** — `remote_work/CLAUDE.md`의 프로젝트 테이블에서 `token_traffic` 행을 `AIPT`로 교체하고, `tcp_congestion`(테이블에 없었음 — 등록 필요했을 수도)도 정리. 실제 코드 이관 완료 후 반영.
   **확정 (2026-09-01): 완료 확인.** 커밋 `fabadc48`에서 `token_traffic` 행을 `AIPT`로 교체. `tcp_congestion` 행은 원래 테이블에 없었으므로 별도 조치 불요. 현재 `remote_work/CLAUDE.md`에 `AIPT` 행만 존재함을 재확인.

## 6. 실행 계획 (다음 단계, 리뷰 후 진행)

이번 세션은 **설계 문서 + 폴더 스켈레톤까지만** 완료. 실제 코드 이동/포팅은
아래 순서로 별도 작업 세션에서 진행 예정:

| Phase | 내용 | 산출물 |
|---|---|---|
| 1 | `native/cwnd_monitor.c`, 순수 공유 가능 core(cwnd/capture/offload) 이관 + 유닛 테스트 이식 | `aipt/core/*.py` + `tests/core/*` 그린 |
| 2 | external_api 도메인 로직(wire/streaming/call/record/metrics/store/scenario/cachebust/providers) 이관 | `aipt/labs/external_api/*`, `aipt/providers/*` + 테스트 그린 |
| 3 | synthetic_mock 도메인 로직(server/probe/conversation/congestion/netem) 이관 | `aipt/labs/synthetic_mock/*` + 테스트 그린 |
| 4 | Flask → FastAPI 라우트 포팅, 템플릿/정적 자산 이관, 랜딩 페이지 신설 | `aipt/web/*` 로컬 구동 확인 |
| 5 | Docker 통합(web + mockserver 2-서비스 compose), 빌드 스테이지 정리 | `docker compose up --build` 통과 |
| 6 | 문서 최종화(README/docs), 모노레포 CLAUDE.md 갱신, 원본 `token_traffic/`·`tcp_congestion/` 삭제(또는 archive) | 단일 소스 오브 트루스 확정 |


## 7. QUIC idle-probe spike — 제거됨 (T3, 2026-09-02)

이전 §7에 있던 QUIC(aioquic 기반) idle-probe 혼잡제어 실험 스파이크
(`aipt/backends/quic_mock/`, `aipt/core/quic_congestion.py`, `docker/Dockerfile.quic_mock_server`,
`docker-compose.yml`의 `quic-mock-server` 서비스, 웹 UI의 Transport(QUIC)
드롭다운, 관련 테스트/문서)는 audit T3 결정에 따라 코드와 문서 양쪽에서
전면 제거됐다. 실측 결과 idle_probe 알고리즘이 순net loss였고(개선 없음),
스파이크 이후 실제로 사용되지 않는 죽은 표면이었다. 후속으로 QUIC/HTTP3
실험이 필요해지면 새 설계 문서로 다시 시작한다.
