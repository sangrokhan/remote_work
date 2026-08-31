# AIPT

**AI Protocol Traffic lab** — LLM 트래픽이 TCP/네트워크 프로토콜에 미치는 영향을
측정하는 실험실. 이전에 별도 프로젝트였던 `token_traffic`(실제 Gemini/OpenAI API
대상 byte/token/latency 측정)과 `tcp_congestion`(idle-reset cwnd 측정, 애초에
`token_traffic`의 코어에서 파생됨)을 하나로 병합해서 만들어졌다.

## 무엇을 측정하는가

클라이언트가 **3개의 backend 중 하나**를 상대로 멀티턴 대화를 재생하면서,
동일한 계측 계층(`aipt/core`)으로 TCP/네트워크 동작을 실측한다.

| Backend | 상대 | 측정 초점 |
|---|---|---|
| **Public AI** (`public_ai`) | 실제 Gemini/OpenAI API (과금) | 대화 히스토리 유지 방식(stateless/stateful-pointer/explicit-cache)별 업로드 바이트·과금 토큰·지연시간 |
| **Mock** (`mock`) | 로컬 mock 서버 | 고정 byte 또는 fixture 재생 트래픽에서 idle 구간 후 TCP congestion window의 slow-start-after-idle 리셋 |
| **Local LLM** (`local_llm`) | 표준 서빙엔진(llama.cpp/vLLM) + 자체 프록시 | 서버측 HTTP 신기능/프로토콜 실험을 위한 실 LLM 트래픽 발생 |

Mock/Local LLM 경로는 **Network Gateway** 컨테이너(`aipt/gateway`, `tc netem`
기반)를 거쳐 지연·지터·손실·재정렬을 주입할 수 있다 — Public AI는 이미 실제
인터넷을 거치므로 Gateway를 경유하지 않는다.

## 문서

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — **최종 아키텍처 레퍼런스**. 컴포넌트
  5개(PublicAI/Mock/LocalLLM/Gateway/프론트) 구조, 폴더 구조, 데이터 흐름 +
  backend별 시퀀스 다이어그램, API 설계, 성능 설계(적응형 cwnd 샘플링, pcap
  운영), 테스트 설계. **먼저 이 문서를 읽을 것.**
- [`DESIGN.md`](./DESIGN.md) — 설계 결정의 근거와 이력(왜 이렇게 정했는지,
  검토했던 대안, 아직 열려 있는 이슈).
- [`MIGRATION.md`](./MIGRATION.md) — token_traffic/tcp_congestion → AIPT
  파일 단위 이관 기록.
- 남은 작업 추적: 이관/병합 이후 발견된 항목(Docker 실검증, 웹 레이어
  TODO 등)은 모두 완료되어 `TODO.md`는 제거됨(2026-08-27). 앞으로의
  미해결/설계상 남은 이슈는 `ARCHITECTURE.md` §7 "아직 열려 있는 것"과
  `DESIGN.md` §6 "미해결 설계 결정"에서 관리한다.

## 빠른 시작

### 로컬 (Docker 없이)

```bash
cd AIPT
python3 -m venv .venv && .venv/bin/pip install -e ".[web,export,dev]"

# native cwnd 모니터 빌드 (netlink 연속 샘플링, 리눅스 전용)
cc -O2 -Wall -o native/cwnd_monitor native/cwnd_monitor.c

.venv/bin/pytest tests/ -q -m "not live"   # 433 passed, 1 skipped, 12 deselected

.venv/bin/uvicorn aipt.web.app:create_app --factory --host 0.0.0.0 --port 10000
# → http://localhost:10000
```

`-m "not live"`를 빼면 실제 소켓/커널 netlink/tcpdump가 필요한 테스트도 함께
돌아간다(환경에 따라 skip 처리됨).

### Docker

`docker-compose.yml`은 `web` → `gateway` → `mock-server` 3-service 토폴로지로
구성되어 있다(`local-llm` 엔진 컨테이너는 무거워서 기본 compose에는 포함하지
않음 — 대신 `LOCAL_LLM_ENGINE_URL`로 외부에서 실행 중인 llama-server/vLLM을
가리키면 `local_llm` backend를 그대로 사용할 수 있다).

```bash
cd AIPT
make up   # .env 없으면 scripts/ensure_env.sh가 .env.example -> .env로 자동 생성 후 docker compose up --build
```

`.env`가 이미 있으면 `make up`은 그 파일을 그대로 사용하고(절대 덮어쓰지 않음), `GEMINI_API_KEY`/`OPENAI_API_KEY` 등 필요한 값만 채우면 된다. 직접 `docker compose`를 쓰고 싶다면 먼저 `./scripts/ensure_env.sh`(idempotent, 이미 있으면 스킵)를 한 번 실행하거나 수동으로 `cp .env.example .env`를 해도 된다 -- `docker-compose.yml`의 모든 서비스는 `${VAR:-default}` 형태로 기본값이 박혀 있어 `.env`가 아예 없어도 부팅은 되지만(`.env.example`과 동일한 기본값), 실제로 그 기본값이 뭔지 매번 `docker compose config`로 확인하는 대신 `.env`가 항상 존재하도록 만드는 쪽이 더 명시적이다.

기동 후 웹 UI: <http://localhost:10000>

- `web`: FastAPI 앱(`aipt.web.app:create_app`). `NET_ADMIN`/`NET_RAW`
  capability로 cwnd 모니터/tcpdump 캡처/NIC offload 제어. `./data/pcaps`가
  호스트에 볼륨 마운트된다.
- `gateway`: `aipt/gateway/` 기반 Network Gateway 컨테이너. **순수 L3 IP
  포워딩**으로 동작한다 — `web`(net-client, 172.28.1.0/24)과
  `mock-server`(net-backend, 172.28.2.0/24)를 분리된 Docker 네트워크에
  두고, `gateway`만 양쪽에 속해 커널(`net.ipv4.ip_forward=1`)로 그
  사이를 라우팅한다. TCP 페이로드를 들여다보는 애플리케이션 프록시가
  아니다. `NET_ADMIN` capability로 양쪽 인터페이스에 `tc netem`
  프로파일을 동시 적용(`GET`/`POST /gateway/profile`, 프리셋:
  `clean`/`broadband`/`3g`/`satellite`/`lossy`/`custom`) — 왕복 요청과
  응답 모두 같은 지연/손실을 겪는다.
- `mock-server`: `aipt.backends.mock.server`를 구동하는 경량 컨테이너.
  호스트에 포트를 노출하지 않고 `net-backend` 네트워크에서만 도달 가능하다.

실행 결과 저장 정책: **Public AI(상용 Gemini/OpenAI API) 요청/응답만**
`./data/public_ai_records/`에 JSON으로 자동 영속 저장된다. 그 외 모든
산출물(cwnd 샘플, pcap, mock/local_llm 턴 기록, CSV)은 인메모리에만
있다가 실행 직후 `bundle.zip`으로 사용자가 직접 다운로드해서 관리한다 —
별도 DB나 파일 저장소는 없다.

개별 서비스만 빌드/재기동하려면 `docker compose build web`,
`docker compose up -d gateway mock-server` 처럼 서비스명을 지정한다.

## 폴더 구조 (요약)

```
AIPT/
├── aipt/
│   ├── core/       # 3-backend 공통 계측: cwnd, capture, offload, wire, streaming, netem
│   ├── backends/   # Backend 프로토콜 + public_ai / mock / local_llm 구현체
│   ├── gateway/    # Network Gateway (tc netem 제어 + 프로파일 API, 별도 컨테이너)
│   ├── export/     # connection.csv / turns.csv(+goodput_bps) / packets.csv / bundle.zip
│   └── web/        # FastAPI 단일 앱 (backend 선택형 실험 UI)
├── native/         # cwnd_monitor.c — netlink 연속 샘플링, 별도 프로세스
├── docker/          # Dockerfile.{web,gateway,mockserver}
├── docker-compose.yml
├── tests/           # core / backends / export / web / gateway — 440+ tests
├── ARCHITECTURE.md  # 최종 아키텍처 레퍼런스 (다이어그램 포함)
├── DESIGN.md        # 설계 결정 이력
└── MIGRATION.md     # 이관 기록
```

전체 트리와 각 모듈의 역할은 `ARCHITECTURE.md` §1.2를 참고.

## 테스트

```bash
.venv/bin/pytest tests/ -q -m "not live"
```

`@pytest.mark.live`가 붙은 테스트는 실제 소켓/커널 netlink/tcpdump 등 이
프로세스가 실행 중인 환경의 실제 자원이 필요하다 — 없는 환경에서는 정직하게
skip되고(예외로 죽지 않음), 있는 환경(예: Docker `web` 컨테이너)에서는 마커를
빼고 전체 실행하면 함께 돌아간다.

## 원본 프로젝트 (참고)

`token_traffic/`, `tcp_congestion/`는 이 병합 작업 완료 후 저장소에서
제거되었다. 각 프로젝트의 마지막 상태는 git 히스토리에 남아 있다
(`git log --oneline -- token_traffic`, 병합 직전 커밋에서 `tcp_congestion`도
동일하게 확인 가능).
