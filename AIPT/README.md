# AIPT

**AI Protocol Traffic lab** — LLM 트래픽이 TCP 프로토콜에 미치는 영향을 두 각도에서
측정하는 실험실. `token_traffic`과 `tcp_congestion`, 두 개의 독립 프로젝트를 하나로
병합하는 작업이 진행 중이다.

- **external_api lab** (구 `token_traffic`): 실제 Gemini/OpenAI API를 상대로,
  대화 히스토리를 유지하는 방식(client-resend / server-side pointer / explicit
  cache / stateless)에 따라 업로드 바이트·과금 토큰·지연시간이 어떻게 달라지는지
  측정.
- **synthetic_mock lab** (구 `tcp_congestion`): 로컬 mock 서버를 상대로, 멀티턴
  대화의 idle 구간(추론 대기) 이후 TCP congestion window가 slow-start-after-idle로
  리셋되는 현상을 netlink 연속 모니터링으로 실측.

두 lab은 공통 코어(`aipt/core/`: netlink cwnd 모니터, tcpdump 캡처, NIC offload
제어)를 공유한다 — 실제로 `tcp_congestion`은 애초에 `token_traffic`의 코어 모듈을
이식해서 만들어진 프로젝트였다.

## 현재 상태

**설계 단계.** 아래 문서를 먼저 검토하라:

- [`DESIGN.md`](./DESIGN.md) — 현황 분석, 목표 폴더 구조, 웹 UI(FastAPI 단일화)/
  Docker 통합 방침, 미해결 결정 사항
- [`MIGRATION.md`](./MIGRATION.md) — 파일 단위 이관 체크리스트 (Phase 1~6)

원본 프로젝트는 아직 이동하지 않았다:

- `../token_traffic/` — external_api lab 원본
- `../tcp_congestion/` — synthetic_mock lab 원본

코드 이관은 설계 리뷰 완료 후 Phase 1부터 순서대로 진행한다.

## Docker로 실행하기

`docker-compose.yml`은 DESIGN.md 4.7(Network Gateway)/B10 방침에 따라
`web` → `gateway` → `mock-server` 3-service 토폴로지로 구성되어 있다
(`local-llm` 엔진 컨테이너는 무거워서 이번 phase에서는 생략 — 대신
`LOCAL_LLM_ENGINE_URL`로 외부에서 실행 중인 llama-server/vLLM을 가리키면
`local_llm` backend를 그대로 사용할 수 있다).

```bash
cd AIPT
cp .env.example .env   # GEMINI_API_KEY / OPENAI_API_KEY 등 필요시 채우기
docker compose up --build
```

기동 후 웹 UI: <http://localhost:10000>

- `web`: FastAPI 앱(`aipt.web.app:create_app`), `NET_ADMIN`/`NET_RAW`
  capability로 cwnd 모니터/tcpdump 캡처/NIC offload 제어. `./data/pcaps`가
  호스트에 볼륨 마운트된다.
- `gateway`: `aipt/gateway/` 기반 Network Gateway 컨테이너, `NET_ADMIN`
  capability로 `tc netem` 프로파일 제어(`/gateway/profile`). 현재는 컨테이너
  토폴로지 수준까지만 배선되어 있고, `gateway`→`mock-server` 실제 L3/L4
  포워딩은 TODO (DESIGN.md 4.7 "미해결 세부사항 1" 참고, `docker-compose.yml`
  상단 주석에도 명시).
- `mock-server`: `aipt.backends.mock.server`를 구동하는 경량 컨테이너.
  호스트에 포트를 노출하지 않고 compose 네트워크 내부(`mock-server:8888`)에서만
  접근 가능하다.

개별 서비스만 빌드/재기동하려면 `docker compose build web`,
`docker compose up -d gateway mock-server` 처럼 서비스명을 지정한다.
