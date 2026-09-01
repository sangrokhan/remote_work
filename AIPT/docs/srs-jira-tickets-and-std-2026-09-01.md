# AIPT — SRS(JIRA 티켓 초안) & STD(테스트 설계서)

- 작성일: 2026-09-01
- 근거 문서: `DESIGN.md` (§4.7~§5.2), `ARCHITECTURE.md` (§4~§6), `docs/engine_gateway_caching_seed.md`
- 범위: 2026-09-01 시점 문서-코드 정합성 감사(§5.2)와 §6 테스트 설계 갭에서 도출된 미구현/미해결 항목을
  JIRA 티켓 후보(SRS 대체)와 그에 대응하는 테스트 케이스(STD)로 정리한다.

## 1. SRS — JIRA 티켓 초안 (요구사항서 대체)

각 행은 실제 JIRA 티켓 생성 시 그대로 입력 가능한 형태로 작성했다.

| Epic Name | Summary | Feature Type | Description | Constraint | Validation Method | Validation |
|---|---|---|---|---|---|---|
| Network Gateway Web Integration | 웹 UI에 Network Profile 선택 기능 추가 (B11) | New Feature | 실험 설정 폼에 Network profile(`clean`/`wired`/`wireless`/`custom`) 드롭다운을 추가하고, 선택값을 Gateway의 `/gateway/profile` API로 전달하는 `aipt/web/routes_gateway` 모듈을 신규 구현한다. | `GATEWAY_HOST`/`GATEWAY_PORT` env는 이미 `web` 서비스에 주입되어 있으나 코드에서 전혀 사용되지 않는 dead config 상태이므로 반드시 활용해야 함. Gateway API는 런타임 프로파일 전환(기존 연결에도 즉시 반영)을 지원해야 함. | FastAPI TestClient 라우트 통합 테스트 + 실제 docker-compose 환경에서 프로파일 전환 후 `tc netem` 적용 확인 | 웹 폼에서 프로파일 선택 시 `/gateway/profile` API가 정상 호출되고, 이후 측정된 RTT가 선택 프로파일의 delay 설정값 근방으로 변화함을 확인 |
| Docker Infra 안정화 | local-llm 컨테이너 HEALTHCHECK 포트 버그 수정 | Bug Fix | `docker/Dockerfile.local_llm`이 상속하는 base 이미지의 HEALTHCHECK가 기본 포트 8080을 찌르지만, 실제 llama.cpp 서버는 `--port 40080`으로 기동되어 항상 `unhealthy`로 표시됨. HEALTHCHECK 정의를 40080으로 수정한다. | 기능 장애는 아님(end-to-end 정상 동작 실측 확인됨, TTFT 583ms). 우선순위는 낮으나 `docker compose ps` 오판 방지를 위해 필요. | `docker compose ps` 상태 확인 + 컨테이너 내부 `curl http://127.0.0.1:40080/health` 직접 확인 | 컨테이너 기동 후 healthcheck interval 경과 시 `docker compose ps`에서 local-llm이 `(healthy)`로 표시됨 |
| 테스트 커버리지 확충 | cache_protocol.py pytest 스위트 편입 | Test / Tech Debt | 2026-09-01 구현된 leaf-hash 요청 중복 제거 프로토콜(`aipt/core/cache_protocol.py`)이 현재 standalone assert 스크립트(`scripts/_smoketest_*.py` 4종)로만 검증되어 CI 기본 실행(`pytest -m "not live"`)에 포함되지 않음. `tests/core/test_cache_protocol.py`로 pytest화한다. | 기존 4개 스크립트(순수 유닛 / 실 HTTP e2e / 캐시 미스 복구 / 409 재전송 경로)가 커버하는 시나리오를 모두 이관해야 하며 회귀 없이 통과해야 함. | `pytest -m "not live"` 전체 스위트 실행 결과 확인 | 신규 테스트가 포함되어 기존 512 passed보다 pass 수가 증가하고 실패 없음 |
| 문서-코드 정합성 | DESIGN.md §4.8 아키텍처 다이어그램에 quic_mock 백엔드 반영 | Documentation | `QuicMockBackend`가 `Backend` 프로토콜에 정식 편입되고 `docker-compose.yml`에 `quic-mock-server` 5번째 서비스로 실존하지만, §4.8 Mermaid 다이어그램의 BACKENDS 서브그래프에는 3개(PublicAI/Mock/LocalLLM)만 표기되어 실제 구성과 불일치함. | 실제 코드/컨테이너 구성과 문서가 반드시 일치해야 함(SSoT 원칙 위반 금지). | 문서 리뷰 + `docker-compose.yml` 서비스 목록 대조 | 다이어그램에 quic-mock-server가 4번째 backend 노드로 표기되고 실제 서비스 5개(web/gateway/mock-server/local-llm/quic-mock-server) 구성과 일치 |
| 저장 정책 문서화 갱신 | DESIGN.md §4.7.1 실행 결과 저장 정책 stale 내용 갱신 | Documentation | "Public AI 기록만 영속 저장, 나머지는 인메모리 최근 50개"로 문서화되어 있으나, 실제로는 2026-08-27 변경으로 `aipt/web/store.py`가 `RUN_STORE_DIR`(`data/runs/`)에 모든 backend의 모든 run을 JSON으로 디스크 영속화하고 있음. 문서를 실제 동작에 맞게 갱신한다. | 과거 정책 원문은 역사적 기록으로 남기고, 실제 동작을 대표하는 문단만 갱신할 것(이미 §4.7.1 상단에 갱신 안내는 삽입되어 있음 — 본문까지 정리 필요). | 코드(`aipt/web/store.py`) 재확인 + 문서 리뷰 | §4.7.1 본문이 실제 `RUN_STORE_DIR` 디스크 영속화 동작을 정확히 서술하고, 과거 정책은 역사적 기록으로만 명확히 구분됨 |
| Gateway 관찰성 확장 | Gateway↔backend leg 트래픽 계측 확장 | Enhancement | 현재 `aipt/core/cwnd.py`/`capture.py`는 client↔gateway 구간만 관찰하며 Gateway↔backend 구간은 별도 관찰 대상이 아님(§4.7 명시된 "필요시 후속 확장" 항목). 향후 확장 시 이 구간의 cwnd/pcap 계측을 추가한다. | Gateway는 L3 순수 forwarding(TCP 상태 비인지, 페이로드/헤더 미검사)이므로 계측 추가 시 이 아키텍처 원칙을 훼손하지 않아야 함. | 신규 계측 코드에 대한 unit test + 실측 pcap 캡처 비교 | Gateway↔backend leg의 cwnd/pcap 데이터가 client↔gateway leg와 별도 파일로 수집·구분되고 payload가 손상 없이 동일하게 전달됨이 확인됨 |

## 2. STD — 테스트 케이스 (SW 실험/테스트 설계서)

위 JIRA 티켓과 1:1 또는 N:1로 대응하는 테스트 케이스.

| Test ID | Description | Precondition | Test Step | Expected Results |
|---|---|---|---|---|
| TC-AIPT-01 | 웹 UI Network Profile 선택이 Gateway API 및 실측 RTT에 반영되는지 검증 | docker-compose 4-service 토폴로지(web/gateway/mock-server 등) 기동 완료, `routes_gateway` 구현 완료 | 1) 웹 UI 실험 설정 폼에서 Network profile을 `wireless`로 선택 2) 실험 실행 3) 실행 중 `GET /gateway/profile` 호출로 현재 프로파일 확인 4) `aipt.core.probe`(idle-gap HTTP PING)로 RTT 측정 | `/gateway/profile` 응답이 `wireless` 프로파일 파라미터를 반환하고, 측정 RTT가 `wireless` 프로파일의 delay 설정값 근방(±허용오차)으로 나타남 |
| TC-AIPT-02 | local-llm 컨테이너 HEALTHCHECK 정상화 검증 | docker-compose로 local-llm 컨테이너 기동 완료, Dockerfile HEALTHCHECK가 40080으로 수정 완료 | 1) `docker compose up -d local-llm` 2) HEALTHCHECK interval(예: 30초) 경과 대기 3) `docker compose ps` 실행 4) 컨테이너 내부에서 `curl http://127.0.0.1:40080/health` 직접 호출 | `docker compose ps`의 local-llm STATUS가 `(healthy)`로 표시되고, curl 응답이 `{"status":"ok"}`와 일치 |
| TC-AIPT-03 | cache_protocol.py pytest 스위트 회귀 검증 | `tests/core/test_cache_protocol.py` 신규 작성 완료, venv 활성화 | 1) `pytest -m "not live" tests/core/test_cache_protocol.py -v` 실행 2) 전체 스위트 `pytest -m "not live"` 실행 | 신규 테스트의 모든 케이스(leaf 순회/치환, path↔label 라운드트립, `CacheMiss` 예외, 원본 body 불변성 등)가 pass, 전체 스위트 pass 수가 기존 512보다 증가하고 실패 없음 |
| TC-AIPT-04 | 캐싱 기능 트래픽 절감 실측 재현 검증 (성능 회귀 방지) | docker-compose 4-service 토폴로지(web → Network Gateway L3/L4 → engine Gateway L7 → llama-server) 기동, `records/perf_short_smoketest.json` 시나리오 존재 | 1) `scripts/measure_perf_cache_savings.py`를 `X-AIPT-Cache: enable` off로 실행(baseline) 2) 동일 스크립트를 on으로 실행(cached) 3) `data/runs/cache_savings_multiturn.csv`에서 턴별 `req_payload_bytes`/`wire_sent` 비교 | 20턴 누적 기준 `req_payload_bytes` 절감률 87.2% 근방, `wire_sent` 절감률 86.3% 근방으로 재현되며, turn 0의 절감률은 0 이하(미세 음수, 캐시맵 오버헤드) 유지 |
| TC-AIPT-05 | DESIGN.md §4.8/§4.7.1 문서-코드 정합성 검증 | 최신 `docker-compose.yml`, `aipt/web/store.py` 코드 확보 | 1) `docker-compose.yml`의 서비스 목록 확인(5개: web/gateway/mock-server/local-llm/quic-mock-server) 2) DESIGN.md §4.8 Mermaid 다이어그램의 BACKENDS 노드 수 확인 3) `aipt/web/store.py`의 `RUN_STORE_DIR` 저장 로직 확인 4) DESIGN.md §4.7.1 서술과 대조 | §4.8 다이어그램에 quic-mock-server를 포함한 4개 backend 노드가 존재해 실제 서비스 구성과 일치하고, §4.7.1이 "모든 backend의 run을 RUN_STORE_DIR에 디스크 영속화"로 정확히 서술됨 |
| TC-AIPT-06 | Gateway↔backend leg 신규 계측 단위 검증 | Gateway↔backend leg 계측 코드 구현 완료(신규), tc/netlink 사용 가능한 테스트 환경(`@pytest.mark.live` 대상 가능) | 1) mock-server 대상 실험 실행 시 Gateway↔backend leg pcap 캡처 활성화 2) client↔gateway leg pcap과 Gateway↔backend leg pcap을 각각 저장 3) 두 pcap의 바이트 수/시퀀스 비교 | 두 leg의 pcap이 독립 파일로 저장되고, Gateway가 순수 L3 forwarding임을 반영해 payload가 손상 없이 동일하게 전달됨(TCP 헤더/타이밍만 차이) |

## 3. 비고

- 위 6개 항목은 모두 2026-09-01 문서-코드 정합성 감사(DESIGN.md §5.2)와 §6 테스트 설계 갭에서 실제로
  식별된 미구현/미해결 사항을 근거로 작성했으며, 임의로 지어낸 요구사항이 아니다.
- JIRA 실입력 시 Epic Name은 팀 컨벤션에 맞는 기존 Epic에 매핑하거나 신규 Epic으로 생성.
- STD 테스트 케이스 중 TC-AIPT-01/04/06은 실제 docker-compose 스택 또는 `@pytest.mark.live` 환경이
  필요한 통합/실측 테스트이며, TC-AIPT-02/03/05는 상대적으로 가벼운 검증이다.
