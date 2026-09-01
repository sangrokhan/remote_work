# Seed — idle-reset TTFT 핵심 연구 질문, 1차 인프라 구축 (2026-09-01)

**출처**: 2026-09-01 AIPT ooo 재정의 세션의 후속 인터뷰. 핵심 연구 질문
"TCP idle-reset(slow-start-after-idle)이 LLM 멀티턴 트래픽의 TTFT에
미치는 영향"을 실측으로 검증하기 위한 실험 인프라 구축 1차분.

## 목표 (Restated Goal)

> net.ipv4.tcp_slow_start_after_idle을 0/1로 직접 토글해 idle-reset의
> 인과 효과를 mock/local_llm에서 먼저 재현 가능하게 검증하고, 이후
> public_ai(실제 인터넷)로 확인한다. 토글은 웹 UI에서 조작 가능해야 한다.

## 결정된 사항 (사용자 확정)

1. **실험 backend**: mock/local_llm(무료, 재현가능, Gateway netem 제어) 먼저
   → public_ai(실제 인터넷, 과금)로 재확인.
2. **효과 공리 방법**: 턴 간격을 흔드는 간접 관찰이 아니라, **sysctl을
   직접 0/1로 토글하는 가장 직접적인 인과 검증**.
3. **웹 UI 토글**: 필수 — CLI/curl이 아니라 브라우저에서 조작 가능해야 함.

## 구현 완료 (이번 세션)

### 인프라
- `aipt/core/idle_reset.py` (신규) — `net.ipv4.tcp_slow_start_after_idle`
  읽기/쓰기 모듈. `aipt.gateway.forwarding`과 동일한 `(ok, reason)`
  never-raises 계약.
- `aipt/backends/mock/server.py` — `/admin/idle-reset` (GET/POST) 엔드포인트
  추가 (mock-server는 이 프로젝트 자체 프로세스라 직접 훅 가능).
- `docker/idle_reset_admin.py` (신규) — local-llm용 사이드카 admin 서버.
  llama-server는 별도 바이너리(업스트림 이미지)라 인프로세스 훅이
  불가능해서 40081 포트에 별도 stdlib HTTP 서버를 **별도 프로세스**로
  띄움.
  - ⚠️ **버그 발견+수정**: 처음엔 스레드로 띄웠으나 `entrypoint_local_llm.py`가
    `os.execvp()`로 프로세스 이미지 전체를 llama-server로 교체하면서
    스레드가 통째로 사라짐. `subprocess.Popen`으로 완전히 분리된 프로세스로
    띄우도록 수정 — 실제 Docker 재빌드로 검증 완료.
- `aipt/web/routes_gateway.py` (신규) — 웹에서 두 컨트롤을 프록시:
  - `GET/POST /api/gateway/profile` — Gateway netem 프로파일 (B11, 지난
    세션에서 발견한 갭도 같이 해소됨)
  - `GET/POST /api/idle-reset?backend=mock|local_llm` — idle-reset 토글
- 웹 폼(`_experiment_form.html`)에 두 드롭다운 + Apply 버튼 추가, `app.js`에
  가시성 로직(mock/local_llm 카드에서만 노출) + fetch 핸들러 연결.
- **`docker-compose.yml`: mock-server/local-llm을 `privileged: true`로 전환**
  (사용자 확정). CAP_NET_ADMIN만으론 `/proc/sys/net/ipv4/tcp_slow_start_after_idle`
  쓰기가 Docker의 기본 read-only 마스킹에 막힘 — 이 프로젝트는 로컬 실험실
  용도라 격리 약화를 감내하기로 결정. **다른 프로젝트에 이 패턴을 복사하지 말 것.**

### 검증 (실제 Docker 기동)
- Gateway profile 프록시: 정상 동작 확인 (기존 API 그대로 프록시).
- idle-reset mock: GET/POST 200, 실제 `enabled: true→false→true` 값
  전환 + 영속 확인.
- idle-reset local_llm: 최초 시도 시 "connection refused"(스레드 버그) →
  수정 후 GET/POST 200, mock과 동일하게 실제 토글 확인.
- pytest 505 passed (신규 23개 포함), 회귀 없음.

## 아직 하지 않은 것 (다음 단계 TODO)

| # | 작업 | 비고 |
|---|---|---|
| E1 | **실제 A/B 실험 실행**: idle-reset enabled=1 vs 0 상태에서 동일한 멀티턴 대화(inference_delay로 idle 재현)를 mock/local_llm 양쪽에서 반복 실행해 TTFT/turns.csv 비교 통계 산출 | 이번 세션은 토글 인프라까지만; 실제 실험 실행/통계는 별도 |
| E2 | Gateway netem 프로파일(3g/satellite 등)과 idle-reset을 조합한 2x2 실험 설계 | RTT가 클수록 idle-reset 효과가 커질 것이라는 가설 검증 |
| E3 | public_ai(Gemini/OpenAI)에서 실측 재확인 — 단, public_ai는 이 프로젝트가 컨테이너 netns를 갖지 않으므로(실제 인터넷 종단) sysctl 토글이 불가능. 클라이언트(`web`) 쪽 sysctl만 토글 가능 — 클라이언트 send-side가 유의미한지 검토 필요 | **주의**: public_ai 응답은 서버(Google/OpenAI 인프라)가 보내므로 그쪽 idle-reset은 우리가 제어 불가. web 컨테이너의 클라이언트 측 idle-reset만 토글 가능하고, 이게 실험적으로 의미 있는지 다음 인터뷰에서 확인 필요 |
| E4 | 실험 결과를 DESIGN.md 또는 별도 research note에 문서화 | 통계/그래프 포함 |

## 다음 단계 제안

- E1(실제 A/B 실행)을 다음 세션에서 진행할지 확인.
- E3의 public_ai 실험 가능 여부(클라이언트 측만 제어 가능하다는 제약)를
  먼저 짚고 넘어가는 게 좋음 — 사용자가 "가장 직접적인 인과 검증"을
  원했는데, public_ai에서는 서버 측 idle-reset을 제어할 수 없으므로
  애초에 sysctl 토글 방식이 안 통한다는 걸 미리 알려야 함.
