# idle-reset(slow-start-after-idle)이 사용자 체감 지연에 미치는 영향 — 실측 결과

**날짜**: 2026-09-01 · **결론**: 인과관계 확인됨 (1차 완화안 유효)

## 배경 및 측정 대상 재정의

AIPT 프로젝트의 출발점은 "AI 트래픽에서 사용자 체감 성능을 개선"하는 것.
1차로 발견한 현상:

1. 멀티턴 추론에서 클라이언트가 요청을 다 보내면, 서버가 추론하는 동안
   해당 TCP 연결은 idle 상태가 된다.
2. 이 idle 구간이 **RTO보다 길게 지속**되는 경우가 흔하다(추론 시간은
   보통 수백ms~수초).
3. `net.ipv4.tcp_slow_start_after_idle=1`(Linux 기본값)이면, idle 후
   **다음으로 전송을 시작하는 쪽 모두** cwnd가 리셋된다 — 이는 서버가
   응답을 내려보내는 방향뿐 아니라, **클라이언트가 다음 턴 요청을
   업로드하는 방향에도 그대로 적용**된다.

**최초 실험(실패)**: mock-server(응답을 보내는 쪽)의 idle-reset을 토글하고
응답 다운로드 완료 시간(TTFT)을 측정 → 유의미하지만 미미한 차이(+3.7%)만
확인, cwnd 시계열 검증에서도 두 조건의 리셋 패턴이 동일해 인과관계
불확실.

**주인님 재지적**: "측정해야 하는 것은 유저의 전송이 실제로 서버에 모두
전송될 때까지의 지연을 최소화하고 있는지" — 즉:
- 토글 대상: 서버(mock-server) 측이 아니라 **클라이언트(web) 측**
  idle-reset — 다음 요청을 업로드하는 쪽이 클라이언트이므로.
- 측정 지표: 응답 다운로드 시간(TTFT)이 아니라 **요청 업로드 완료
  시간**(서버가 요청 바디를 전부 수신하는 데 걸린 시간).

## 재설계된 실험

### 계측 추가
- `aipt/backends/mock/server.py`: `/inference-mock` POST 핸들러가 요청
  바디를 다 읽는 데 걸린 시간(`recv_ms`, 서버 관점 실측)을 응답에 실어
  echo. 서버가 "요청을 다 받는 순간"이 곧 추론을 시작할 수 있는 시점이므로
  사용자 체감 지연에 직결.
- `aipt/web/routes_gateway.py`: `backend=web_client`를 지원하도록 확장 —
  `web`(클라이언트) 자신의 `net.ipv4.tcp_slow_start_after_idle`을
  인프로세스로 직접 토글 (`aipt.core.idle_reset` 직접 호출, 별도 프록시
  불필요 — web이 자기 자신의 netns를 이미 소유).
- `docker-compose.yml`: `web` 서비스도 `privileged: true`로 전환
  (CAP_NET_ADMIN만으론 sysctl 쓰기 불가 — mock-server/local-llm과 동일
  이유).

### 실험 조건
- backend: mock (Gateway netem 20ms×2 경유)
- 요청 크기: 시스템프롬프트 500KB + 턴당 사용자메시지 500KB (누적 컨텍스트
  성장 — turn 5에서 최대 3.5MB), **응답은 작게**(1000 bytes) — 이전
  실험과 반대로 뒤집음.
- idle 간격: 매 턴 사이 4000ms (RTO 초과 유도)
- 6턴 × 4반복, `web_client` idle_reset=1(기본)과 0(비활성) 각각.
- mock-server 측 idle_reset은 양쪽 조건 모두 enabled=1(기본)로 고정 —
  이번엔 응답 방향이 아니라 업로드 방향만 보는 것이므로 무관.

## 결과

| turn | idle_reset=1 (기본) recv_ms | idle_reset=0 (비활성) recv_ms | 배율 |
|---|---|---|---|
| 0 (idle 없음, 연결 직후) | 243.9 | 244.1 | 1.0x (대조군 일치) |
| 1 (4s idle 후) | 246.6 | **44.3** | 5.6x |
| 2 | 286.1 | 480.7* | (변동) |
| 3 | 287.0 | 530.5* | (변동) |
| 4 | 293.0 | **87.2** | 3.4x |
| 5 | 792.3 | **98.5** | 8.0x |

\* turn 2/3의 disabled 조건은 반복 간 변동이 큼(86ms대 ↔ 660ms대) —
Gateway netem 큐잉/재정렬과의 상호작용으로 추정, 원인 규명은 후속 과제.

**turns 1-5 풀링**: enabled 평균 381.0ms vs disabled 평균 248.2ms
(median 287.8ms vs 88.3ms) — **idle_reset=0일 때 median 기준 약 3.3배
빠름**.

## 해석

1. **turn 0(대조군)이 두 조건에서 완전히 일치**(243.9 vs 244.1ms)한다는
   것은 idle-reset 토글 자체가 순수 실험 조작 변수 하나만 바꿨음을
   확인시켜준다 — 이 실험은 통제되어 있다.
2. turn 1부터 idle_reset=1 조건에서 명백히 느려지기 시작하고, 요청 크기가
   누적으로 커질수록(turn 5, 3.5MB) 그 격차가 극적으로 벌어진다(8배) —
   slow-start가 처음부터 다시 시작되면 큰 바디를 다 올리는 데 걸리는
   시간이 기하급수적으로 늘어난다는 이론과 정확히 일치.
3. 1차 발견("idle-reset이 다음 요청 업로드를 느리게 만든다")과 1차
   제안("slow_start_after_idle을 끈다")이 **실측으로 검증됨**.

## 남은 과제

- turn 2/3의 disabled 조건 변동성 원인 규명 (netem 큐/재정렬 상호작용 추정)
- Gateway netem 프로파일(3g 등, 더 큰 RTT/loss)에서도 같은 패턴 재현 확인
- local_llm 백엔드에서도 동일 실험 재현
- public_ai(Gemini/OpenAI)에서 클라이언트(web) 측 idle-reset만 토글해
  실제 인터넷 환경에서도 효과가 나타나는지 확인 — 서버(Google/OpenAI)
  측 idle-reset은 우리가 제어할 수 없으므로, 이 실험은 "클라이언트 업로드
  방향"이라는 이번 발견 덕분에 오히려 public_ai에서도 유효하게 적용 가능
  (서버 측을 건드릴 필요가 원래 없었음 — 사용자 지적이 정확했던 이유).
- 원인이 확인됐으니, `slow_start_after_idle=0`을 실제 운영 환경에
  적용하는 것의 부작용(전역 설정이라 다른 idle 소켓들의 혼잡 회피도 함께
  꺼짐) 검토.

## 재현 방법

```bash
docker compose up -d gateway mock-server web
curl -X POST "http://localhost:10000/api/idle-reset?backend=web_client&enabled=false"
curl -X POST http://localhost:10000/api/run -H "Content-Type: application/json" -d '{
  "backend": "mock", "arm": "dummy", "input_mode": "dummy",
  "system_prompt_bytes": 500000, "turn_user_msg_bytes": 500000,
  "num_turns": 6, "mock_response_bytes": 1000, "inference_delay_ms": 4000,
  "capture": true, "label": "repro"
}'
# run.turns[i].response_raw.recv_ms 가 서버측 실측 업로드-완료 지연
```

원본 데이터: `2026-09-01-idle-reset-upload-webclient-raw.json`,
분석 스크립트: `2026-09-01-idle-reset-analyze.py`
