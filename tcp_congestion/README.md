# tcp_congestion

Multi-turn LLM 대화에서 idle 구간(추론 대기) 후 TCP cwnd가 리셋되는 현상을,
누적 컨텍스트로 매 턴 커지는 페이로드와 함께 실측하는 실험실.

## 측정 대상

| 지표 | 방법 |
|---|---|
| `cwnd`, `rto`, `rtt`, `delivery_rate` | netlink sock_diag, 2ms 주기 연속 샘플링 (`native/cwnd_monitor.c`) |
| idle 구간 RTT | HTTP PING (`probe.py`) — delivery_rate는 갱신하지 않음 |
| 턴별 프롬프트 크기 | 누적 컨텍스트: turn N = turn 1..N-1 전체 + 새 사용자 입력 |
| idle 리셋 시점 | 다음 전송 시 cwnd가 IW(10)로 떨어지는 정확한 샘플 (스냅샷 추론 아님) |

## 구성 요소

| 모듈 | 역할 |
|---|---|
| `tcp_congestion/server.py` | HTTP/1.1 keep-alive 서버 (`/ping`, `/health`, `POST /inference-mock`) |
| `tcp_congestion/probe.py` | idle 구간 RTT 전용 HTTP PING |
| `tcp_congestion/tcpinfo.py` | 1회성 `getsockopt(TCP_INFO)` 스냅샷 (delivery_rate 등) |
| `tcp_congestion/cwnd.py` + `native/cwnd_monitor.c` | 연속 netlink 모니터링 (token_traffic에서 이식) |
| `tcp_congestion/conversation.py` | 멀티턴 시나리오: 누적 컨텍스트 성장 + idle probe + cwnd 추적 + (선택) 패킷 캡처 + 알고리즘 선택 |
| `tcp_congestion/congestion.py` | 4개 알고리즘(cubic/reno/bbr/vegas) 로드 여부 + qdisc 확인, 웹 UI 안내 문구 생성 |
| `tcp_congestion/capture.py` | tcpdump 패킷 캡처 (run 시작~종료 구간, NET_RAW 필요) |
| `tcp_congestion/export.py` | 결과를 CSV로 변환 (cwnd 연속 샘플 / 턴별 요약) |
| `tcp_congestion/app.py` + `templates/index.html` | 웹 프론트엔드: 설정 폼(알고리즘 선택 포함) → 실행 → cwnd 곡선 차트 → CSV/pcap/zip 다운로드 |
| `tcp_congestion/netem.py` | `tc netem` 지연 주입 (환경변수 기반) |

## 4개 congestion-control 알고리즘 비교

웹 UI 상단의 "Congestion algorithm" 배너가 매 페이지 로드 시 커널 상태를 확인합니다:

- `cubic`/`reno`/`bbr`/`vegas` 4개가 모두 `tcp_available_congestion_control`에 있는지
- 클라이언트 컨테이너의 네트워크 인터페이스(`eth0`) qdisc가 `fq`인지 (BBR의 정밀 페이싱 요구사항, `fq_codel`은 CoDel 조기 드롭이 cubic/reno/vegas 신호를 오염시킴)

준비가 안 되어 있으면 배너에 정확한 `modprobe`/`tc qdisc replace` 명령이 표시됩니다. 폼의 알고리즘 드롭다운에서 아직 로드 안 된 항목은 "(not loaded)"로 비활성화됩니다.

실행 시 선택한 알고리즘은 `connect()` 전에 `TCP_CONGESTION` 소켓옵션으로 적용되며, 결과 JSON의 `algorithm`(실제 적용값, `getsockopt`로 재확인)과 `algorithm_requested`(요청값)가 다르면 UI에 경고가 뜹니다 — 로드 안 된 알고리즘을 요청했을 때 조용히 기본값으로 폴백되는 것을 방지합니다.

## 왜 연속 모니터링인가

1회성 TCP_INFO 스냅샷 2~3개로는 "idle 후 다음 전송에서 cwnd가 리셋된다"는
사실을 정확히 잡을 수 없다 (리셋이 idle 시작과 idle 종료 사이 어느 시점에
일어났는지 모름). `cwnd.Monitor`는 연결이 열려 있는 전체 시간 동안 2ms 주기로
샘플링하므로, 리셋이 실제로 일어난 정확한 샘플(`reset_events`)을 잡아낸다.

## 빠른 시작

### 로컬 (Docker 없이)

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/ -v      # 60 tests

# C 헬퍼 빌드 (netlink 모니터링)
cc -O2 -Wall -o native/cwnd_monitor native/cwnd_monitor.c

# 웹 UI 실행
.venv/bin/python -c "
import threading
from tcp_congestion import server
s = server.Server(host='127.0.0.1', port=8888)
threading.Thread(target=s.serve_forever, daemon=True).start()
import uvicorn
from tcp_congestion.app import create_app
uvicorn.run(create_app(), host='127.0.0.1', port=8080)
"
# → http://127.0.0.1:8080 접속, host=127.0.0.1 port=8888로 설정 후 Run
```

### Docker

```bash
docker compose up --build
# → http://localhost:10000 접속 (웹 UI)
#   서버는 http://localhost:8888 (직접 curl 가능)

# 인위적 RTT: 클라이언트→서버 20ms, 서버→클라이언트 20ms
CLIENT_NETEM_DELAY_MS=20 SERVER_NETEM_DELAY_MS=20 docker compose up --build
```

패킷 캡처를 켜려면 웹 UI에서 "Capture packets (tcpdump)" 체크박스를 선택.
컨테이너에는 `NET_RAW` capability가 이미 부여되어 있음. 캡처된 pcap은
`./data/pcaps/`에 저장되며 호스트에서 Wireshark로 바로 열 수 있다.

## 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `CLIENT_NETEM_DELAY_MS` | `0` | 클라이언트 → 서버 방향 단방향 지연 (ms) |
| `SERVER_NETEM_DELAY_MS` | `0` | 서버 → 클라이언트 방향 단방향 지연 (ms) |
| `CLIENT_NETEM_IFACE` / `SERVER_NETEM_IFACE` | `eth0` | 컨테이너 인터페이스 |
| `SERVER_HOST_PORT` | `8888` | 호스트→서버 포트 포워딩 |
| `CLIENT_HOST_PORT` | `10000` | 호스트→웹UI 포트 포워딩 |

## 웹 UI에서 설정하는 값

| 필드 | 의미 |
|---|---|
| Turns | 대화 턴 수 |
| System prompt | 턴 1에만 1회 실리는 시스템 프롬프트 크기 (이후 history에 포함되어 유지, 재전송 안 함) |
| Turn user message | 매 턴 새로 추가되는 사용자 메시지 크기 |
| Mock response size | 서버가 반환하는 응답 크기 (누적 컨텍스트 계산용) |
| Inference delay per turn | `/inference-mock` 응답 지연 (prefill 시뮬레이션) |
| Idle duration per turn | 요청 간 idle 시간 (사용자 생각시간 시뮬레이션) |
| Ping probe interval | idle 중 RTT probe 간격 (아래 체크박스가 켜져 있을 때만 적용) |
| Send HTTP PING during idle gap | 체크 해제 시 idle 구간 동안 HTTP PING을 전혀 보내지 않음 -- idle 시간 자체는 그대로 유지되며, probe 트래픽(작은 keepalive성 요청)이 cwnd/idle-reset 관측에 주는 영향을 배제하고 싶을 때 사용. 결과 JSON의 `ping_probes_enabled`로 실제 적용 여부 확인 가능 |

## 결과 읽기

`POST /api/run` 응답 JSON (= `conversation.run()` 반환값):

```json
{
  "algorithm_requested": "bbr",
  "algorithm": "bbr",
  "algorithm_error": "",
  "idle_resets": 1,
  "peak_cwnd": 18,
  "final_cwnd": 10,
  "reset_events": [{"t_ms": 6254.1, "local": "...", "from": 19, "to": 10, "idle_ms": 1}],
  "samples": [...],          // 연속 cwnd 샘플 (전체 대화 기간)
  "turns": [
    {"turn": 0, "prompt_bytes": 200000, "request_ms": 1.4, "idle_ms": 2500},
    {"turn": 1, "prompt_bytes": 400100, "request_ms": 2.7, "idle_ms": 2500}
  ],
  "probes": [{"turn": 0, "samples": [{"ts": ..., "rtt_ms": 20.1}, ...]}],
  "pcap": {"ok": true, "file": "capture_..._<token>.pcap", "bytes": 9349, ...}
}
```

`idle_resets`가 1 이상이면 실제로 idle 후 slow start 재진입이 관측된 것.
`reset_events`가 정확히 어느 샘플(t_ms)에서 cwnd가 IW로 떨어졌는지 보여준다.
`algorithm`은 `getsockopt(TCP_CONGESTION)`으로 재확인한 실제 적용값 -- 요청한
`algorithm_requested`와 다르면 커널에 그 알고리즘이 없어서 소켓 기본값으로
폴백된 것이니 `algorithm_error`를 확인.

## 산출물 다운로드

웹 UI에서 실행 완료 후 나타나는 링크, 또는 API 직접 호출:

| 엔드포인트 | 내용 |
|---|---|
| `GET /api/download/cwnd.csv` | 연속 cwnd 샘플 전체 (t_ms, snd_cwnd, rtt_us, delivery_rate 등) |
| `GET /api/download/turns.csv` | 턴별 요약 (prompt_bytes, request_ms, idle_ms, probe RTT 통계) |
| `GET /api/download/pcap` | 캡처된 pcap (`capture=true`로 실행했을 때만 존재, Wireshark로 오픈 가능) |
| `GET /api/download/bundle.zip` | 위 세 파일을 한 번에 zip으로 묶어서 다운로드. 파일명이 `tcp_congestion_<algorithm>_<label>.zip` 형식이라 cubic/reno/bbr/vegas 4번 실행 결과를 받아도 파일명만으로 구분 가능 (zip 내부 항목도 `<algorithm>_cwnd.csv` 식으로 접두어가 붙음) |

가장 최근 1회 실행 결과만 메모리에 유지되며, 새로 실행하면 이전 결과는 덮어써진다.

