# Seed: engine Gateway 요청 중복 전송 절감(leaf-hash 캐싱) 프로토콜

- **상태**: 설계 확정, 구현 미착수 (2026-09-01 Slack 인터뷰 결과)
- **범위**: 이번 문서는 **프로토콜 설계만**. `docker/engine_gateway.py`의
  캐싱 로직 구현과 `web` 클라이언트 코드 수정은 별도 후속 작업.
- **적용 backend**: `local-llm`(engine Gateway)만. `public_ai` 등 다른
  backend는 대상 아님 — AIPT의 local_llm 실험 목적에 한정.

## 1. 문제 정의

LLM 멀티턴 대화는 매 요청마다 `messages` 배열 전체(이전 턴 누적분 포함)를
재전송하는 게 일반적인 API 관례다. 턴이 쌓일수록 요청 바디의 대부분이
"이미 서버가 받았던 내용"의 반복이 되어 HTTP 전송량이 선형으로 증가한다.
이 낭비는 **애플리케이션 레벨(클라이언트 코드)이 아니라 HTTP 프로토콜
계층에서** 능동적으로 줄이자는 것이 이 설계의 동기다. 클라이언트는 응답을
캐싱할 필요가 없다(매번 새 응답을 받으므로) — 캐싱 대상은 오직
**클라이언트가 이미 전송했던 요청 내용**이다.

## 2. Opt-in 신호

양단(web ↔ engine Gateway) 모두 이 로직을 지원할 때만 동작해야 하므로,
전용 헤더로 활성화한다:

```
X-AIPT-Cache: enable
```

이 헤더가 없으면 engine Gateway는 지금처럼 순수 패스스루로만 동작한다
(기존 구현 변경 없음).

## 3. 세션 경계: TCP 커넥션

- 캐시 저장소는 **session 단위**로만 유지되고 전역 공유하지 않는다.
- session의 정의는 **HTTP keep-alive TCP 커넥션 그 자체**다. 별도의 세션
  ID 헤더/토큰을 발급하지 않는다.
- 구현상 자연스러운 지점: `engine_gateway.py`가 `BaseHTTPRequestHandler`
  기반이므로, 커넥션이 유지되는 동안 같은 핸들러 인스턴스가 여러 요청을
  처리한다(`protocol_version = "HTTP/1.1"`). 캐시 딕셔너리를 그 핸들러
  인스턴스(또는 커넥션 단위 컨텍스트)에 붙이면, 커넥션 종료 시 자동으로
  해당 세션의 캐시가 사라진다 — 별도의 만료/정리 로직 불필요.
- **TTL은 이번 설계에서 고려하지 않는다** (커넥션 생존 = 캐시 생존).

## 4. 캐싱 대상 단위: leaf 값, depth 무관하게 "가장 마지막 depth"만

- 캐싱은 JSON 트리의 **leaf(최하위) 값 단위**로만 이루어진다. 객체나
  배열 전체를 통째로 치환하지 않는다.
- 각 leaf 값이 **직렬화했을 때 특정 바이트 크기 임계값 이상**일 때만
  캐싱 후보가 된다 (임계값은 조정 가능한 설정값, 예: 200 bytes).
- 예: `messages[i].content`처럼 긴 텍스트 leaf는 대상이 되지만,
  `messages[i].role`처럼 항상 짧은 leaf는 임계값을 넘지 않아 자연히
  대상에서 제외된다 — 별도의 필드 화이트리스트가 필요 없다.

## 5. 와이어 포맷

### 5.1 최초 등장(캐시 미스, 최초 전송)

hash가 아직 로컬/서버 어느 쪽에도 없으므로 **원본 값을 그대로 전송**한다.
전송과 동시에 양쪽이 각자 동일한 해시 함수로 hash를 계산해 로컬에
저장해 둔다(클라이언트: 보낸 값 저장, 서버: 받은 값 저장) — 이 시점에는
왕복 확인 없이 각자 독립적으로 저장한다.

### 5.2 재등장(캐시 히트)

클라이언트가 로컬 저장소에 동일 hash가 있는 leaf를 보내야 할 때, 그
leaf의 값 자체를 hash 문자열로 치환하고, 어느 경로가 치환됐는지 알려주는
보조 필드를 최상위에 추가한다.

**Before (원본):**
```json
{
  "model": "local-model",
  "stream": false,
  "messages": [
    {"role": "system", "content": "You are a helpful assistant... (250 bytes)"},
    {"role": "user", "content": "이전 사용자 발화... (300 bytes)"},
    {"role": "assistant", "content": "이전 LLM 응답... (400 bytes)"},
    {"role": "user", "content": "새로운 질문입니다"}
  ]
}
```

**After (leaf 단위 치환, 클라이언트가 실제 전송):**
```json
{
  "model": "local-model",
  "stream": false,
  "messages": [
    {"role": "system", "content": "a1b2c3d4e5"},
    {"role": "user", "content": "f6a7b8c9d0"},
    {"role": "assistant", "content": "e1f2a3b4c5"},
    {"role": "user", "content": "새로운 질문입니다"}
  ],
  "$aipt_cache_map": {
    "hashed_0": "\"messages\".0.\"content\"",
    "hashed_1": "\"messages\".1.\"content\"",
    "hashed_2": "\"messages\".2.\"content\""
  }
}
```

- `role` 키는 값이 짧아 임계값을 넘지 않으므로 항상 원본 그대로.
- `$aipt_cache_map`의 key는 `hashed_N`(순번 라벨, 조회용 식별자가
  아니라 사람이 읽기 위한 라벨), value는 JSON 경로를 문자열로 표기한
  것 — 문자열 key는 따옴표로 감싸고 배열 index는 숫자 그대로 이어붙인다
  (`"messages".0."content"`). 이 필드는 서버에게 "이 경로의 값은
  리터럴이 아니라 hash 참조다"를 알려주는 역할만 한다.

## 6. 처리 흐름

### 클라이언트(web) 송신 시
1. 전송할 요청 바디의 모든 leaf를 순회, 임계값 이상인 leaf만 검사.
2. 로컬(이 커넥션 세션) 캐시에 동일 hash가 있으면 → 그 leaf 값을 hash로
   치환 + `$aipt_cache_map`에 경로 기록.
3. 로컬 캐시에 없으면(최초) → 원본 값 그대로 두고, hash를 계산해 로컬에
   저장(다음 턴부터 재사용 대비). `$aipt_cache_map`에는 기록하지 않음.
4. 사전 확인 왕복(서버에 "이 hash 아냐?" 물어보는 요청)은 하지 않는다.

### 서버(engine Gateway) 수신 시
1. `X-AIPT-Cache: enable` 헤더 확인. 없으면 기존 패스스루 그대로.
2. `$aipt_cache_map`에 나열된 경로는 → 이 커넥션(세션)의 서버측 캐시
   저장소에서 hash로 원본 값을 조회해 그 경로에 복원.
   - 캐시에 없으면(예: 서버 프로세스 재시작으로 세션 캐시 소실) 에러
     처리 방식은 **후속 설계 과제**로 남긴다(이번 Seed 범위 밖).
3. `$aipt_cache_map`에 없는 임계값 이상 leaf(=원본 그대로 온 값)는 →
   동일한 해시 함수로 hash를 계산해 서버측 세션 캐시에 저장(클라이언트와
   대칭 동작).
4. 위 치환/복원을 마친 **완전한 원본 형태의 body**를 llama-server로
   포워딩한다 — llama-server는 이 프로토콜을 전혀 모르므로 항상 완전한
   JSON만 받아야 한다.

## 7. 스트리밍(stream:true)과의 관계

- 이 설계는 **요청(request) 캐싱**에 관한 것이고, 기존
  `docker/engine_gateway.py`의 stream/non-stream 분기(2026-09-01
  1차 구현)는 **응답(response) 처리 방식**에 관한 것이라 서로 직교한다.
- 캐시 맵 해석/복원은 요청 바디 파싱 시점(스트림 여부 판단 이전)에
  이루어지므로, `stream:true`/`false` 여부와 무관하게 항상 수행된다.
- 기존 구현의 `on_cacheable_request`/`on_cacheable_response` 훅(현재
  no-op)은 이번 설계와는 다른 용도로 이미 자리잡아 둔 것 — "LLM 응답
  자체를 캐싱"하는 상상이었으나 실제 목적은 "요청 컨텍스트 중복 제거"로
  정정됨. 이 두 훅을 이번 설계 구현에 재사용할지, 별도 미들웨어로 새로
  둘지는 구현 단계에서 결정.

## 8. 확정된 설계 결정 사항 (2026-09-01 인터뷰 완료)

1. **해시 함수**: SHA-256, hexdigest **앞 20자(80 bit)**로 자름. 표준
   라이브러리(`hashlib`)만으로 충분, 별도 패키지 설치 불필요.
   - 충돌 확률 검증(생일 역설 근사식 `p ≈ n²/(2×2^bits)`,
     2026-09-01 계산): session 내 캐싱 leaf 1,000개 기준 80bit는
     p ≈ 4.1×10⁻¹⁹, 100,000개여도 p ≈ 4.1×10⁻¹⁵ — 이 실험 규모(session당
     leaf 수십~수백 개)에서는 사실상 무시 가능한 수준. 16자(64bit)도
     이미 충분히 낮지만(1,000개 기준 2.7×10⁻¹⁴), 계산 비용 차이가 없어
     20자를 택함 — 해시 충돌은 조용한 오동작(다른 콘텐츠를 같은 것으로
     착각해 잘못된 원본으로 복원)으로 이어지므로 비용 없이 더 안전한
     쪽을 선택.

2. **캐시 미스 시 처리**: 서버가 `$aipt_cache_map`에 명시된 hash를 자신의
   세션 캐시에서 찾지 못하면 (예: 커넥션이 아직 살아있는데도 서버 프로세스
   재시작 등으로 상태가 유실된 경우) **HTTP 409 Conflict**를 반환하고,
   응답 바디에 미스난 경로 목록을 담는다:
   ```json
   {"error": "cache_miss", "missing_paths": ["\"messages\".0.\"content\""]}
   ```
   클라이언트는 이 목록을 보고 **해당 경로만** 원본 값으로 채워 넣어
   재전송한다(전체 요청을 다시 만들 필요 없음 — 나머지 경로는 여전히
   hash로 유지). 재전송된 원본 값은 서버가 다시 hash로 저장한다(§6
   "최초 등장" 절차와 동일).

3. **임계값**: leaf 값 직렬화 기준 **200 bytes** 이상일 때만 캐싱 후보.

4. **클라이언트(web) 구현 위치**: `aipt/backends/local_llm/gateway.py`의
   `Gateway.send()` 안, **`on_request` 훅 실행 지점**
   (`self._run_request_hooks(req)`, `json.dumps` 이전 · `post()` 호출
   이전). 이 시점엔 `req["body"]`가 아직 dict라서 leaf 경로 순회/치환이
   가능하고, JSON으로 직렬화된 뒤에는 구조 정보가 사라지므로 이 지점이
   유일하게 유효한 위치다. 새 훅 함수를 만들어
   `self._gateway.on_request(cache_hook)`로 등록하는 방식으로 구현하며,
   기존 `Gateway.send()` 코드 자체는 변경하지 않는다.

   **주의(폴더 네이밍 관련)**: `aipt/backends/`라는 이름 때문에 "서버측
   (수신측) 코드"로 오인하기 쉽지만, 이 디렉토리는 전부 **web 프로세스
   안에서 실행되는 발신측(클라이언트) 어댑터**다. 실제 수신측 코드는
   backend마다 다른 곳에 있다:
   - `local_llm`: 수신측은 `docker/engine_gateway.py`(이번에 만든 L7
     프록시) + 진짜 `llama-server` 바이너리(재구현하지 않음, `docker/
     entrypoint_local_llm.py`가 exec). AIPT 자체 수신측 코드 없음.
   - `mock`: 수신측은 `aipt/backends/mock/server.py` — 이름은 같은
     폴더에 있지만 컨테이너 안에서 별도 프로세스로 기동되는 진짜 서버.
   - `public_ai`: 수신측 없음 (외부 Gemini/OpenAI API).
   이번 캐싱 로직은 **local_llm 발신측(`aipt/backends/local_llm/
   gateway.py`)에만** 구현되므로, 이 코드 경로를 타는 것 자체로
   "local-llm 요청에서만 활성화"라는 요건이 자동 충족된다(다른 backend는
   각자 별개 경로를 쓰므로 영향받지 않음).

5. **전송 스택**: `Gateway.send()`는 `aipt.core.wire.session()`을 통해
   **`requests.Session`**(소켓 바이트 카운팅을 위해 커스텀
   `HTTPAdapter`/urllib3 커넥션 클래스를 장착한 버전, `aipt/core/
   wire.py`)으로 실제 전송을 수행한다. 캐싱 로직은 이 전송 스택보다
   위(dict 레벨)에서 동작하므로 `wire.py` 자체는 변경할 필요가 없다.

6. **커넥션(세션)별 상태 보관 지점**:
   - **서버측**: `docker/engine_gateway.py`의 `_Handler`
     (`BaseHTTPRequestHandler` 서브클래스) 인스턴스 속성으로 캐시 딕셔너리를
     둔다. `ThreadingHTTPServer`는 keep-alive 커넥션 동안 같은 핸들러
     인스턴스가 여러 요청을 처리하므로, 그 인스턴스 생존 기간이 곧 세션
     생존 기간과 일치한다.
   - **클라이언트측**: `LocalLLMBackend.connect()`가 실행마다 `Gateway`
     인스턴스를 새로 만들고(`self._gateway = Gateway(...)`), 그 인스턴스가
     실행 전체에서 재사용되며 `wire.session()`의 풀링된 커넥션도 그 실행
     동안 유지된다. 따라서 `Gateway` 인스턴스 자체에 캐시 딕셔너리를 두면
     TCP 커넥션 생존 기간과 자연스럽게 맞아떨어진다.

## 9. 남은 구현 세부사항 (설계는 확정, 코드 작성 시 다뤄야 함)

이번 인터뷰에서 검토하며 발견한, 코드 작성 시 실제로 부딪힐 지점들 —
설계 원칙은 이미 확정되어 있으니 코드화하면서 그대로 처리하면 되지만,
누락 없이 짚어둔다:

1. **409 응답을 `Gateway.send()`의 기존 에러 처리와 어떻게 구분할지**:
   `Gateway.send()`는 현재 `resp.status_code != 200`이면 무조건
   `result.error`로 기록하고 반환한다(재시도 없음). 409(cache_miss)는
   이 기존 분기와 별개로 **먼저 가로채서** "미스난 경로만 원본으로 채워
   1회 재전송"하는 별도 루프를 거쳐야 한다 — 그 재전송도 실패하면 그때
   기존 에러 처리로 떨어지는 식으로 구현.

2. **TCP 커넥션이 실행 도중 끊기고 재연결되는 경우**: `wire.session()`은
   `connect()` 시점에 `reset_session()`으로 새 커넥션을 강제하지만, 한
   실행(run) 도중 커넥션이 끊기고 urllib3가 투명하게 재연결하면 서버측은
   새 `_Handler` 인스턴스(=새 세션, 캐시 빈 상태)가 된다. 이 경우 클라이언트가
   들고 있던 hash를 서버가 전부 모르는 상태가 되어 **첫 캐시 참조마다
   409가 발생**하지만, 위 1번의 재전송 로직이 그대로 이를 흡수한다 —
   별도 처리 불필요, 자연스럽게 self-healing됨을 구현 시 확인만 하면 됨.

3. **`messages[i].content`가 항상 문자열이라는 전제**: 현재
   `EngineAdapter.build_body()`가 만드는 `content`는 항상 plain string이다
   (멀티모달 등 리스트/객체 형태 content는 이 코드베이스에 없음). leaf
   순회 로직은 "문자열 leaf만 대상"으로 단순화해도 되며, 향후 멀티모달
   content가 추가되면 이 전제를 재검토해야 한다.

4. **측정값(wire_sent) 해석 변화**: `Gateway.send()`가 갖고 있는
   `result.wire_sent`(실제 전송 바이트, AIPT 실험의 핵심 측정값)는 캐싱이
   켜지면 당연히 줄어든다 — 이건 의도된 효과이지만, `X-AIPT-Cache: enable`
   이 켜진 실행과 꺼진 실행을 같은 실험의 "arm 비교"로 묶어서 보면 안 되고
   (캐싱 여부가 곧 새로운 arm이 되어야 함). turn_record에 캐싱 활성화
   여부를 기록해서 이후 비교/분석 시 arm처럼 구분 가능하게 할지는 구현
   단계에서 결정.

## 10. 관련 기존 코드

- `docker/engine_gateway.py` — L7 리버스 프록시(2026-09-01 1차 구현,
  stream/non-stream 분기 + no-op 캐시 훅). 이 설계의 구현 지점.
- `aipt/backends/local_llm/engine_adapter.py` — `build_body()`가
  요청 바디를 만드는 지점(클라이언트 측 구현 후보 지점).
- `aipt/backends/local_llm/gateway.py` — 기존 in-process
  `on_request`/`on_response` 훅(이번 설계와는 다른, 이미 존재하는
  별개의 확장점).
