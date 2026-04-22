# PRD: 채팅 인터페이스 + 우측 로그 패널 + 중앙 설정 모달 정교화

## 1. 목적
메인 채팅 화면의 기본 흐름은 유지하되, 우측 패널은 로그 조회에만 사용하고 설정은 우측 패널이 아닌 중앙 모달로 분리해 화면 구조를 단순화한다.

## 2. 범위
- 적용 대상: React + Vite + 순수 CSS 기반 현재 화면
- 포함 항목
  - 메인 채팅 인터페이스
  - 우측 25% 패널(로그 전용)
  - 중앙 라운드 설정 모달(블러 배경)
  - 배포 파이프라인 반영
- 제외 항목
  - 실제 LLM API 연동
  - 사용자 인증, 채팅 이력 영구 저장, 권한 관리

## 3. 사용자 시나리오
1. 사용자는 기본 화면에서 대화를 주고받는다.
2. 사용자는 로그가 필요할 때 우측 토글 버튼으로 로그 패널을 열어 상태를 확인한다.
3. 사용자는 설정이 필요할 때 상단의 설정 버튼으로 중앙 모달을 열어 항목을 수정한다.
4. 패널 또는 모달을 닫고 즉시 대화 화면으로 복귀한다.

## 4. UX 목표
- 대화 흐름을 방해하지 않는 동작
- 로그/설정 경로가 서로 충돌하지 않는 정보 구조
- 상태가 명확한 열기/닫기 제어
- 키보드 조작 가능성 유지
- 화면 전체 톤은 `slate & steel` 계열을 기본으로 하고, 액센트는 과하지 않은 연한 하늘색으로 제한적으로 적용
- 파란색은 포인트 액센트로만 제한하고, 기본 면/텍스트/보더는 slate-steel 비중을 우선 적용
- 배경색과 상단 헤더는 기본 우선톤에 맞춰 과도한 블루/채도 제거 후 재정렬
- 블러 오버레이를 활용한 모달 집중도 확보

### 4.1 컬러 팔레트(최종 확정)
- `base-bg`: `#f3f5f8`
- `surface`: `#f7f9fc`
- `surface-soft`: `#eef1f6`
- `surface-deep`: `#ebedf2`
- `surface-elev`: `#ffffff`
- `line`: `#d8deeb`
- `line-soft`: `rgba(129, 139, 153, 0.32)`
- `text`: `#1f2630`
- `text-soft`: `#5f6a78`
- `accent`: `#7f95a9` (약한 블루 포인트)
- `accent-soft`: `#a2b2c5`
- `focus`: `rgba(127, 149, 169, 0.28)`
- `ok`: `#4f8f73`
- `warn`: `#9f8144`
- `error`: `#ad5f6e`

채택 규칙:
- `accent`/`accent-soft`/`focus`는 hover·focus·아이콘 상태 등 매우 제한된 상황에서만 사용
- 배경, 메시지 버블, 입력영역, 패널, 모달은 최대한 `slate & steel` 계열 값으로 통일
- 파란색 노출은 인지 가능한 정도로만 유지하고, UI 면적이 넓은 영역에서는 사용 금지

## 5. 화면 구조
### 5.1 상위 레이아웃
- 헤더
  - 좌측: 페이지/세션 타이틀
  - 우측: 버튼 그룹
    - 로그 패널 토글 버튼은 화면 세로 중앙 고정
    - 버튼은 우측 경계에 반쯤 걸치고, 패널 열림/닫힘에 따라 경계 위치를 맞춰 이동
    - 설정 모달 열기 버튼
    - 로그 패널 토글 버튼(`<>`)
- 메인 대화 영역
  - 메시지 로그 리스트
  - 입력 영역(텍스트 입력 + 전송)

### 5.2 우측 사이드패널(로그 전용)
- 위치: 화면 오른쪽 고정, 25% 너비
- 내용: 시스템 로그 목록만 표시
- 스크롤: 패널 내부 독립 스크롤
- 열림/닫힘: 슬라이드 트랜지션

### 5.3 설정 모달
- 위치: 화면 중앙
- 모양: 라운드 엣지(최소 18px)
- 배경: 반투명 + 블러 처리
- 구성: 닫기 버튼, 기본 설정 입력(예: 응답 모드, 최대 토큰)
- 닫기: 닫기 버튼, 배경 클릭, ESC

## 6. 상태 및 전이
### 6.1 우측 패널 상태
- `closed` (기본): 패널 미노출, 백드롭 없음
- `open`: 패널 노출, 채팅 영역 width만 25% 축소

### 6.2 설정 모달 상태
- `closed` (기본): 모달 미노출
- `open`: 배경 블러 처리와 함께 중앙 모달 노출

### 6.3 채팅 영역 제약
- 패널 오픈/클로즈 시 채팅 영역은 가로 폭만 조정
- 패널은 채팅 내부 레이아웃(정렬, 요소 위치, 높이) 자체를 변경하지 않음

### 6.3 전이 트리거
- 로그 패널 토글 버튼: 패널 `open ↔ closed`
- 패널 닫기 버튼: 패널 `open → closed`
- 설정 버튼: 모달 `closed → open`
- 모달 닫기 버튼/배경 클릭/ESC: 모달 `open → closed`

## 7. 핵심 요구사항
### 7.1 채팅 영역
- 사용자/어시스턴트 말풍선 구분
- 입력창(텍스트 입력)은 채팅 영역 너비의 80%로 제한
- 입력/전송 영역은 하단 가운데 정렬
- 입력창은 기본 3줄 높이, 최대 5줄까지 자동 확장
- 5줄 초과 시 입력창 내부 스크롤로 처리되며 오른쪽 스크롤바는 시각적으로 고정되지 않음
- 입력창 바로 아래에 부담스럽지 않은 크기의 모델 선택 버튼형 드롭다운 배치
- 드롭다운은 과도한 패딩/테두리 없이 compact하게 표시되고, 현재 선택 모델이 한눈에 보이도록 구성
- 전송 버튼은 입력창 기준 아래쪽 오른쪽 정렬된 원형 버튼으로, 꼬리형 상단 화살표 아이콘 사용
- 샘플 응답 동작 유지

### 7.2 로그 패널
- 시간/레벨/메시지로 구성된 로그 항목 표시
- 긴 텍스트도 줄바꿈 유지
- 로그 없음 상태 메시지
- 로그 목록은 단방향 조회만 제공(패널에서 로그 조절 금지)

### 7.3 설정 모달
- 모달 오픈 시 블러 오버레이 표시
- 설정 변경은 모달 내부에서만 수행
- 입력값 검증 및 오류 메시지 표시
- 모달 닫기 동작은 채팅 흐름을 가로막지 않음

## 8. 접근성
- 패널/모달 토글 버튼 라벨 제공
- 로그 패널 버튼: `aria-expanded`, `aria-controls`
- 모달: `role="dialog"`, `aria-modal="true"`
- 키보드 포커스 이동 정책
  - 패널/모달 열림 시 첫 대화형 요소로 이동
  - 닫힘 시 원래 트리거 버튼으로 복귀
- `prefers-reduced-motion: reduce` 대응
- 버튼 배치 시 세로 중심 정렬 기준(`align-items: center`) 준수

## 9. 구현 계획
1. `isPanelOpen`과 `isSettingsOpen` 분리 관리
2. 로그 패널은 조회 전용으로 UI 정리
3. 설정 모달 오픈/클로즈 및 블러 배경 구현
4. 트리거/백드롭/ESC 이벤트 통합
5. 접근성 속성 점검 후 배포

## 10. 완료 기준(DoD)
- 로그 패널이 열리고 닫히며 채팅 폭이 25% 기준으로 정상 동작
- 패널 상태 변화는 채팅 영역 크기(`width`) 변화만 유발하고, 위치/패딩/정렬 구조는 유지됨
- 설정 모달이 중앙 라운드 형태로 블러 배경과 함께 열림
- 로그 패널과 설정 모달의 제어가 서로 충돌 없이 작동
- `docker compose up --build` 배포 후 `http://localhost:10000` 접속 가능
- 우측 패널 토글 버튼이 헤더 세로 중앙(상하 중앙) 정렬 상태로 표시됨
- 우측 패널 토글 버튼이 화면 오른쪽 경계에 50% 정도 겹쳐 보이도록 배치됨
- 우측 패널에서 workflow를 텍스트가 아닌 박스형 노드/엣지 그래프로 렌더

## 11. 작업 가이드 (필수)
- 작업 단위별 필수 진행 순서는 다음을 따른다.
  1. 사용자 요청 반영 사항을 `prd.md`에 기록한다.
  2. 구현/수정 작업을 수행한다.
  3. 수정 완료 후 `prd.md`에서 해당 항목의 결과를 검토하고 상태를 갱신한다.
  4. `docker compose up --build -d` 실행
  5. `docker compose ps`로 서비스 상태가 `Up`인지 확인
- 최종 검증 URL: `http://localhost:10000`

## 12. 백엔드 이전 및 Docker 실행 환경 (요청 반영)

- 작업 반영: `langgraph_vis/python_backend`의 Python 백엔드 코드를 루트 `backend/`로 이관
- 반영 항목
  - `backend/stategraph_workflow.py`
  - `backend/graph_schema.py`
  - `backend/app/main.py`
  - `backend/requirements.txt`
  - `backend/Dockerfile`
  - `docker-compose.yml`에 `workflow-api` 서비스 추가 (`10001:8000` 노출)
- 완료 기준(임시)
  - `docker compose up --build -d`
  - `docker compose ps`에서 `chat-front`, `workflow-api` 상태 확인
  - `http://localhost:10001/health`, `http://localhost:10001/graph` 응답

## 13. 현재 우선 반영 범위: 패널 상태 + 그래프 구성 조회

- 사용자 요청 반영: 사이드 패널의 `open/closed` 상태를 유지 관리하고, 패널이 `open`일 때만 workflow graph를 조회하도록 수정
- run 관련 기능 제거
  - 백엔드 `run` API 삭제
  - 프론트 패널에서 실행/스트리밍 관련 UI/로직 제거
- 적용
  - 프론트 상태에서 `isPanelOpen` 변화 감시
  - `isPanelOpen === true`일 때 `workflow` Graph API 호출
    - 기본 호출: `http://localhost:10001/graph`
    - 프록시/환경변수 확장 고려 시 `VITE_WORKFLOW_GRAPH_URL` 사용 가능
  - 패널은 조회 전용 `워크플로우 구성` 표시 영역으로 운영

## 14. 우측 패널 그래프 시각화 + 의존성 반영 (요청 반영)

- 사용자 요청 반영:
  - 패널에서 텍스트 출력이 아닌 박스 형태의 workflow 그래프 뷰 렌더링
  - 패널 열림 상태에서만 API 호출(기존 상태관리 정책 유지)
  - run 관련 기능은 제거된 상태 유지
- 구현 항목:
  - `package.json`에 `cytoscape` 의존성 추가
  - `/graph` 응답(`nodes`, `edges`)을 Cytoscape 엘리먼트로 변환해 패널에 박스 노드/엣지 렌더링
- 노드 시작/종료 지점은 시각적으로 구분되는 스타일 적용
- 조건(Condition) 값은 현재 뷰에서 표시하지 않음(선 라벨 비노출)
- langgraph_vis 기반 색상 팔레트를 적용해 노드별 박스 색상 구분:
    - planner: `#d7ecff` / `#68a8ee` / `#12365f`
    - executor: `#d7f4dd` / `#7dcf90` / `#1a4f2f`
    - refiner: `#fff1c7` / `#e2be5e` / `#5e4b17`
    - synthesizer: `#f5d9fc` / `#c78ce0` / `#5d2e69`
    - start: `#e7e7f8` / `#9ca2df` / `#32366c`
    - end: `#dceeff` / `#5f8bb0` / `#263246`
- 레이아웃은 `dagre` + `rankDir: TB`(상단→하단)으로 고정
- 노드 기본 박스 크기(`min-width/min-height`) 및 텍스트 폰트 크기(`font-size`) 확대

### 14.1 최근 반영(추가)

- 엣지 텍스트 라벨 제거
- 시작/종료(`__start__`, `__end__`) 노드 박스는 다시 표시
- 노드 박스 라벨에서 `__`로 둘러싼 표기(예: `__start__`)는 표시 텍스트에서 제거하고 `START/END`로 처리

### 15. 백엔드 WebSocket 연결 대기 엔드포인트(요청 반영)

- 백엔드 FastAPI에 `/ws/connect` WebSocket 엔드포인트 추가
  - 클라이언트 연결 시 `connected` 메시지로 준비 상태 응답
  - `graph|get_graph|refresh` 수신 시 `/graph`와 동일한 워크플로우 데이터 반환
  - `ping` 수신 시 `pong` 응답
  - `close` 수신 시 정상 종료

### 16. 프론트 WebSocket 연결 반영(요청 반영)

- 프론트 사이드에서 `isPanelOpen` 상태와 연동해 `/ws/connect`에 WebSocket으로 연결
- 연결 시 `get_graph` 메시지 송신 후 `type: graph` 응답을 받아 그래프 상태 갱신
- 20초 간격 `ping` 전송으로 연결 유지
- WebSocket 오류/비정상 종료 시 REST `/graph`로 폴백 조회
- 배포 가이드:
  - `docker compose up --build -d`
  - `docker compose ps`로 `chat-front`, `workflow-api` 상태 확인

### 17. WebSocket 주소 설정 파일 추가(요청 반영)

- 루트 `.env` 파일 추가
  - `VITE_WORKFLOW_WS_URL`: WebSocket 접속 주소(예: `ws://localhost:10001/ws/connect`)
  - `VITE_WORKFLOW_GRAPH_URL`: REST Graph 조회 주소(예: `http://localhost:10001/graph`)

### 18. 메시지 전송→워크플로우 실행 + 스트리밍 응답 반영(요청 반영)

- 백엔드 WebSocket 메시지 프로토콜 확장
  - `run_workflow` 명령 수신 시 `run_id` 기반 실행
  - 각 노드 `실행됨`/완료 이벤트를 `workflow_started`, `workflow_event`, `workflow_complete`, `workflow_error`로 스트리밍 전송
- 프론트 동작
  - 전송 버튼/Enter로 `run_workflow` 메시지 전송
  - `run_id`를 메시지와 함께 관리해 같은 실행은 동일 assistant bubble에 누적 출력
  - `workflow_complete/error` 수신 시 해당 bubble 상태 갱신

### 19. WebSocket 연결 실패 대응(요청 반영)

- 사용자 이슈 반영: "웹소켓이 닫혀 있음" 현상 대응
- 진단 결과: 백엔드 컨테이너에서 `uvicorn` 기본 설치만으로 WebSocket 업그레이드 라이브러리가 없어 `/ws/connect` 핸드셰이크 시 `Unsupported upgrade request`, `No supported WebSocket library detected`, 연결 `404`가 발생
- 조치:
  - `backend/requirements.txt`의 `uvicorn`을 `uvicorn[standard]`로 변경해 WebSocket 런타임(예: `websockets`) 포함
  - Docker 재빌드 후 `workflow-api` 재기동
- 완료 기준:
  - `docker compose up --build -d`
  - `docker compose ps`에서 두 컨테이너 `Up` 확인
  - WebSocket 연결이 정상적으로 열리고 `connected` 수신

### 20. 메시지 스트리밍 표시 UX 개선(요청 반영)

- 사용자 요청 반영: 새 응답 수신 시 메시지가 가려지지 않도록 스크롤을 가장 아래로 자동 이동하고, 스크롤이 필요하지 않을 때는 메시지 영역 우측 스크롤바를 숨김
- 조치:
  - 채팅 메시지 컨테이너에 ref 기반 자동 스크롤 효과 추가 (새 메시지/업데이트 시 `scrollTop = scrollHeight` 강제)
  - 메시지 컨테이너 크기 변화/윈도우 리사이즈시 `scrollHeight > clientHeight` 상태를 계산해 오버플로우 여부를 추적
  - 스크롤바는 오버플로우가 있어도 사용자가 실제 스크롤 동작할 때만 임시 표시되도록 별도 토글 상태 적용
  - CSS를 클래스 기반(`messages-scrollbar-visible`)으로 분기:
    - 기본: 스크롤바 미노출
    - 오버플로우 + 사용자 스크롤 동작 시: 스크롤바 노출
- 완료 기준:
  - 배포 후 새 메시지 도착 시 항상 하단으로 노출
  - 메시지 목록이 한 화면 내에 완전 표시되면 우측 스크롤바 미노출
  - 오버플로우가 있어도 스크롤 동작이 없으면 우측 스크롤바 미노출

### 21. 실행 스트림 노드 하이라이트(요청 반영)

- 사용자 요청 반영: 패널이 열려 있는 상태에서 워크플로우 스트림 수신 중 현재 실행 노드를 사이드 패널 그래프에서 하이라이트
- `langgraph_vis` 동작과 정합성 확인 포인트
  - `langgraph_vis`는 스트림 이벤트에서 노드 식별과 구간(stage)을 받아 UI 노드를 `active` 처리
  - 현재 구현은 `node_started` 또는 `stage: start` 이벤트에서 현재 노드를 하이라이트
- 백엔드 조치
  - 스트림 이벤트에 `name`(노드명) 및 `stage`(start/end) 값을 함께 포함
- 프론트 조치
  - WebSocket 메시지 수신 시 노드별 실행 상태를 `workflowExecutionRef`로 추적
  - 실행 중이면 `wf-active` 클래스 적용 후 `node` 스타일을 강조(`background-color`, `border-color`, `shadow`)
  - 패널 닫힘 시 하이라이트 제거, 패널 재열림 또는 그래프 재렌더링 시 진행중 노드 반영
- 완료 기준
  - 스트림 중 `node_started` 시 해당 노드가 패널에서 즉시 하이라이트
  - `workflow_complete` / `workflow_error` 시 하이라이트 제거

### 22. 워크플로우 그래프 가독성 개선(요청 반영)

- 사용자 요청 반영: 패널에서 보이는 워크플로우 박스가 더 잘 보이도록 크기를 키우고, 엣지 길이를 기본적으로 짧게 조정해 흐름 밀도를 높임
- 조치:
  - Cytoscape 노드 스타일 통일
    - `width`/`height`를 고정해 모든 노드를 동일 크기로 렌더링
    - `width`를 `180`, `height`를 `80`으로 고정
    - `text-max-width`를 `180px`로 조정
    - `padding`을 `12`로 조정
    - 노드 폰트 크기를 `50`으로 확대
  - Dagre 레이아웃 압축
    - `nodeSep` 축소
    - `rankSep` 축소
    - `spacingFactor` 축소
    - 레이아웃 `fit`을 `false`로 변경해 자동 축소를 방지해 노드/폰트 크기 유지
    - `minZoom`/`zoom`을 `1`로 고정해 자동 축소 기반 뷰 변형을 줄임
  - 완료 기준:
  - 워크플로우 박스가 고정 크기로 동일하게 렌더링됨
  - 가로 180 대비 세로 80 비율로 넓은 직사각형 형태 유지
  - 노드 텍스트 폰트가 더 크게 표시됨
  - 엣지 길이가 짧아져 노드 간 거리가 더 밀집됨

### 23. 워크플로우 노드 라벨 경고 정리(요청 반영)

- 원인 반영:
  - Cytoscape 노드 스타일 경고가 반복적으로 발생.
  - 실제 런타임 기준으로 확인된 유효 프로퍼티 기준에 맞춰 `label`을 제거하고 `content` 기반으로 정리.
- 조치:
  - `src/App.jsx`에서 노드 라벨 스타일을 `content: data(label)`로 통일.
  - 파서 경고 가능성을 배제하기 위해 `content`를 문자열 함수(`node => node.data('label')`) 형태로 최종 적용.
  - Cytoscape가 무효 판정한 `max-width`, `max-height`, `shadow-*` 값을 제거해 경고를 추가 정리.
  - `wheelSensitivity` 사용자 지정 옵션도 제거해 줌 경고 제거.
- 보조 조치:
  - `langgraph_vis/frontend/workflow_graph_widget.js`도 동일 파싱 기준으로 `content: data(label)` 유지.
- 배포:
  - `docker compose up --build -d`
  - `docker compose ps`로 `chat-front`, `workflow-api` 상태 확인
- 배포 보강:
  - `nginx` 정적 서빙 시 `index.html`는 `Cache-Control`을 `no-store, no-cache`로 강제해 구형 해시 번들 재사용을 차단.
- 완료 기준:
  - 브라우저 콘솔에서 `style property \`content: data(label)\` is invalid` 메시지 미발생

### 24. 워크플로우 노드 글씨가 작게 보이는 문제 수정(요청 반영)

- 사용자 요청 반영: 우측 패널 워크플로우 노드의 글씨가 `font-size` 설정(30px) 대비 훨씬 작게 보이는 문제를 수정.
- 원인 분석:
  - Cytoscape는 노드 라벨을 캔버스 2D 컨텍스트로 렌더링함.
  - 기존 스타일이 `'font-family': 'inherit'`을 사용했으나, 캔버스 2D의 `ctx.font` 문자열은 `inherit` 키워드를 지원하지 않음.
  - 결과적으로 폰트 문자열 파싱이 실패해 캔버스 기본 폰트(작은 기본값, 약 10px sans-serif)로 fallback되어 `font-size: 30` 설정이 무시되고 텍스트가 작게 렌더링됨.
  - 검증: DevTools로 `font-family`만 `Arial, sans-serif`로 바꾸니 30→40px 변화가 즉시 반영되는 것을 확인.
- 조치:
  - `src/App.jsx`의 노드/엣지/배치 직후 배치 업데이트 3곳에서 `'font-family': 'inherit'`을 명시적인 시스템 폰트 스택으로 교체:
    - `system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`
- 배포:
  - `docker compose up --build -d`
  - `docker compose ps`로 `chat-front`, `workflow-api` 상태 확인
- 완료 기준:
  - 우측 워크플로우 패널의 각 노드 라벨(`START`, `planner`, `executor`, `refiner`, `synthesizer`, `END`)이 180×80 박스 안에서 `font-size: 30` 설정 그대로 크게 보임.

### 25. 노드/폰트 크기 축소(요청 반영)

- 사용자 요청 반영: §24 수정 이후 글자가 과도하게 커서 박스·폰트를 축소.
- 조치(`src/App.jsx`):
  - Cytoscape 노드 스타일 변경
    - `font-size`: `30` → `20`
    - `min-zoomed-font-size`: `30` → `20`
    - `width`/`min-width`: `180` → `100`
    - `height`/`min-height`: `80` → `50`
    - `text-max-width`: `180px` → `100px`
    - `padding`: `12` → `8`
  - 배치 직후 보정 블록의 `font-size` 30 → 20, `text-max-width` 180 → 100 동일 반영.
- 배포:
  - `docker compose up --build -d`
  - `docker compose ps`로 `chat-front`, `workflow-api` 상태 확인
- 완료 기준:
  - 노드 박스 100×50, 라벨 폰트 20px로 렌더링되어 패널 공간 내에서 적정 크기로 표시됨. (→ §27에서 80×40/16px로 재조정)

### 26. 우측 워크플로우 패널 기본 숨김(요청 반영)

- 사용자 요청 반영: 앱 진입 시 우측 워크플로우 패널을 기본적으로 닫힌 상태로 표시.
- 조치:
  - `src/App.jsx`에서 `useState(true)` → `useState(false)`로 `isPanelOpen` 초기값 변경.
  - 토글 버튼(`.log-toggle-btn`)으로 필요 시 열 수 있음.
- 배포:
  - `docker compose up --build -d`
  - `docker compose ps`로 `chat-front`, `workflow-api` 상태 확인
- 완료 기준:
  - 페이지 최초 로드 시 워크플로우 패널이 닫혀 있고, 토글 버튼이 열림 상태로 노출됨.

### 27. 워크플로우 그래프 중앙 정렬 및 박스/폰트 재조정(요청 반영)

- 사용자 요청 반영: 우측 패널 노드가 좌측에 치우쳐 중앙 정렬이 안 되는 문제 수정, 박스/폰트 추가 축소.
- 원인 분석:
  - `align: 'UL'` dagre 옵션 → 그래프를 좌상단 기준으로 배치.
  - `fit: false` + 초기 `pan: {x:0, y:0}` → 레이아웃 완료 후 중앙 이동 없음.
- 조치:
  - `layout`에서 `align: 'UL'` 제거.
  - Cytoscape 초기화 후 `cy.center()` 호출 → 그래프를 뷰포트 중앙으로 이동.
  - 노드 `width`/`height`: `100×50` → `80×40`
  - `font-size`/`min-zoomed-font-size`: `20` → `16`
  - `text-max-width`: `100px` → `80px`
  - `padding`: `8` → `6`
  - 배치 후 batch 블록 동일 반영.
- 배포:
  - `docker compose up --build -d`
  - `docker compose ps`로 `chat-front`, `workflow-api` 상태 확인
- 완료 기준:
  - 노드 박스 80×40, 폰트 16px로 렌더링되며 패널 내 수평 중앙 정렬됨.
