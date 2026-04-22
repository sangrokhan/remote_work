# 사이드 패널 박스/폰트 크기 설정 비교 정리

## 1) 요약(먼저 읽기)
- `langgraph_vis` 쪽은 사이드 패널이 고정 배치되는 우측 분할 영역(전체 레이아웃의 20% 고정)이며, 노드 박스는 작고 폰트도 상대적으로 작음.
- 현재 폴더의 채팅 앱은 우측 로그 패널을 슬라이드로 열고 닫는 오버레이 구조이며, 워크플로우 노드 박스와 폰트 크기가 크게 설정되어 있음.
- 특히 `font-size`는 `langgraph_vis`(노드 13) 대비 현재 폴더(노드 30)로 크게 확대되어 있음.

## 2) `langgraph_vis`의 사이드 패널 및 박스/폰트 설정

### 사이드 패널
- 대상 파일: `langgraph_vis/frontend/styles.css`
- 패널 컨테이너: `.workflow-canvas-area`
- 핵심값
  - `width: 20vw`
  - `min-width: 260px`
  - `max-width: 20%`
  - `min-height: 100vh`
  - `border-left: 1px solid ...`
  - `padding: 8px`
- 배경/배치
  - `.workflow-page { display: flex; width: 100vw; min-height: 100vh; }`
  - 패널이 좌우 분할 레이아웃의 고정된 일부로 항상 노출됨

### 그래프 박스(Node box)
- 대상 파일: `langgraph_vis/frontend/workflow_graph_widget.js`
- `node` 스타일
  - `width: 150`, `height: 52`
  - `min-width: 150`, `min-height: 52`
  - `padding: "8px 10px"`
  - `text-max-width: 132`
  - `font-size: 13`
  - `font-family: "system-ui, -apple-system, 'Segoe UI', Arial, sans-serif"`
  - `font-weight: 500`
- `edge` 라벨/텍스트 스타일
  - `font-size: 10`

## 3) 현재 폴더의 사이드 패널 및 박스/폰트 설정

### 사이드 패널(로그 패널)
- 대상 파일: `src/styles.css`
- 패널 컨테이너: `.log-panel`
- 핵심값
  - `position: fixed; top: 0; right: 0; height: 100vh; width: var(--panel-width)`
  - `transform: translateX(100%)` (닫힘 상태)
  - `.app-shell.panel-open .log-panel { transform: translateX(0); }` (열림)
  - `padding: 20px; padding-top: 12px`
  - `box-shadow: -22px 0 44px ...`
- 패널 너비 제어
  - `src/App.jsx`에서 `const PANEL_WIDTH = '25%'`
  - `style={{ '--panel-width': PANEL_WIDTH }}`
- 배경
  - `background: var(--bg-surface)`
  - 일반 UI는 오버레이 + 토글 버튼(`.log-toggle-btn`)로 제어

### 로그 패널 내부 폰트/박스
- 대상 파일: `src/styles.css`
  - `h2`(패널 제목): `font-size: 20px`
  - `panel-label`: `font-size: 11px`
  - 상태 메시지 `.panel-state`: `font-size: 12px`
  - 로그 라인 `.log-line`: `font-size: 12px`
- 닫기 버튼
  - `.panel-close`: `font-size: 21px`
- 그래프 영역/컨테이너
  - `.graph-view`: `height: calc(100vh - 190px)`
  - `.workflow-graph`: `width: 100%; height: 100%`

### 워크플로우 노드 박스/폰트
- 대상 파일: `src/App.jsx`
- `node` 스타일
  - `padding: 12`
  - `'font-size': 30`
  - `'min-zoomed-font-size': 30`
  - `text-max-width: '180px'`
  - `width: 180`, `height: 80`
  - `min-width: 180`, `min-height: 80`
  - `'font-weight': 600`
  - `shape: 'round-rectangle'`
- 배치/여백/동작 관련
  - `layout`에 `padding: 20`
  - `fit: false`

## 4) 차이점(우선순위로 정리)

1. 패널 구조
- `langgraph_vis`: 고정 분할 레이아웃(전체 화면 분할) / 항상 표시
- 현재: 화면 위에 고정되는 슬라이드 오버레이 / 열기/닫기 토글

2. 패널 너비 기준
- `langgraph_vis`: `20vw`, `min-width:260px`, `max-width:20%`
- 현재: `var(--panel-width)` = `25%` (상대값 단일)

3. 패널 패딩
- `langgraph_vis`: `8px`
- 현재: `20px` (상단 제외 `12px`)

4. 텍스트 폰트 크기(사이드 패널)
- `langgraph_vis` 텍스트 로그/라벨은 일반 웹 폰트 크기 기본~소형(예: 그래프 노드 라벨 13, 엣지 라벨 10)
- 현재 패널은 제목 20, 라벨 11, 상태/로그 12, 닫기버튼 21 등 상대적으로 더 크게 보이거나 구분이 명확한 타이포 스케일

5. 노드 박스 크기/패딩/폰트
- `langgraph_vis` 노드 박스: `150x52`, 패딩 `8px 10px`, `font-size 13`
- 현재 노드 박스: `180x80`, 패딩 `12`, `font-size 30`, `min-zoomed-font-size 30`
- 현재 노드가 시각적으로 더 크고 가독성이 크게 강화된 설정

6. 보완/주의점
- 현재 버전은 노드 텍스트를 크게 잡아 렌더 공간/줌 동작 제약이 커지므로, 장문 라벨일 경우 `text-max-width` 및 `nodeDimensionsIncludeLabels` 튜닝이 필요할 수 있음.
- `langgraph_vis`는 엣지 라벨 폰트 크기를 명시했지만, 현재 폴더는 엣지 폰트 크기 미지정(기본값 의존)이라 라벨 우선순위 제어가 덜 명시적임.
