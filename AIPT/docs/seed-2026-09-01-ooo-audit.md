# Seed — AIPT ooo 재정의 1차 인터뷰 산출물 (2026-09-01)

**출처**: `ooo`(Ouroboros) 인터뷰 워크플로우로 AIPT 전체를 재정의하는 과정에서
"주인님 머릿속 요구사항 vs 실제 코드/문서 구현 상태"를 전 모듈(core/backends/
gateway/export/web/native) 전수 대조한 결과. 3개 서브에이전트가 실제
`docker compose build/up`, curl 헬스체크, 실제 run 실행까지 동원해 검증.

## 목표 (Restated Goal)

> AIPT의 설계 문서(DESIGN.md)와 실제 구현 사이의 괴리를 코드 레벨로 전수
> 검증해 문서를 최신 상태로 갱신하고, 발견된 갭을 실행 가능한 작업
> 목록으로 남긴다. 이후 AIPT의 모든 개발은 ooo 루프(interview → seed →
> run → evaluate)로 관리한다.

## 검증 방법

- backends(public_ai/mock/local_llm/quic_mock), gateway/export/web,
  native C + DESIGN.md §6 미해결 결정 6개를 3개로 나눠 병렬 서브에이전트
  조사.
- Gateway는 실제 `docker compose build/up`으로 5개 컨테이너를 기동해
  `/gateway/health`, `/gateway/profile` 실호출로 netem 적용 확인.
- local_llm은 실제 GGUF(Qwen2.5-0.5B-Instruct)를 로드한 llama.cpp
  서버에 AIPT 웹 API(`POST /api/run`, backend=local_llm)로 3턴 대화를
  실행해 TTFT(583ms)/wire bytes 실측.

## 확정된 사실 (Facts, verified)

1. DESIGN.md §6의 미해결 설계 결정 6개는 **전부 확정 완료**(코드에 반영됨).
   DESIGN.md에 확정 근거를 각 항목 아래 인라인으로 추가함(2026-09-01).
2. `native/cwnd_monitor.c` ↔ `aipt/core/cwnd.py`의 `track` 명령 프로토콜,
   NDJSON 출력 필드셋(`SAMPLE_FIELDS`)이 완전히 일치.
3. 3-backend(public_ai/mock/local_llm) + Network Gateway(L3 netem) +
   export 3-layer CSV(cwnd/turns/packets)는 실제 기동·실측으로 동작 확인.
4. `local_llm` 백엔드는 스텁이 아니라 실제 vLLM/llama.cpp 엔진과 통신
   가능한 완성된 프록시(`gateway.py`+`engine_adapter.py`) — 2026-09-01
   최초로 end-to-end 검증 완료(그 전까지는 `test_engine_live.py`가
   항상 스킵되어 이 저장소 자체엔 실행 증거가 없었음).

## 확정된 문서 갱신 (이번 세션에서 DESIGN.md에 반영 완료)

- §6 6개 항목에 "확정 (2026-09-01)" 인라인 주석 추가.
- §5.2(신규) "문서-코드 정합성 점검" 절 추가 — 아래 갭 3개 + HEALTHCHECK
  버그 발견 기록.
- §4.7.1에 "2026-09-01 갱신" 경고문 추가 — 실제로는 영속 저장(`data/runs/`)
  중임을 명시(원문은 역사적 기록으로 보존).
- §4.8 Mermaid 다이어그램에 `QuicMockBackend` 노드 추가, `routes_gateway`를
  점선/TODO로 표시.
- `docker/Dockerfile.local_llm`의 HEALTHCHECK를 8080→40080으로 수정
  (실제 재빌드로 `healthy` 전환 검증 완료).

## 남은 작업 (TODO, 이번 세션에선 Seed 기록만 — 실행은 별도 승인 후)

| # | 우선순위 | 작업 | 근거 |
|---|---|---|---|
| T1 | 높음 | **B11: 웹 UI Network Profile 드롭다운 구현** — `aipt/web/routes_gateway.py` 신규(GET/POST `/gateway/profile` 프록시), 실험 폼에 `clean/broadband/3g/satellite/lossy/custom` 드롭다운 추가, `GATEWAY_HOST`/`GATEWAY_PORT`를 실제로 사용하도록 연결 | Gateway 백엔드 API는 완성·실동작하지만 프론트가 없어 사용자가 컨테이너에 직접 curl해야 함 |
| T2 | 중간 | **§4.7.1 저장 정책 문서를 실제 동작(영속화)에 맞춰 정식 개정** — "확정" 절 자체를 다시 쓰거나 새 절로 대체 | 이번엔 경고문만 추가, 근본적으로는 정책 재작성 필요 |
| T3 | 중간 | **quic_mock을 §4.5/§4.8 정식 아키텍처에 편입할지 결정** — 4번째 backend로 승격할지, spike 전용으로 격리 유지할지 | 현재는 §7에만 문서화된 애드온, `RunRequest`/웹 UI에 미연결 |
| T4 | 낮음 | **`test_engine_live.py`를 CI에서 최소 1회는 실행하는 경로 마련** (로컬 GPU/모델 러너, 또는 이번처럼 수동 검증 절차를 문서화) | local_llm 실동작 검증이 이번이 처음이었고 자동화된 회귀 방지가 없음 |
| T5 | 참고 | **핵심 연구 질문(idle-reset → TTFT 실측)은 이번 세션 범위 밖** — 별도 ooo interview로 다룰 것 (주인님 요청: 이번엔 문서 갱신만) | 다음 인터뷰 주제 후보 |

## 다음 단계 제안

- `ooo run` 또는 일반 승인 절차로 T1(웹 UI Gateway 프로필)부터 착수 검토.
- 핵심 연구 질문(TTFT 실측)을 별도 `ooo interview`로 진행.
