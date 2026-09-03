# AIPT 불일치 항목 처리 계획 — 2026-09-02

`docs/audit-2026-09-02/INDEX.md`의 불일치 12건에 대해 주인님과 순차 질의로
확정한 처리방안. **이 문서는 계획만 기록 — 실행은 각 Task 승인 후 별도
세션에서 착수한다.**

## 범주별 요약

| 범주 | 건수 | 항목 |
|---|---|---|
| 문서만 개정(코드 변경 없음) | 8건 | #1, #2, #6, #7, #9, #10, #12, (#11은 #3에 흡수) |
| 코드 변경 필요 | 3건 | #3(QUIC 제거), #4(export 필드 추가), #8(offload 통일) |
| 흡수/통합 | 1건 | #11 → #3에 통합 |

---

## Task 목록 (실행 순서 제안)

### T1 [문서] idle-reset client-only 재설계 명시 — #1
- **대상**: DESIGN.md, ARCHITECTURE.md
- **내용**: idle-reset 토글이 client(`web`)에만 적용되는 이유(2026-09-01
  causal 실험 결과)와, 과거 존재했던 서버측 admin 라우트/사이드카가
  2026-09-02 제거된 사실을 본문에 명시적으로 기록.
- **근거**: core.md §4.1, web.md §5.1
- **코드 변경**: 없음

### T2 [문서] Gateway profile 웹 UI 연동(B11) 구현완료 반영 — #2
- **대상**: DESIGN.md, ARCHITECTURE.md
- **내용**: "B11 미구현" 서술을 `routes_gateway.py` 구현·테스트 완료 사실로
  수정. ARCHITECTURE.md 다이어그램의 점선(`-.->`)을 실선(`-->`)으로 변경.
- **근거**: gateway.md §4.1, web.md §5.4
- **코드 변경**: 없음

### T3 [코드+문서] QUIC 관련 기능 전면 제거 — #3, #11 통합
- **대상 삭제 범위**:
  - `aipt/backends/quic_mock/` 디렉토리 전체(backend.py, congestion.py,
    experiment.py, server.py, spike_runner.py, __init__.py)
  - `aipt/core/quic_congestion.py`
  - `aipt/web/routes_run.py`의 QUIC import/분기 코드
  - 웹 UI(프론트엔드)의 QUIC/transport=http3 옵션
  - `docker/Dockerfile.quic_mock_server`, `docker/entrypoint_quic_mock_server.py`,
    docker-compose.yml의 quic-mock-server 서비스 블록
  - tests/backends의 quic_mock 관련 테스트
  - DESIGN.md/ARCHITECTURE.md/MIGRATION.md의 quic_mock 관련 서술 제거·정리
- **근거**: backends.md §4.1, core.md §4.6
- **코드 변경**: 있음 (별도 승인 후 착수)

### T4 [코드] export 계층에 cwnd 신뢰도 필드 추가 — #4
- **대상**: `aipt/export/connection.py`
- **내용**: `CONNECTION_SUMMARY_COLUMNS`에 `interval_reason`,
  `measurement_confidence` 컬럼 추가, `connection_summary_csv()`가 이를
  읽어 반영하도록 수정 (packets.py의 `gap_confidence_summary`와 대칭 조치).
- **근거**: export.md §4.1
- **코드 변경**: 있음 (별도 승인 후 착수)

### T5 [문서] HEALTHCHECK 비대칭 설명 추가 — #6
- **대상**: DESIGN.md 또는 ARCHITECTURE.md (native/docker 관련 절)
- **내용**: `local-llm`에만 HEALTHCHECK가 있고 나머지(web/mock-server/
  gateway/quic-mock-server — 단 quic-mock-server는 T3로 제거 예정이므로
  web/mock-server/gateway만)엔 없는 이유를 명시.
- **근거**: native.md §3.2
- **코드 변경**: 없음

### T6 [문서] MIGRATION.md Phase 1 체크리스트 갱신 — #7
- **대상**: MIGRATION.md
- **내용**: Phase 1 체크리스트(`native/cwnd_monitor.c`, `cwnd.py`,
  `capture.py`, 테스트 이관)를 `[ ]` → `[x]` 완료로 갱신.
- **근거**: native.md §3.3
- **코드 변경**: 없음

### T7 [코드] offload.py 두 API feature-set 통일 — #8
- **대상**: `aipt/core/offload.py`
- **내용**: capture-time API(`Window`, 현재 tso/gso/gro만)와 entrypoint-time
  API(`build_commands`/`apply`, 현재 tso/gso/sg/gro/lro)의 feature-set을
  통일(5개 feature 공통화) + 복원 로직(capture-time의 "복원" vs
  entrypoint-time의 "한번 끄고 유지") 통일. DESIGN.md L38 서술("사실상
  같은 기능, env var 네이밍만 다름")이 실제로 참이 되도록 코드를 맞춘다.
- **근거**: core.md §4.2
- **코드 변경**: 있음 (별도 승인 후 착수, 두 API 통합 방식은 세부 설계
  필요 — 착수 시 Socratic 질의로 세부 확정)

### T8 [문서] §4.7.1 저장 정책 정식 개정 — #9
- **대상**: DESIGN.md §4.7.1
- **내용**: 기존 stale 경고문("2026-09-01 갱신 — public_ai만 영속" 서술이
  stale)을 제거하고, 모든 backend(public_ai/mock/local_llm)의 run이
  `RUN_STORE_DIR`에 디스크 영속화되는 것을 기본 동작으로 정식 서술.
- **근거**: web.md §5.2
- **코드 변경**: 없음

### T9 [문서] docker-compose.yml 헤더 주석 수정 — #10
- **대상**: docker-compose.yml 파일 헤더 주석(L38-40 부근)
- **내용**: "tc netem이 gateway 양쪽에 동일 프로파일 적용" 서술을 최신
  설계(client leg만 비대칭 shaping, backend leg는 고정 baseline)에 맞게
  수정.
- **근거**: gateway.md §4.4
- **코드 변경**: 없음(주석만)

### T10 [문서] native NDJSON 필드 순서 주석 수정 — #12
- **대상**: `aipt/core/cwnd.py`의 `SAMPLE_FIELDS` 관련 주석/문서
- **내용**: "C가 emit하는 순서 그대로"라는 주석을 실제 순서(`rto_us`/
  `ato_us` 위치 차이 반영)에 맞게 수정. 필드 집합 자체는 이미 일치하므로
  기능 변경 없음.
- **근거**: native.md §1.2
- **코드 변경**: 없음(주석만)

---

## 실행 그룹 제안

- **그룹 A (문서만, 저위험)**: T1, T2, T5, T6, T8, T9, T10 — 한 번에
  일괄 처리 가능. dev-task-guidelines의 "오타 수정/문서 갱신" 예외 범주에
  해당하므로 가벼운 delegate_task 단발 처리로 충분.
- **그룹 B (코드 변경 필요)**: T3(QUIC 제거), T4(export 필드), T7(offload
  통일) — 각각 claude-code-enforced-subagent-loop 경로(PRD 인터뷰 →
  progress.md → 승인 → loop 기동)로 개별 착수. T7은 세부 설계(두 API
  통합 방식)에 대한 추가 Socratic 질의 필요.

각 그룹의 착수는 이 계획서 승인 후 별도로 진행한다.
