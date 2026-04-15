# Week 4 Failure Diagnostic Schema (v1)

## 목적
실행 실패를 재현 가능하고 조치 가능한 진단 단위로 정규화한다.

## 스키마 v1.0.0

- `schemaVersion`: `string`, 정적값 `"1.0.0"`.
- `eventType`: `string`, 고정값 `"run_failed"`만 허용.
- `eventId`: `string`, 실패 이벤트 식별자.
- `eventSeq`: `integer`, 실패 이벤트 시퀀스.
- `errorAt`: `string`, ISO8601 timestamp.
- `nodeId`: `string | null`, 실패가 특정 노드에서 발생했으면 노드 ID.
- `failureCode`: `string`, 실패 코드 (`UNKNOWN` 허용).
- `failureCategory`: `string`, 카테고리 (`unknown` 기본).
- `retryable`: `boolean | null`, 재시도 가능성.
- `retryInfo`: `object | null`, 클라이언트/엔진에서 전달된 재시도 메타.
- `rootCause`: `string`, 원인 설명.
- `resolutionHints`: `string[]`, 사용자/운영 가이드.
- `evidenceRefs`: `string[]`, 트레이스/로그 참조 ID 목록.
- `reason`: `string | null`, 기존 필드 호환성.
- `error`: `string | null`, 기존 필드 호환성.

## 매핑 규칙 (History -> failureContext)

- `failureCode` = `payload.failureCode || payload.errorCode || payload.code || "UNKNOWN"`.
- `failureCategory` = `payload.failureCategory || payload.category || "unknown"`.
- `errorAt` = 실패 canonical 이벤트의 `issuedAt`.
- `rootCause` 우선순위:
  - `payload.rootCause`
  - `payload.reason`
  - `payload.error`
  - `payload.message`
  - 값이 없으면 `"failure was not annotated"`.
- `resolutionHints`:
  1. `payload.resolutionHints`가 문자열 배열이면 그대로 사용.
  2. 비어있거나 유효하지 않으면 카테고리별 기본값 사용.
     - `llm`: 모달/모델 상태 확인, 컨텍스트 축소 재시도, 트레이스 보존.
     - `io`: 네트워크/자격증명/쿼터 재확인, 인프라 복구 후 재시도.
     - `state`: 이벤트 순서 재검증, checkpoint 재생성, 상태전환 불변성 점검.
     - `unknown`: trace 수집 + on-call 에스컬레이션.

## 노출 계약

- `GET /api/runs/{run_id}/history` 응답의 `failureContext`는 위 스키마를 따른다.
- 기존 호환 필드(`reason`, `error`)는 유지한다.
- 실패가 없으면 `failureContext = null`.

## 미해결 과제(TODO)
- `evidenceRefs`의 타입 정합성(문자열 ID 형태)을 API 진입점에서 엄격 검증.
- `retryable` 필드의 기본값(미지정 시 false/unknown) 정책 결정.
