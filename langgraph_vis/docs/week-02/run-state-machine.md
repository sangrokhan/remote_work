# Week 2 run 상태머신

## 상태 집합
- `queued`
- `running`
- `awaiting_input`
- `completed` (terminal)
- `failed` (terminal)
- `cancelled` (terminal)

## 전이 규칙
- `queued` → `running`, `cancelled`
- `running` → `awaiting_input`, `completed`, `failed`, `cancelled`
- `awaiting_input` → `running`, `failed`, `cancelled`
- `completed` → (no transition)
- `failed` → (no transition)
- `cancelled` → (no transition)

## 정합성 규칙
- `run`의 terminal 상태는 `completed`/`failed`/`cancelled`만 허용한다.
- terminal 상태 이후에는 상태 변경 이벤트를 무시한다.
- `awaiting_input`은 처리 대기 상태이므로 상태 조회/동기화에서 별도 실패로 처리하지 않는다.
