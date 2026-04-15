# Week 2 재동기화 가이드

## SSE 재연결 정책
- 클라이언트는 마지막 적용한 `eventSeq`와 `eventId`를 보관한다.
- 재연결 시 `/api/runs/{runId}/events`로 `fromSeq` 또는 `lastEventId`를 전달해 누락 구간을 재요청한다.
- `fromSeq`는 기준 시퀀스보다 **큰** 이벤트만 재전송한다.
- `lastEventId`는 `eventId`를 의미하며, 내부에서 eventSeq로 변환되어 `fromSeq` 대체값으로 동작한다.
- SSE 스트림(`id:`)은 `eventId`를 전달한다.

## 연결 유지 정책
- heartbeat: 20초
- idle timeout: 90초
- 연결 실패 시 지수적 재시도(backoff) 후 상태 조회(`/api/runs/{runId}/state`)와 비교해 누락 이벤트를 복원한다.

## 정합성 규칙
- 클라이언트는 eventSeq 오름차순으로 정렬하고 `eventId` 중복을 제거한다.
- 이미 적용한 eventSeq 또는 eventId는 멱등 처리해 건너뛴다.
- terminal 상태(`completed`/`failed`/`cancelled`)의 추가 상태 변경은 무시한다.
