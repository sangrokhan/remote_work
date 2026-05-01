# CLAUDE.md — SPAR 프로젝트 규약

> AI 에이전트(Claude Code 등) 전용 운영 지시사항. 상세 규약은 `AGENTS.md` 참조.

---

## 필수: 작업 완료 후 문서 업데이트

**모든 구현 작업 완료 후 반드시 다음 문서를 업데이트할 것:**

1. **`README.md`** — 디렉토리 구조, 현 상태(현 상태 섹션) 반영
2. **`AGENTS.md`** — 디렉토리 맵, 현 단계, 기술 스택 반영
3. **`docs/prd.md`** — 완료/진행 중 Task 체크박스 갱신, 산출물 항목 체크

업데이트 기준:
- 새 모듈/파일 추가 → 디렉토리 맵에 반영
- Task 산출물 파일 생성 → prd.md 해당 체크박스 체크
- Task 완료 → prd.md에 완료 날짜 + merge commit 기록 (Task 1.6 형식 참고)
- 현 단계 변경 → README `현 상태`, AGENTS.md `현 단계` 동시 갱신

---

## 정답 출처

- **PRD**: `docs/prd.md` — Phase/Task 진행 기준
- **규약**: `AGENTS.md` — 컨벤션, 빌드, 디렉토리
- PRD와 코드가 충돌 시 PRD 우선. PRD가 현실과 맞지 않을 때는 PRD를 갱신하고 변경 사유 기록.

---

## 작업 원칙

- 구현 작업은 worktree 분기 후 진행 (`git worktree add .worktrees/<branch>`)
- 한 세션에 한 Phase/Task 집중
- 커밋: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
