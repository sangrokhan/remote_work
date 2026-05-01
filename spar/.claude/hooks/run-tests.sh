#!/bin/bash
# Stop (asyncRewake): test → 문서 업데이트 확인 → commit+push 순서 강제

SENTINEL="/tmp/spar-py-modified"
[[ ! -f "$SENTINEL" ]] && exit 0
rm -f "$SENTINEL"

cd /home/han/.openclaw/workspace/remote_work/spar

# 1. pytest
RESULT=$(.venv/bin/pytest --tb=short -q 2>&1)
EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
  printf '❌ 테스트 실패 — 수정 후 다시 진행:\n%s' "$RESULT"
  exit 2
fi

# 2. 문서 변경 확인 (prd.md, AGENTS.md, README.md)
DOCS_DIRTY=$(git status --porcelain -- docs/prd.md AGENTS.md README.md | wc -l | tr -d ' ')
if [[ "$DOCS_DIRTY" -eq 0 ]]; then
  printf '✅ 테스트 통과.\n⚠️  문서 업데이트 필요: docs/prd.md, AGENTS.md, README.md 중 변경 없음.\n완료된 Task 체크박스 및 디렉토리 맵을 업데이트한 후 다시 완료하세요.'
  exit 2
fi

# 3. 문서까지 완료 → commit+push 지시
printf '✅ 테스트 통과, 문서 업데이트 확인.\n지금 변경 사항을 커밋하고 push하세요.'
exit 2
