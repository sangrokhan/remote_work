#!/bin/bash
# Stop (asyncRewake): 센티넬 있을 때만 pytest 실행 후 Claude 재기동

SENTINEL="/tmp/spar-py-modified"
[[ ! -f "$SENTINEL" ]] && exit 0
rm -f "$SENTINEL"

cd /home/han/.openclaw/workspace/remote_work/spar
RESULT=$(.venv/bin/pytest --tb=short -q 2>&1)
EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
  printf '✅ 모든 테스트 통과.\n커밋할까요?'
else
  printf '❌ 테스트 실패 — 커밋 전에 수정 필요:\n%s' "$RESULT"
fi

exit 2
