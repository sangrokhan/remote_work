#!/bin/bash
# SubagentStart: 서브에이전트 태스크를 Codex CLI headless로 라우팅

PAYLOAD=$(cat)
PROMPT=$(echo "$PAYLOAD" | jq -r '.prompt // .task // empty' 2>/dev/null)
[[ -z "$PROMPT" ]] && exit 0  # 프롬프트 없으면 pass-through

PROJECT_ROOT="/home/han/.openclaw/workspace/remote_work/spar"
OUTPUT_LOG="$PROJECT_ROOT/.claude/codex-output.log"
PROMPT_FILE="/tmp/spar-codex-prompt.txt"

echo "$PROMPT" > "$PROMPT_FILE"

# Codex headless 실행 (동기)
{
  echo "=== SubagentStart Task ($(date)) ==="
  cat "$PROMPT_FILE"
  echo ""
  echo "=== Codex Output ==="
  cd "$PROJECT_ROOT"
  codex exec "$(cat "$PROMPT_FILE")" --approval-mode full-auto 2>&1
  echo "=== Exit: $? ==="
} > "$OUTPUT_LOG" 2>&1

CODEX_EXIT=$?
rm -f "$PROMPT_FILE"

RESULT=$(cat "$OUTPUT_LOG")
RESULT_JSON=$(printf '%s' "$RESULT" | jq -Rs .)

# 서브에이전트 블록 + Codex 결과를 additionalContext로 주입
printf '{"decision":"block","reason":"Codex CLI가 처리함 — .claude/codex-output.log 참조","hookSpecificOutput":{"hookEventName":"SubagentStart","additionalContext":%s}}' "$RESULT_JSON"
exit 2
