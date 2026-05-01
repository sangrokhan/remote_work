#!/bin/bash
# SubagentStart: 서브에이전트 태스크를 Codex CLI headless로 라우팅

PAYLOAD=$(cat)
PROMPT=$(echo "$PAYLOAD" | jq -r '.prompt // .task // empty' 2>/dev/null)
[[ -z "$PROMPT" ]] && exit 0  # 프롬프트 없으면 pass-through

PROJECT_ROOT="/home/han/.openclaw/workspace/remote_work/spar"
OUTPUT_LOG="$PROJECT_ROOT/.claude/codex-output.log"
PROMPT_FILE="/tmp/spar-codex-prompt.txt"

printf '%s' "$PROMPT" > "$PROMPT_FILE"

cd "$PROJECT_ROOT"

# 1단계: Codex headless 실행
{
  echo "=== SubagentStart Task ($(date)) ==="
  cat "$PROMPT_FILE"
  echo ""
  echo "=== [Codex] Output ==="
  codex exec --approval-mode full-auto < "$PROMPT_FILE" 2>&1
  echo "=== Codex Exit: $? ==="
} > "$OUTPUT_LOG" 2>&1

CODEX_EXIT=$?

# 토큰 소진 여부 판단 (exit 비정상 + 관련 메시지)
TOKEN_EXHAUSTED=0
if [[ $CODEX_EXIT -ne 0 ]]; then
  grep -qiE "quota|rate.limit|429|token.*limit|context.*length|insufficient_quota|out of token" "$OUTPUT_LOG" && TOKEN_EXHAUSTED=1
fi

# 2단계: Codex 토큰 소진 시 Gemini로 fallback
if [[ $TOKEN_EXHAUSTED -eq 1 ]]; then
  {
    echo ""
    echo "=== [Gemini Fallback] Codex 토큰 소진 → Gemini 실행 ($(date)) ==="
    cat "$PROMPT_FILE" | gemini --approval-mode yolo -p "다음 태스크를 수행해줘:" 2>&1
    echo "=== Gemini Exit: $? ==="
  } >> "$OUTPUT_LOG" 2>&1
fi

FINAL_EXIT=$?
rm -f "$PROMPT_FILE"

RESULT=$(cat "$OUTPUT_LOG")
RESULT_JSON=$(printf '%s' "$RESULT" | jq -Rs .)

# 서브에이전트 블록 + Codex 결과를 additionalContext로 주입
printf '{"decision":"block","reason":"Codex CLI가 처리함 — .claude/codex-output.log 참조","hookSpecificOutput":{"hookEventName":"SubagentStart","additionalContext":%s}}' "$RESULT_JSON"
exit 2
