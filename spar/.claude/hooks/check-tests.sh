#!/bin/bash
# PostToolUse: src/*.py 수정 시 대응 테스트 파일 확인 + 없으면 additionalContext 주입

FILE=$(jq -r '.tool_input.file_path // empty' 2>/dev/null)
[[ -z "$FILE" ]] && exit 0
[[ "$FILE" != *.py ]] && exit 0
[[ "$FILE" == */tests/* ]] && exit 0
[[ "$FILE" != */src/spar/* ]] && exit 0

# 소스 파일 수정됨 — Stop hook 센티넬 생성
touch /tmp/spar-py-modified

# 예상 테스트 경로 도출
# e.g., .../src/spar/llm/client.py → tests/llm/test_client.py
SPAR_RELATIVE="${FILE#*/src/spar/}"
MODULE_DIR=$(dirname "$SPAR_RELATIVE")
MODULE_NAME=$(basename "$SPAR_RELATIVE" .py)

# src/spar/ 이전 경로 = 프로젝트 루트
PROJECT_ROOT="${FILE%/src/spar/*}"
[[ -z "$PROJECT_ROOT" ]] && exit 0

TEST_PATH="$PROJECT_ROOT/tests/$MODULE_DIR/test_${MODULE_NAME}.py"

if [[ ! -f "$TEST_PATH" ]]; then
  RELATIVE_TEST="tests/$MODULE_DIR/test_${MODULE_NAME}.py"
  MSG="⚠️ 테스트 없음: ${RELATIVE_TEST} — 해당 모듈 테스트를 작성하세요."
  printf '{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"%s"}}' "$MSG"
fi
