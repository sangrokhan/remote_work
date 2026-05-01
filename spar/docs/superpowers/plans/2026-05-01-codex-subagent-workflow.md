# Codex CLI Headless Subagent Workflow

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Claude가 계획만 담당하고, 구현·테스트·리뷰는 Codex CLI가 headless로 수행하도록 hook 기반 워크플로우를 구성한다.

**Architecture:** Claude가 `.claude/codex-task.txt`에 태스크 설명을 쓰면 PostToolUse hook이 감지해 `codex exec` headless로 실행한다. Codex 완료 후 결과가 `.claude/codex-output.log`에 저장되고 Stop hook이 Claude를 재기동해 최종 검증을 요청한다.

**Tech Stack:** `@openai/codex` CLI (already installed: `/home/han/.npm-global/bin/codex`), bash hooks, Claude Code PostToolUse/Stop hooks

---

## File Map

| 파일 | 역할 |
|------|------|
| `.claude/hooks/dispatch-codex.sh` | 새로 생성 — codex-task.txt 감지 후 headless 실행 |
| `.claude/hooks/await-codex.sh` | 새로 생성 — Stop hook, codex 완료 결과 present |
| `.claude/settings.json` | 수정 — PostToolUse hook 추가 |
| `.claude/.gitignore` | 새로 생성 또는 수정 — task/output 파일 gitignore |

신호 파일 (runtime only, gitignored):
- `.claude/codex-task.txt` — Claude가 쓰는 태스크 설명
- `.claude/codex-output.log` — Codex 실행 결과
- `/tmp/spar-codex-running` — 실행 중 센티넬

---

### Task 1: Codex headless 동작 검증

**Files:**
- 없음 (검증만)

- [ ] **Step 1: codex 버전 및 인증 상태 확인**

```bash
codex --version
codex whoami 2>/dev/null || echo "NOT_LOGGED_IN"
```

Expected: 버전 출력, 로그인 상태 확인. `NOT_LOGGED_IN`이면 Step 2로.

- [ ] **Step 2: 필요 시 codex login**

```bash
codex login
```

브라우저 또는 터미널 흐름으로 인증. 완료 후 `codex whoami` 재확인.

- [ ] **Step 3: headless exec 테스트**

```bash
cd /home/han/.openclaw/workspace/remote_work/spar
codex exec "현재 디렉토리의 Python 파일 목록을 출력해줘" --approval-mode full-auto 2>&1 | head -20
```

Expected: 오류 없이 파일 목록 출력. `--approval-mode full-auto` 플래그 작동 확인.

- [ ] **Step 4: approval-mode 옵션 확인**

```bash
codex exec --help 2>&1 | grep -E "approval|auto|quiet|model"
```

Expected: 사용 가능한 플래그 목록 확인. `full-auto` 외 다른 옵션 확인.

---

### Task 2: dispatch-codex.sh 훅 작성

**Files:**
- Create: `.claude/hooks/dispatch-codex.sh`

- [ ] **Step 1: 훅 스크립트 작성**

```bash
cat > /home/han/.openclaw/workspace/remote_work/spar/.claude/hooks/dispatch-codex.sh << 'EOF'
#!/bin/bash
# PostToolUse: codex-task.txt 쓰여질 때 Codex CLI headless 실행

FILE=$(jq -r '.tool_input.file_path // empty' 2>/dev/null)
[[ -z "$FILE" ]] && exit 0
[[ "$FILE" != *".claude/codex-task.txt" ]] && exit 0

TASK_FILE="$FILE"
OUTPUT_LOG="$(dirname "$FILE")/codex-output.log"
SENTINEL="/tmp/spar-codex-running"
PROJECT_ROOT="$(dirname "$(dirname "$FILE")")"

# 이미 실행 중이면 스킵
[[ -f "$SENTINEL" ]] && {
  printf '{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"⏳ Codex 이미 실행 중 — /tmp/spar-codex-running 존재"}}'
  exit 0
}

TASK=$(cat "$TASK_FILE" 2>/dev/null)
[[ -z "$TASK" ]] && exit 0

touch "$SENTINEL"

# Codex headless 실행 (백그라운드)
(
  cd "$PROJECT_ROOT"
  {
    echo "=== Codex Task ==="
    echo "$TASK"
    echo "=== Codex Output ($(date)) ==="
    codex exec "$TASK" --approval-mode full-auto 2>&1
    echo "=== Exit: $? ==="
  } > "$OUTPUT_LOG"
  rm -f "$SENTINEL"
  # Stop hook이 감지할 완료 센티넬
  touch /tmp/spar-codex-done
) &

printf '{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"🚀 Codex 실행 시작 — %s\n결과: %s"}}\n' "$TASK_FILE" "$OUTPUT_LOG"
EOF
chmod +x /home/han/.openclaw/workspace/remote_work/spar/.claude/hooks/dispatch-codex.sh
```

- [ ] **Step 2: 스크립트 문법 확인**

```bash
bash -n /home/han/.openclaw/workspace/remote_work/spar/.claude/hooks/dispatch-codex.sh
echo "Exit: $?"
```

Expected: `Exit: 0` (문법 오류 없음)

---

### Task 3: await-codex.sh 훅 작성

**Files:**
- Create: `.claude/hooks/await-codex.sh`

Codex 완료 센티넬(`/tmp/spar-codex-done`)을 Stop hook에서 감지해 Claude를 재기동, 결과 파일을 보여준다.

- [ ] **Step 1: await 훅 스크립트 작성**

```bash
cat > /home/han/.openclaw/workspace/remote_work/spar/.claude/hooks/await-codex.sh << 'EOF'
#!/bin/bash
# Stop (asyncRewake): Codex 완료 후 결과 보고 + Claude 최종 검증 요청

DONE_SENTINEL="/tmp/spar-codex-done"
RUNNING_SENTINEL="/tmp/spar-codex-running"
OUTPUT_LOG="/home/han/.openclaw/workspace/remote_work/spar/.claude/codex-output.log"

# Codex 완료 센티넬 없으면 pass-through (기존 run-tests.sh로)
[[ ! -f "$DONE_SENTINEL" ]] && exit 0

# 아직 실행 중이면 대기 안내
[[ -f "$RUNNING_SENTINEL" ]] && {
  printf '⏳ Codex 아직 실행 중입니다. 잠시 후 다시 Stop을 호출하세요.'
  exit 2
}

rm -f "$DONE_SENTINEL"

if [[ -f "$OUTPUT_LOG" ]]; then
  printf '🤖 Codex 작업 완료. 결과를 검토하고 최종 검증을 수행하세요:\n\n'
  cat "$OUTPUT_LOG"
  printf '\n\n---\n⚡ Claude: 위 결과를 검토하고, 테스트 통과 여부와 코드 품질을 확인한 후 커밋 여부를 결정하세요.'
else
  printf '⚠️ Codex 완료됐으나 출력 파일 없음: %s' "$OUTPUT_LOG"
fi

exit 2
EOF
chmod +x /home/han/.openclaw/workspace/remote_work/spar/.claude/hooks/await-codex.sh
```

- [ ] **Step 2: 문법 확인**

```bash
bash -n /home/han/.openclaw/workspace/remote_work/spar/.claude/hooks/await-codex.sh
echo "Exit: $?"
```

Expected: `Exit: 0`

---

### Task 4: settings.json에 훅 등록

**Files:**
- Modify: `.claude/settings.json`

현재 settings.json:
```json
{
  "hooks": {
    "PostToolUse": [
      { "matcher": "Write|Edit", "hooks": [{ "type": "command", "command": "bash .../check-tests.sh", ... }] }
    ],
    "Stop": [
      { "hooks": [{ "type": "command", "command": "bash .../run-tests.sh", "asyncRewake": true, ... }] }
    ]
  }
}
```

- [ ] **Step 1: settings.json 업데이트**

`dispatch-codex` 훅은 `Write` 매처에서 `codex-task.txt` 파일만 반응한다. `await-codex` 훅은 Stop에 추가한다. **await-codex를 run-tests보다 먼저** 배치해야 codex 완료 시 우선 처리된다.

최종 settings.json:
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "bash /home/han/.openclaw/workspace/remote_work/spar/.claude/hooks/check-tests.sh",
            "statusMessage": "테스트 파일 확인 중..."
          },
          {
            "type": "command",
            "command": "bash /home/han/.openclaw/workspace/remote_work/spar/.claude/hooks/dispatch-codex.sh",
            "statusMessage": "Codex 디스패치 확인 중..."
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash /home/han/.openclaw/workspace/remote_work/spar/.claude/hooks/await-codex.sh",
            "asyncRewake": true,
            "rewakeSummary": "Codex 완료 결과",
            "statusMessage": "Codex 완료 확인 중..."
          },
          {
            "type": "command",
            "command": "bash /home/han/.openclaw/workspace/remote_work/spar/.claude/hooks/run-tests.sh",
            "asyncRewake": true,
            "rewakeSummary": "테스트 실행 결과",
            "statusMessage": "pytest 실행 중..."
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 2: settings.json 파일 실제 수정**

Edit 툴로 `.claude/settings.json`을 위 내용으로 교체.

- [ ] **Step 3: JSON 유효성 확인**

```bash
python3 -m json.tool /home/han/.openclaw/workspace/remote_work/spar/.claude/settings.json > /dev/null && echo "OK"
```

Expected: `OK`

---

### Task 5: .claude/.gitignore 설정

**Files:**
- Create/Modify: `.claude/.gitignore`

- [ ] **Step 1: .gitignore 작성**

```bash
cat > /home/han/.openclaw/workspace/remote_work/spar/.claude/.gitignore << 'EOF'
codex-task.txt
codex-output.log
EOF
```

- [ ] **Step 2: 커밋**

```bash
cd /home/han/.openclaw/workspace/remote_work/spar
git add .claude/hooks/dispatch-codex.sh .claude/hooks/await-codex.sh .claude/settings.json .claude/.gitignore
git commit -m "feat(hooks): Codex CLI headless subagent workflow

Claude writes .claude/codex-task.txt → PostToolUse dispatches codex exec
headlessly → Stop hook presents output for Claude final verification."
```

---

### Task 6: 종단간 테스트

**Files:**
- 없음 (기존 코드에 테스트)

- [ ] **Step 1: 테스트용 태스크 파일 작성**

Claude가 실제로 사용하는 방식 시뮬레이션:

```bash
cat > /home/han/.openclaw/workspace/remote_work/spar/.claude/codex-task.txt << 'EOF'
src/spar/llm/registry.py의 get_client 함수에 docstring을 추가해줘.
docstring은 한국어로, 함수 역할과 파라미터를 설명해야 해.
변경 후 pytest tests/llm/test_registry.py -q 실행해서 기존 테스트가 통과하는지 확인해줘.
EOF
```

- [ ] **Step 2: hook 트리거 확인 (실제는 Claude가 Write 툴로 파일 작성)**

```bash
# hook 직접 실행 시뮬레이션
echo '{"tool_input":{"file_path":"/home/han/.openclaw/workspace/remote_work/spar/.claude/codex-task.txt"}}' \
  | bash /home/han/.openclaw/workspace/remote_work/spar/.claude/hooks/dispatch-codex.sh
```

Expected: `{"hookSpecificOutput":...}` JSON 출력 + `/tmp/spar-codex-running` 생성

- [ ] **Step 3: 실행 완료 대기 후 결과 확인**

```bash
# codex 완료까지 대기 (최대 2분)
for i in $(seq 1 24); do
  [[ ! -f /tmp/spar-codex-running ]] && break
  sleep 5
done
cat /home/han/.openclaw/workspace/remote_work/spar/.claude/codex-output.log
```

Expected: Codex 실행 로그 + docstring 추가 결과 출력

- [ ] **Step 4: await hook 시뮬레이션**

```bash
# /tmp/spar-codex-done 있는 경우 테스트
touch /tmp/spar-codex-done
bash /home/han/.openclaw/workspace/remote_work/spar/.claude/hooks/await-codex.sh
echo "Exit: $?"
```

Expected: 결과 로그 출력 + `Exit: 2` (asyncRewake 트리거)

---

## 사용 워크플로우 요약

```
[Claude]  계획 작성 → docs/superpowers/plans/*.md
[Claude]  구현 태스크를 .claude/codex-task.txt에 Write
[Hook]    dispatch-codex.sh 감지 → codex exec headless 실행 (백그라운드)
[Codex]   구현 + pytest + 셀프리뷰 + 커밋
[Hook]    Stop → await-codex.sh → Claude 재기동
[Claude]  codex-output.log 검토 + 최종 검증 + 커밋 승인
```

**Codex에 전달할 태스크 작성 형식 (Claude 참고):**
```
<구현 내용 명확히>
변경 후 pytest <test_path> -q 실행해서 테스트 통과 확인.
문제 있으면 수정 후 재실행.
완료 후 git commit -m "<message>" 실행.
```
