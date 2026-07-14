#!/usr/bin/env bash
# Pre-merge gate: unit tests + docker build + mock-container smoke.
# Run from gemini_token_test/:  ./preflight.sh   (or: make preflight)
# Exits non-zero on the first failure. No GCP creds / quota needed (mock mode).
set -euo pipefail
cd "$(dirname "$0")"

IMAGE=gemini-preflight
CONTAINER=gemini-preflight-run
VENV=.venv-preflight

grn() { printf '\033[32m%s\033[0m\n' "$*"; }
step() { printf '\n=== %s ===\n' "$*"; }
cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

step "1/4  Python env + unit tests"
python3 -m venv "$VENV"
"./$VENV/bin/pip" -q install flask requests urllib3 pytest >/dev/null
# The suite is written as pytest functions. `unittest discover` collects only the
# handful that happen to subclass TestCase and reports success for the rest, which
# is worse than not running them at all.
"./$VENV/bin/python" -m pytest tests -q

step "2/4  Docker build"
docker build -t "$IMAGE" .

step "3/4  Start mock container"
FREEPORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
docker run -d --rm --name "$CONTAINER" -p "$FREEPORT:8080" -e GEMINI_MOCK=1 "$IMAGE" >/dev/null
for _ in $(seq 1 40); do
  curl -sf "http://localhost:$FREEPORT/" >/dev/null 2>&1 && break
  sleep 0.5
done

step "4/4  Smoke: /compare"
# The whole experiment through the HTTP surface, in mock mode: every arm answers
# every turn, and each turn carries the byte split and the five marks. A record
# missing one of those is the regression this gate exists to catch -- the numbers
# would still look plausible in the UI.
curl -sf -X POST "http://localhost:$FREEPORT/compare" \
  -H 'Content-Type: application/json' -d '{"turns":2}' \
| python3 -c '
import sys, json
d = json.load(sys.stdin)
arms = d["params"]["arms"]
steady = [r for r in d["records"] if r["phase"] == "steady"]
assert d["mode"] == "comparison" and arms, "no arms ran"
assert not d["summary"]["failures"], d["summary"]["failures"]
marks = ("req_sent_ms", "ttfb_ms", "ttft_ms", "ttlt_ms", "turn_end_ms")
for r in steady:
    assert r["wire_sent"] > 0 and r["wire_recv"] > 0, f"no bytes counted: {r[\"arm\"]}"
    assert all(m in r for m in marks), f"missing marks: {r[\"arm\"]}"
    assert r["input_tokens"] > 0, f"no tokens: {r[\"arm\"]}"
print("  /compare OK ", len(arms), "arms,", len(steady), "measured turns")
'

grn $'\nPREFLIGHT PASSED'
