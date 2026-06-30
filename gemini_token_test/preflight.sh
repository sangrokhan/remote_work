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
"./$VENV/bin/pip" -q install flask requests urllib3 >/dev/null
"./$VENV/bin/python" -m unittest discover -s tests

step "2/4  Docker build"
docker build -t "$IMAGE" .

step "3/4  Start mock container"
FREEPORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
docker run -d --rm --name "$CONTAINER" -p "$FREEPORT:8080" -e GEMINI_MOCK=1 "$IMAGE" >/dev/null
for _ in $(seq 1 40); do
  curl -sf "http://localhost:$FREEPORT/" >/dev/null 2>&1 && break
  sleep 0.5
done

step "4/4  Smoke: /run + /inspect"
curl -sf -X POST "http://localhost:$FREEPORT/run" \
  -H 'Content-Type: application/json' -d '{"mode":"stateless","turns":5}' \
| python3 -c 'import sys,json;d=json.load(sys.stdin);t=d["summary"]["totals"];assert d["mode"]=="stateless" and t["tokens"]>0,"run failed";print("  /run OK  mode",d["mode"],"tokens",t["tokens"])'

# allow_private to reach the container'\''s own loopback; wire_recv>0 guards against
# the makefile recv-counter regression.
curl -sf -X POST "http://localhost:$FREEPORT/inspect" \
  -H 'Content-Type: application/json' \
  -d '{"url":"http://127.0.0.1:8080/history","allow_private":true}' \
| python3 -c 'import sys,json;d=json.load(sys.stdin);assert d["ok"],d.get("error");assert d["wire_recv"]>0,"wire_recv==0 (recv counter regression)";print("  /inspect OK  wire_recv",d["wire_recv"])'

grn $'\nPREFLIGHT PASSED'
