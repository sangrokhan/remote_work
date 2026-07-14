#!/usr/bin/env bash
# Pre-merge gate: unit tests + docker build + a mock container that is actually
# driven through its real routes.
#
# Run from token_traffic/:   ./preflight.sh          (or: make preflight)
#                            ./preflight.sh --smoke-only   (skip tests; make smoke)
#
# Nothing here can spend money: the container runs with TRAFFIC_MOCK=1 and no keys,
# so a mistake in this script costs a wasted minute, not a bill.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE=token-traffic
CONTAINER=token-traffic-smoke
VENV=.venv
MOCK_ENV=(-e TRAFFIC_MOCK=1 -e GEMINI_MOCK=1 -e OPENAI_MOCK=1)

SMOKE_ONLY=0
[ "${1:-}" = "--smoke-only" ] && SMOKE_ONLY=1

grn()  { printf '\033[32m%s\033[0m\n' "$*"; }
step() { printf '\n=== %s ===\n' "$*"; }
cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

if [ "$SMOKE_ONLY" -eq 0 ]; then
  step "1/4  Python env + unit tests"
  python3 -m venv "$VENV"
  "./$VENV/bin/pip" -q install -r requirements.txt pytest >/dev/null
  # Count what pytest collects before trusting what it reports. The last lab's gate
  # ran `unittest discover`, collected 1 of ~230 pytest-function tests, and passed.
  collected=$("./$VENV/bin/python" -m pytest tests -q --collect-only 2>/dev/null | grep -c '::' || true)
  if [ "$collected" -lt 1 ]; then
    echo "FAIL: pytest collected $collected tests from tests/ -- collection is broken."
    exit 1
  fi
  echo "collected $collected tests"
  TRAFFIC_MOCK=1 GEMINI_MOCK=1 OPENAI_MOCK=1 "./$VENV/bin/python" -m pytest tests -q
else
  step "1/4  Unit tests skipped (--smoke-only)"
fi

step "2/4  Docker build"
docker build -t "$IMAGE" .

step "3/4  Start mock container"
FREEPORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
cleanup
docker run -d --rm --name "$CONTAINER" -p "$FREEPORT:8080" "${MOCK_ENV[@]}" "$IMAGE" >/dev/null
for _ in $(seq 1 60); do
  curl -sf "http://localhost:$FREEPORT/api/config" >/dev/null 2>&1 && break
  sleep 0.5
done

BASE="http://localhost:$FREEPORT"

step "4/4  Smoke: the current routes, on a mock run"
# The container is driven exactly the way an operator drives it: preflight the
# selection, run it, then take the numbers away as CSV. Each assertion below is a
# regression that would still leave plausible-looking numbers on the screen.
curl -sf "$BASE/api/config" -o /tmp/tt_config.json
curl -sf -X POST "$BASE/api/preflight" -H 'Content-Type: application/json' \
     -d '{"measure":"bytes","turns":2}' -o /tmp/tt_preflight.json
curl -sf -X POST "$BASE/api/run" -H 'Content-Type: application/json' \
     -d '{"measure":"bytes","turns":2}' -o /tmp/tt_run.json

EXEC_ID=$(python3 -c 'import json;print(json.load(open("/tmp/tt_run.json"))["run"]["exec_id"])')
curl -sf "$BASE/api/runs" -o /tmp/tt_runs.json
curl -sf "$BASE/api/runs/$EXEC_ID/records.csv" -o /tmp/tt_records.csv
curl -sf "$BASE/api/runs/$EXEC_ID/summary.csv" -o /tmp/tt_summary.csv

EXEC_ID="$EXEC_ID" python3 - <<'PY'
import csv, json, os

cfg      = json.load(open("/tmp/tt_config.json"))
pre      = json.load(open("/tmp/tt_preflight.json"))
run      = json.load(open("/tmp/tt_run.json"))["run"]
listing  = json.load(open("/tmp/tt_runs.json"))
records  = list(csv.DictReader(open("/tmp/tt_records.csv")))
summary  = list(csv.DictReader(open("/tmp/tt_summary.csv")))
exec_id  = os.environ["EXEC_ID"]

MARKS = ("req_sent_ms", "ttfb_ms", "ttft_ms", "ttlt_ms", "turn_end_ms")
PREP_PHASES = {"cachegen", "setup"}

# The container must know it is mocked, and say so before anything goes out --
# a preflight that reports billable calls in mock mode is how the operator stops
# believing the preflight.
assert cfg["mock"] is True, "container is not in mock mode"
assert pre["ok"] and pre["mock"] is True and pre["billable_calls"] == 0, pre
assert run["mock"] is True, "run not marked mock"
assert not run["summary"]["failures"], run["summary"]["failures"]

# One run, both vendors. The whole point of the rewrite was that a Gemini arm and an
# OpenAI arm are measured in the same run, on the same fixture, or they are not
# comparable at all.
providers = set(run["params"]["providers"])
assert providers == {"gemini", "openai"}, f"expected both providers, got {providers}"
assert {r["provider"] for r in records} == {"gemini", "openai"}, "CSV lost a provider"

# Bytes came off the socket, both directions, on every turn that counts. A zero here
# is the failure mode that still charts: the numbers look like measurements.
steady = [r for r in records if r["phase"] == "steady"]
assert steady, "no steady records"
for r in steady:
    up, down = int(r["wire_sent"] or 0), int(r["wire_recv"] or 0)
    assert up > 0 and down > 0, f"no bytes counted: {r['provider']}:{r['arm']} turn {r['turn']}"
    assert int(r["input_tokens"] or 0) > 0, f"no tokens: {r['provider']}:{r['arm']}"

# The five marks are columns, not an afterthought -- a latency the export drops is a
# latency nobody can audit.
header = records[0].keys()
missing = [m for m in MARKS if m not in header]
assert not missing, f"records.csv is missing marks: {missing}"

# Prep is phased and it is outside the totals. A cache build costs real bytes; billing
# them to the arm's steady traffic is how a cached arm can be made to look expensive
# (or a stateful one cheap) without anyone lying.
prep = [r for r in records if r["phase"] != "steady"]
assert prep, "no prep records: the cached / stateful arms did no setup"
phases = {r["phase"] for r in prep}
assert phases <= PREP_PHASES, f"unexpected prep phases: {phases - PREP_PHASES}"
assert phases, "prep rows carry no phase"

by_key = {f"{r['provider']}:{r['arm']}": r for r in summary}
for key, p in (run["summary"].get("prep") or {}).items():
    row = by_key[key]
    steady_up = sum(int(r["wire_sent"] or 0) for r in steady
                    if f"{r['provider']}:{r['arm']}" == key)
    assert int(row["wire_sent"]) == steady_up, \
        f"{key}: summary wire_sent includes prep ({row['wire_sent']} != {steady_up})"
    assert int(row["prep_calls"]) == p["calls"] > 0, f"{key}: prep not reported"
    assert int(row["prep_wire_sent"]) > 0, f"{key}: prep cost nothing?"
    assert int(row["turns"]) == len([r for r in steady
                                     if f"{r['provider']}:{r['arm']}" == key]), \
        f"{key}: prep rows counted as turns"

# The mock run is in the mock bucket and nowhere else. 122 synthetic runs sitting in
# the live list, indistinguishable from paid ones, is the accident this asserts away.
live = [r["exec_id"] for r in listing["runs"]]
mocked = [r["exec_id"] for r in listing["mock_runs"]]
assert exec_id in mocked, f"{exec_id} is not in the mock bucket"
assert exec_id not in live, f"{exec_id} leaked into the live list"
assert all(r["mock"] for r in listing["mock_runs"]), "mock bucket holds an unflagged run"

print(f"  routes OK  {len(run['summary']['keys'])} arms, {len(steady)} steady turns, "
      f"{len(prep)} prep call(s) in phases {sorted(phases)}")
print(f"  mock run {exec_id} is in the mock bucket ({len(mocked)}) and not the live "
      f"list ({len(live)})")
PY

grn $'\nPREFLIGHT PASSED'
